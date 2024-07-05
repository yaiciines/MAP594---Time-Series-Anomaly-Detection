import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

import datasets
from positional_embeddings import PositionalEmbedding

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

#This approach creates overlapping sequences of 100 points each. Your model would then learn to generate sequences of 100 points.
#sequence_length determines how many time steps are in each sequence.
#stride allows you to control the overlap between sequences. A stride of 1 means each sequence starts one time step after the previous one, while a larger stride reduces overlap.

# Data Normalization: You might want to add normalization to your TimeSeriesDataset class. This can help with training stability.

# import the time series data and creates sliding windows. The TimeSeriesDataset class handles this by using the sequence_length and stride parameters.
import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, folder_path, sequence_length=100, stride=1, normalize=True):
        self.train_data = np.load(f'{folder_path}/train.npy')
        self.train_timestamp = np.load(f'{folder_path}/train_timestamp.npy')
        self.train_label = np.load(f'{folder_path}/train_label.npy')
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        
        if self.normalize:
            self.mean = np.mean(self.train_data)
            self.std = np.std(self.train_data)
            self.train_data = (self.train_data - self.mean) / self.std
        
    def __len__(self):
        return (len(self.train_data) - self.sequence_length) // self.stride + 1
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        sequence = self.train_data[start_idx:end_idx]
        sequence_labels = self.train_label[start_idx:end_idx]
        return torch.FloatTensor(sequence), self.train_timestamp[start_idx], sequence_labels
    
    def denormalize(self, data):
        if self.normalize:
            return data * self.std + self.mean
        return data

# Example usage:
# dataset = TimeSeriesDataset(folder_path='path/to/data', normalize=True)
"""When you're using the normalized data and need to convert it back to the original scale (e.g., for visualization or evaluation), you can use the denormalize method:
original_scale_data = dataset.denormalize(normalized_data)"""

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


#Model Architecture: Ensure your MLP class is properly implemented for time series data. You might want to consider using a more specialized architecture like a Temporal Convolutional Network (TCN) or a Transformer.
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, hidden_layers=3, emb_size=128,
                 time_emb="sinusoidal", input_emb="sinusoidal"):
        super().__init__()
        
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        
        concat_size = len(self.time_mlp.layer) + len(self.input_mlp.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, input_size))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x_emb = self.input_mlp(x)
        t_emb = self.time_mlp(t)
        
        print(f"x shape: {x.shape}")
        print(f"t shape: {t.shape}")
        print(f"x_emb shape: {x_emb.shape}")
        print(f"t_emb shape: {t_emb.shape}")
        
        # Broadcast t_emb across the sequence length dimension
        t_emb = t_emb.unsqueeze(1).expand(-1, x_emb.size(1), -1)
        
        x = torch.cat((x_emb, t_emb), dim=-1)
        return self.joint_mlp(x)
    
class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps

def denoise_image(model_path, image):
    # Load the pretrained model

    model = MLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Add noise to the image
    noise = torch.randn_like(image)
    noisy_image = image + noise
    
    # Denoise the image using the pretrained model
    denoised_image = model(noisy_image)
    
    return denoised_image

def detect_anomalies(model, data_loader, noise_scheduler, threshold):
    model.eval()
    anomalies = []
    with torch.no_grad():
        for sequence, timestamp, labels in data_loader:
            # Add noise
            noise = torch.randn_like(sequence)
            noisy_sequence = noise_scheduler.add_noise(sequence, noise, torch.tensor([noise_scheduler.num_timesteps - 1]))
            
            # Denoise
            for t in reversed(range(noise_scheduler.num_timesteps)):
                t_batch = torch.full((sequence.shape[0],), t, device=sequence.device)
                model_output = model(noisy_sequence, t_batch)
                noisy_sequence = noise_scheduler.step(model_output, t, noisy_sequence)
            
            # Compute reconstruction error
            mse = F.mse_loss(noisy_sequence, sequence, reduction='none').mean(dim=1)
            anomalies.extend((mse > threshold).tolist())
    
    return anomalies

def visualize_sample(sample, epoch, step, save_path):
    plt.figure(figsize=(10, 5))
    for i in range(min(10, sample.shape[0])):  # Plot up to 10 series
        plt.plot(sample[i].cpu().numpy())
    plt.title(f'Generated Time Series - Epoch {epoch}, Step {step}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig(save_path)
    plt.close()
    
def main(config):
    sequence_length = 100
    
    # Set up dataset and dataloader
    dataset = TimeSeriesDataset(config.data_path)
    
    # Shuffling: The dataloader is currently not shuffling the data (shuffle=False). For training, it's usually better to shuffle to prevent the model from learning sequence order. I've changed this to shuffle=True in the artifact.
    # Dropping last batch: You might want to set drop_last=True in the DataLoader to ensure all batches are of the same size. This can be important for some operations.
    
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True , drop_last=True)
    # Adjust model initialization
    model = MLP(
        input_size=sequence_length,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding)

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )
    
    # the training loop
    global_step = 0
    frames = []
    losses = []
    print("Training model...")
    # Training over windows: The code does train over these windows. Each batch contains multiple sequences of length sequence_length.

    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, (batch, timestamps, labels) in enumerate(tqdm(dataloader)):
            noise = torch.randn(batch.shape)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],)
            ).long()

            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
            
        progress_bar.close()
        
        # Evaluation and visualization
        """if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            model.eval()
            # Change this line:
            sample = torch.randn(config.eval_batch_size, sequence_length)  # Use sequence_length instead of dataset[0][0].shape[0]
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
                with torch.no_grad():
                    residual = model(sample, t)
                sample = noise_scheduler.step(residual, t[0], sample)
            
            save_path = os.path.join(config.output_dir, f"sample_epoch_{epoch}.png")
            visualize_sample(sample, epoch, i, save_path)"""


    print("Saving model...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the folder containing the dataset")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=10)
    parser.add_argument("--experiment_name", type=str, default='NonName')

    config = parser.parse_args()
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    main(config)