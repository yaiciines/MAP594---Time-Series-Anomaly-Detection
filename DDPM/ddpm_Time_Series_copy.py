import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import os

from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

from positional_embeddings import PositionalEmbedding

from tqdm import tqdm

#This approach creates overlapping sequences of 100 points each. Your model would then learn to generate sequences of 100 points.
#sequence_length determines how many time steps are in each sequence.
#stride allows you to control the overlap between sequences. A stride of 1 means each sequence starts one time step after the previous one, while a larger stride reduces overlap.

# Data Normalization: You might want to add normalization to your TimeSeriesDataset class. This can help with training stability.

# import the time series data and creates sliding windows. The TimeSeriesDataset class handles this by using the sequence_length and stride parameters.

class TimeSeriesDataset(Dataset):
    def __init__(self, folder_path, sequence_length=5000, stride=500, normalize=True):
        self.train_data = np.load(f'{folder_path}/train.npy')
        #self.train_timestamp = np.load(f'{folder_path}/train_timestamp.npy')
        self.train_label = np.load(f'{folder_path}/train_label.npy')
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        
        if self.normalize:
            """self.mean = np.mean(self.train_data)
            self.std = np.std(self.train_data)
            self.train_data = (self.train_data - self.mean) / self.std"""
            self.min_val = self.train_data.min()
            if self.min_val < 0:
                self.train_data = self.train_data - self.min_val
        
    def __len__(self):
        return (len(self.train_data) - self.sequence_length) // self.stride + 1
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        sequence = self.train_data[start_idx:end_idx]
        sequence_labels = self.train_label[start_idx:end_idx]
        return torch.FloatTensor(sequence), sequence_labels #self.train_timestamp[start_idx], sequence_labels
    
    def denormalize(self, data):
        if self.normalize:
            return data * self.std + self.mean
        return data

class TimeSeriesTestDataset(Dataset):
    def __init__(self, folder_path, sequence_length=10000, stride=1000, normalize=True, mean=None, std=None):
        self.test_data = np.load(f'{folder_path}/test.npy')
        #self.test_timestamp = np.load(f'{folder_path}/test_timestamp.npy')
        self.test_label = np.load(f'{folder_path}/test_label.npy')
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        
        if self.normalize:
            # If mean and std are provided, use them; otherwise, calculate from test data
            """if mean is not None and std is not None:
                self.mean = mean
                self.std = std
            else:
                self.mean = np.mean(self.test_data)
                self.std = np.std(self.test_data)
            self.test_data = (self.test_data - self.mean) / self.std"""
            self.min_val = self.test_data.min()
            if self.min_val < 0:
                self.test_data = self.test_data - self.min_val
        
    def __len__(self):
        return (len(self.test_data) - self.sequence_length) // self.stride + 1
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        sequence = self.test_data[start_idx:end_idx]
        sequence_labels = self.test_label[start_idx:end_idx]
        return torch.FloatTensor(sequence),sequence_labels # self.test_timestamp[start_idx], sequence_labels
    
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
#================================================================================================
# Models 
#================================================================================================

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

"""
Analysis of your current MLP architecture:

Positional Embeddings: You're using positional embeddings for both input and time, which is good for capturing temporal information.
Multi-layer structure: The use of multiple hidden layers with GELU activations can help capture complex patterns.
Concatenation of embeddings: You're concatenating time and input embeddings, which allows the model to consider both simultaneously.
Dimensionality: The model maintains the input size throughout, which is appropriate for reconstruction tasks in diffusion models.

Feedback:

While this architecture can work, MLPs might struggle to capture long-range dependencies in time series data efficiently.
The model doesn't explicitly leverage the sequential nature of time series data.
For anomaly detection, it might be beneficial to have an architecture that can more easily capture temporal patterns and anomalies."""  

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, t):
        # Add channel dimension if it's not present
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Now x has shape (batch_size, sequence_length, 1)
        
        # Add time information
        t = t.unsqueeze(1).unsqueeze(2).expand(-1, x.size(1), -1)
        x = torch.cat([x, t], dim=-1)
        
        # Reshape for 1D convolution (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)
        
        y1 = self.tcn(x)
        return self.linear(y1.transpose(1, 2)).squeeze(-1)
    
# Example usage:
# model = TCN(input_size=2, output_size=1, num_channels=[64, 128, 256, 512], kernel_size=3, dropout=0.2)
# x = torch.randn(32, 100, 1)  # (batch_size, sequence_length, features)
# t = torch.randn(32)  # (batch_size,)
# output = model(x, t)

class ImprovedTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(ImprovedTCN, self).__init__()
        self.tcn_layers = nn.ModuleList([TemporalConvNet(input_size if i == 0 else num_channels[i-1], 
                                                         [num_channels[i]], 
                                                         kernel_size=kernel_size, 
                                                         dropout=dropout) 
                                         for i in range(len(num_channels))])
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, t):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        t = t.unsqueeze(1).unsqueeze(2).expand(-1, x.size(1), -1)
        x = torch.cat([x, t], dim=-1)
        x = x.permute(0, 2, 1)
        
        skip_connections = []
        for tcn in self.tcn_layers:
            x = tcn(x)
            skip_connections.append(x)
        
        x = torch.sum(torch.stack(skip_connections), dim=0)
        return self.linear(x.transpose(1, 2)).squeeze(-1)
"""
This TCN-based architecture offers several advantages for time series anomaly detection:

1. Temporal Modeling: TCNs are designed to capture temporal dependencies effectively, which is crucial for time series data.

2. Long-range Dependencies: The dilated convolutions allow the model to capture long-range dependencies without the need for very deep networks.

3. Parallelization: Unlike RNNs, TCNs can be parallelized, which can lead to faster training and inference.

4. Stable Gradients: TCNs don't suffer from vanishing gradients, which can be an issue with very long sequences in RNNs.

5. Time Information: The model incorporates time information by concatenating it with the input, allowing it to learn time-dependent patterns.

Key components:

- `TemporalBlock`: The basic building block of the TCN, consisting of dilated causal convolutions, residual connections, and normalization.
- `TemporalConvNet`: Stacks multiple `TemporalBlock`s with increasing dilation.
- `TCN`: The main model that combines the `TemporalConvNet` with a final linear layer.

To use this for anomaly detection:

1. Train the model to reconstruct normal time series data.
2. During inference, compare the model's reconstruction with the input data.
3. Large reconstruction errors may indicate anomalies.

You can adjust the `num_channels`, `kernel_size`, and `dropout` parameters to fine-tune the model for your specific dataset and task.

This TCN architecture should be more effective at capturing temporal patterns in your time series data compared to the MLP architecture, potentially leading to better anomaly detection performance. However, you may need to experiment with both architectures to determine which works best for your specific dataset and use case.
"""
    
#================================================================================================
# Noise Scheduler
#================================================================================================

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

def main(config):
    sequence_length = 10000
    input_size = 2  # Assuming 1D time series data
    
    # Set up dataset and dataloader
    dataset = TimeSeriesDataset(config.data_path,normalize=True)
    
    # Shuffling: The dataloader is currently not shuffling the data (shuffle=False). For training, it's usually better to shuffle to prevent the model from learning sequence order. I've changed this to shuffle=True in the artifact.
    # Dropping last batch: You might want to set drop_last=True in the DataLoader to ensure all batches are of the same size. This can be important for some operations.
    
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=False, drop_last=True)
    # Adjust model initialization================================================================================================
    
    """model = MLP(
        input_size=sequence_length,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding)"""
    
    # Initialize TCN model
    model = TCN(
        input_size=input_size,  # +1 for time information
        output_size=1,
        num_channels=[64, 128, 256, 512],
        kernel_size=3,
        dropout=0.2
    )
    #================================================================================================
    
    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule
    )

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
            #batch = batch.unsqueeze(-1)  # Add channel dimension: (batch_size, sequence_length, 1)
            
            noise = torch.randn_like(batch)
            
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],)
            ).long()

            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            
            print(f"Batch shape: {batch.shape}")
            print(f"Noise shape: {noise.shape}")
            print(f"Noisy shape: {noisy.shape}")

            # visualize the input and noisy data 
            fig, axs = plt.subplots(2, 2, figsize=(20, 10))
            for i in range(min(4, config.train_batch_size)):
                ax = axs[i // 2, i % 2]
                ax.plot(batch[i].cpu().numpy())
                ax.plot(noisy[i].cpu().numpy())
                ax.set_title(f"Input vs Noisy Sample {i+1}")
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Value")
                # color labels
                ax.legend(["Input", "Noisy"])
                
            plt.tight_layout()
            save_path = os.path.join(config.output_dir, f"noisy_sample_epoch_{epoch}.png")
            plt.savefig(save_path)
            plt.close()
            
            # pred noise from the model
            noise_pred = model(noisy, timesteps)
            print(f"Model output shape: {noise_pred.shape}")
            
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            
            """order proposed by claude --- to verify 
            loss = nn.MSELoss()(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step() 
            """
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
        if epoch % config.save_eval_step == 0 or epoch == config.num_epochs - 1:
            model.eval()
            eval_batch_size = config.eval_batch_size
            sample = torch.randn(eval_batch_size, sequence_length)
            timesteps = list(range(noise_scheduler.num_timesteps))[::-1]

            with torch.no_grad():
                for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
                    time_tensor = torch.full((eval_batch_size,), t, device=sample.device, dtype=torch.long)
                    residual = model(sample, time_tensor)
                    sample = noise_scheduler.step(residual, time_tensor[0], sample)
        
        # Visualize the generated samples
            fig, axs = plt.subplots(2, 2, figsize=(20, 10))
            for i in range(min(4, eval_batch_size)):
                ax = axs[i // 2, i % 2]
                ax.plot(sample[i].cpu().numpy())
                ax.set_title(f"Generated Sample {i+1}")
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Value")
            
            plt.tight_layout()
            save_path = os.path.join(config.output_dir, f"sample_epoch_{epoch}.png")
            plt.savefig(save_path)
            plt.close()

            print(f"Saved evaluation samples to {save_path}")
            
            
            
        # Evaluation and anomaly detection
        if epoch % config.save_eval_step == 0 or epoch == config.num_epochs - 1:
            model.eval()
            all_errors = []
            all_labels = []
            
            with torch.no_grad():
                for batch, timestamps, labels in tqdm(dataloader, desc="Evaluating"):
                    # Generate a sample starting from noise
                    sample = torch.randn_like(batch)
                    timesteps = list(range(noise_scheduler.num_timesteps))[::-1]
                    
                    for t in timesteps:
                        time_tensor = torch.full((batch.shape[0],), t, device=sample.device, dtype=torch.long)
                        residual = model(sample, time_tensor)
                        sample = noise_scheduler.step(residual, time_tensor[0], sample)
                    
                    # Calculate reconstruction error
                    error = F.mse_loss(sample, batch, reduction='none').mean(dim=1)
                    all_errors.extend(error.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Convert to numpy arrays for easier processing
            all_errors = np.array(all_errors)
            all_labels = np.array(all_labels)
            
            # Calculate threshold (e.g., 95th percentile of errors)
            threshold = np.percentile(all_errors, 95)
            
            # Identify anomalies
            predicted_anomalies = all_errors > threshold
            
            # Calculate metrics
            true_positives = np.sum((predicted_anomalies == 1) & (all_labels == 1))
            false_positives = np.sum((predicted_anomalies == 1) & (all_labels == 0))
            false_negatives = np.sum((predicted_anomalies == 0) & (all_labels == 1))
            
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            print(f"Epoch {epoch} - Anomaly Detection Results:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1_score:.4f}")
            
            # Visualize results
            plt.figure(figsize=(12, 6))
            plt.scatter(range(len(all_errors)), all_errors, c=all_labels, cmap='coolwarm')
            plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            plt.xlabel('Sample Index')
            plt.ylabel('Reconstruction Error')
            plt.title(f'Anomaly Detection - Epoch {epoch}')
            plt.legend()
            plt.colorbar(label='True Label (0: Normal, 1: Anomaly)')
            
            save_path = os.path.join(config.output_dir, f"anomaly_detection_epoch_{epoch}.png")
            plt.savefig(save_path)
            plt.close()
            
            print(f"Saved anomaly detection results to {save_path}")
            
        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    torch.save(model.state_dict(), f"{config.output_dir}/model_epoch_{epoch}.pth")

    print("Saving model...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")
    return model, losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the folder containing the dataset")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_eval_step", type=int, default=10)
    parser.add_argument("--experiment_name", type=str, default='NonName')
    
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)

    parser.add_argument("--save_model_epochs", type=int, default=2)

    config = parser.parse_args()
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    main(config)