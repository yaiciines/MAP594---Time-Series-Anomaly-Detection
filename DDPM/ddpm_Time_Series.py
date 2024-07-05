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



#==============================================================================================================================================================
# Usage od the code
#==============================================================================================================================================================

"""usage: ddpm.py [-h] [--experiment_name EXPERIMENT_NAME] [--dataset {circle,dino,line,moons}]
               [--train_batch_size TRAIN_BATCH_SIZE] [--eval_batch_size EVAL_BATCH_SIZE] [--num_epochs NUM_EPOCHS]
               [--learning_rate LEARNING_RATE] [--num_timesteps NUM_TIMESTEPS] [--beta_schedule {linear,quadratic}]
               [--embedding_size EMBEDDING_SIZE] [--hidden_size HIDDEN_SIZE] [--hidden_layers HIDDEN_LAYERS]
               [--time_embedding {sinusoidal,learnable,linear,zero}] [--input_embedding {sinusoidal,learnable,linear,identity}]
               [--save_images_step SAVE_IMAGES_STEP]

optional arguments:
  -h, --help            show this help message and exit
  --experiment_name EXPERIMENT_NAME
  --dataset {circle,dino,line,moons}
  --train_batch_size TRAIN_BATCH_SIZE
  --eval_batch_size EVAL_BATCH_SIZE
  --num_epochs NUM_EPOCHS
  --learning_rate LEARNING_RATE
  --num_timesteps NUM_TIMESTEPS
  --beta_schedule {linear,quadratic}
  --embedding_size EMBEDDING_SIZE
  --hidden_size HIDDEN_SIZE
  --hidden_layers HIDDEN_LAYERS
  --time_embedding {sinusoidal,learnable,linear,zero}
  --input_embedding {sinusoidal,learnable,linear,identity}
  --save_images_step SAVE_IMAGES_STEP
[porsche Project-tiny-diffusion]$ """

#==============================================================================================================================================================
# Explication of the code
#==============================================================================================================================================================
"""The provided code integrates various components to form a complete model training setup, specifically targeting generative modeling with a focus on learning 
representations of data through the addition of noise and subsequent reconstruction. Below, the key components and their functionalities are explained:

### Block and MLP Classes
- **`Block`**: A simple neural network block that performs a linear transformation followed by a GELU activation function. It is used as a building block for 
more complex models.
- **`MLP` (Multi-Layer Perceptron)**: A neural network designed to work with two-dimensional inputs and temporal information. It employs positional embeddings 
for both input features and time, concatenates these embeddings, and processes the result through a series of `Block` instances and linear layers to predict a 
two-dimensional output.

### NoiseScheduler Class
- **`NoiseScheduler`**: Manages the addition of noise to the input data across different timesteps during training, based on a predefined schedule. It supports 
linear and quadratic scheduling of the noise levels (`beta`) and calculates several important variables for the diffusion process, such as `alphas`, `alphas_cumprod`
, and their square roots or inverses. This class also provides methods for reconstructing the original data from the noisy input and for adding noise to data at specific timesteps.

### Training Logic
- The main training loop involves iterating over a dataset, adding noise to the data, and training the `MLP` model to predict the added noise. This process 
simulates a diffusion model, where the model learns to denoise data, effectively learning the data distribution.
- **Noise Addition**: Noise is added to the data based on the timestep and the schedule managed by `NoiseScheduler`. The model is trained to predict this noise.
- **Reconstruction and Prediction**: The `NoiseScheduler` also supports reconstructing the original data from the noisy input and predicting the data of previous 
timesteps, facilitating the generative aspect of the model.

### Additional Components
- **Positional Embedding**: The `PositionalEmbedding` class (presumably defined elsewhere) is used to encode temporal information and input features into embeddings 
that are then processed by the MLP. The choice of embedding (sinusoidal, learnable, etc.) can be specified in the configuration.
- **Command-line Interface**: The script uses `argparse` to allow configuration of various parameters such as dataset, model architecture, and training hyperparameters 
through command-line arguments.

### Execution Flow
- Initialization: The script initializes the dataset, model, noise scheduler, and optimizer based on the provided command-line arguments.
- Training Loop: For each epoch, it iterates over the dataset, adds noise to the data, computes the loss by comparing the model's noise prediction to the actual noise, 
and updates the model's weights.
- Visualization and Saving: Optionally, at specified intervals, the script generates samples from the model to visualize the learning process and saves the model's parameters.

This setup is particularly suited for tasks that involve learning complex data distributions through generative modeling, leveraging the principles of diffusion models."""


#==============================================================================================================================================================
class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons","horizontal_lines","vertical_lines","concentric_circles","s_curve","blobs","inclined_lines","horizontal_lines_poisson","vertical_lines_poisson","inclined_lines_poisson","mnist","random"])
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)
    config = parser.parse_args()

    dataset = datasets.get_dataset(config.dataset)
    dataloader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    model = MLP(
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

    global_step = 0
    frames = []
    losses = []
    print("Training model...")
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            batch = batch[0]
            noise = torch.randn(batch.shape)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],)
            ).long()

            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            # generate data with the model to later visualize the learning process
            model.eval()
            sample = torch.randn(config.eval_batch_size, 2)
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
                with torch.no_grad():
                    residual = model(sample, t)
                sample = noise_scheduler.step(residual, t[0], sample)
            frames.append(sample.numpy())

    print("Saving model...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")

    print("Saving images...")
    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)
    frames = np.stack(frames)
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
    for i, frame in enumerate(frames):
        plt.figure(figsize=(10, 10))
        plt.scatter(frame[:, 0], frame[:, 1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f"{imgdir}/{i:04}.png")
        plt.close()

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    print("Saving frames...")
    np.save(f"{outdir}/frames.npy", frames)

# ==============================================================================================================================================================
# new way : 

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, root_dir, num_folders=5):
        self.root_dir = root_dir
        self.folders = random.sample(os.listdir(root_dir), num_folders)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        train_data = np.load(f'{self.root_dir}/{folder}/train.npy')
        train_timestamp = np.load(f'{self.root_dir}/{folder}/train_timestamp.npy')
        train_label = np.load(f'{self.root_dir}/{folder}/train_label.npy')
        return train_data, train_timestamp, train_label

root_dir = 'datasets/UTS/WSD'
time_series_dataset = TimeSeriesDataset(root_dir)
time_series_dataloader = DataLoader(time_series_dataset, batch_size=1, shuffle=True)


class TimeSeriesMLP(nn.Module):
    def __init__(self, hidden_size=128, hidden_layers=3, emb_size=128,
                 time_emb="sinusoidal", input_emb="sinusoidal"):
        super().__init__()
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + len(self.input_mlp.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 1))  # Change to 1 for time series output
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x_emb = self.input_mlp(x)
        t_emb = self.time_mlp(t)
        x = torch.cat((x_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)
    config = parser.parse_args()

    model = TimeSeriesMLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding
    )

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    global_step = 0
    losses = []
    print("Training model...")

    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(time_series_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, (train_data, train_timestamp, train_label) in enumerate(time_series_dataloader):
            train_data = torch.tensor(train_data, dtype=torch.float32)
            train_timestamp = torch.tensor(train_timestamp, dtype=torch.float32)
            train_label = torch.tensor(train_label, dtype=torch.float32)

            noise = torch.randn(train_data.shape)
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (train_data.shape[0],)).long()

            noisy = noise_scheduler.add_noise(train_data, noise, timesteps)
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

    print("Saving model...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

