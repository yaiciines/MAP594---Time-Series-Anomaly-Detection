import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

import ddpm_Time_Series_copy
import numpy as np
import pandas as pd
import torch

from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset


import ddpm_Time_Series_copy
from ddpm_Time_Series_copy import TimeSeriesDataset , TimeSeriesTestDataset
from noise_scheduler import NoiseScheduler


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


from pytorch_tcn import TCN  # Import the TCN class from pytorch-tcn

import json

from sklearn.metrics import classification_report


#===================================================================================================================================================

# f1 score and auprc score for 0/1 classification

from sklearn.metrics import f1_score, average_precision_score

# define the function to calculate the f1 score and auprc score
def calculate_f1_auprc_score(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0 ) #, labels=np.unique(y_pred) zero_division=np.nan
    auprc = average_precision_score(y_true, y_pred, average='weighted')
    return f1, auprc

# plot functions : 
def plot_samples(sample, noisy_sample, denoised_sample):  
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    ax = axs[0]
    ax.plot(sample.cpu().numpy())
    ax.set_title("Input Sample")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend(["Input"])
    
    ax = axs[1]
    ax.plot(noisy_sample.cpu().numpy())
    ax.set_title("Noisy Sample")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend(["Noisy"])
    
    ax = axs[2]
    ax.plot(denoised_sample.detach().numpy())
    ax.set_title("Denoised Sample")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend(["Denoised"])
    
    plt.tight_layout()
    plt.show()

def plot_predictions(sample, denoised_sample, outliers, labels):
    
    # in the same graph visualize the input sample and the predicted outliers and the labels
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(sample[0].cpu().numpy(), label="Input")

    outlier_indices = np.where(outliers[0] == 1)[0]
    ax.plot(outlier_indices, sample[0][outlier_indices].cpu().numpy(), 'ro', label="Outliers")

    ax.set_title("Input Sample with Outliers")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # in the same graph visualize the input sample and the predicted outliers and the labels
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(sample[0].cpu().numpy(), label="Input")

    label_indices = np.where(labels[0] == True)[0]
    ax.plot(label_indices, sample[0][label_indices].cpu().numpy(), 'ro', label="Labels")
    
    ax.set_title("Input Sample with labels")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    
def trainer(config, model, noise_scheduler, dataset):
    
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=False , drop_last=True)

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
        
    for epoch in tqdm(range(config.num_epochs)):
        model.train()
        #progress_bar = tqdm(total=len(dataloader))
        #progress_bar.set_description(f"Epoch {epoch}")
        
        for step, (batch, labels) in enumerate(dataloader):
            std_data = torch.std(batch)
            
            noise = torch.randn_like(batch)
            
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],)
            ).long()
            
            #print("timesteps", timesteps)
            
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            
            noisy = noisy.unsqueeze(-1) 
            # pred noise from the model
            noise_pred = model(noisy)
            
            #print("noise_pred", noise_pred.shape)
            batch = batch.unsqueeze(-1)
            
            loss = F.mse_loss(noise_pred, batch) # je vais predire la distribution de base 
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        #progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "step": global_step}
        losses.append(loss.detach().item())
        #progress_bar.set_postfix(**logs)
        global_step += 1
    
    #progress_bar.close()              
    return model, losses


def train_tracker(config, model, dataset):
    # save the model, losses, and all the config parameters in a directory 
    import os
    import json
    import matplotlib.pyplot as plt

    # output dir contains the num epochs 
    output_dir = os.path.join(config.output_dir, str(config.num_epochs))
    # create the directory if it does not exist
    os.makedirs(output_dir, exist_ok=True) 

    # Save the model
    model_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    # Save the losses and other parameters in a json file 
    losses_path = os.path.join(output_dir, "losses.json")
    with open(losses_path, "w") as f:
        json.dump(losses, f)
        
    # Save the config 
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f)

    # Save the losses plot
    # Plot the loss
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    
    # Save the plot before showing it
    losses_plot_path = os.path.join(output_dir, "losses_plot.png")
    plt.savefig(losses_plot_path)
    
    # Display the plot
    plt.show()
        
    """print("Model saved at", model_path)
    print("Losses saved at", losses_path)
    print("Config saved at", config_path)
    print("Loss plot saved at", losses_plot_path)"""

    return model, losses

    
def tester(config, model, noise_scheduler, test_dataset): 
    output_dir = os.path.join(config.output_dir, str(config.num_epochs))
    
    # Create dataloader for the test dataset
    test_dataloader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False, drop_last=True)
    
    model.eval()
    eval_batch_size = config.eval_batch_size
    
    average_f1 = 0
    average_auprc = 0
    
    results_df = pd.DataFrame(columns=["std_data", "f1", "auprc", "threshold"])

    for step, (sample, labels) in enumerate(test_dataloader):
        
        std_data = torch.std(sample)
        
        # after sampling random noise, predict using the model to denoise the sample
        with torch.no_grad():
            timesteps = torch.randint(0, 50 , (eval_batch_size,)).long()
                
            #noise_level = 0.5
            #noise = torch.randn_like(sample) * std_data * noise_level
            noise = torch.randn_like(sample) 
            noisy_sample = noise_scheduler.add_noise(sample, noise, timesteps)
            
            noisy_sample = noisy_sample.unsqueeze(-1)
            sample = sample.unsqueeze(-1)
            denoised_sample = model(noisy_sample)

            labels = labels.cpu().numpy()
                
            best_f1 = 0
            best_threshold = 0
            best_auprc = 0
            # computing the best threshold for the outliers
            for threshold in torch.arange(std_data * 0.5, std_data * 2.5, std_data * 0.2):
                outliers = (torch.abs(sample - denoised_sample) > threshold).cpu().numpy()
                
                f1 = 0
                auprc = 0
                
                # calculate the f1 score and auprc score
                for i in range(len(outliers)):
                    f1_temp, auprc_temp = calculate_f1_auprc_score(labels[i], outliers[i])
                    f1 += f1_temp
                    auprc += auprc_temp
                
                f1 = f1 / len(outliers)
                auprc = auprc / len(outliers)
         
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_auprc = auprc
                
            print("F1 score:", best_f1)
            print("AUPRC score:", best_auprc)
            
            # save results in output directory
            results = {
                "std_data": std_data.item(),  # Convert tensor to a scalar
                "f1": best_f1,
                "auprc": best_auprc,
                "threshold": best_threshold  # Convert tensor to a scalar
            }
            
            # Convert the results dictionary to a DataFrame and concatenate with the existing results_df
            results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
            
            # plot one random sample 
            if step == 10:
                plot_samples(sample[10], noisy_sample[10], denoised_sample[10])
                plot_predictions(sample, denoised_sample, outliers, labels)
                
                
    # save as a csvfile 
    results_df.to_csv(os.path.join(output_dir, "results.csv"))
    
    average_f1 += best_f1
    average_auprc += best_auprc
            
    average_f1 = average_f1 / len(test_dataloader)
    average_auprc = average_auprc / len(test_dataloader)
    
    print("Average F1 score:", average_f1)
    print("Average AUPRC score:", average_auprc)
    
    return average_f1, average_auprc


# metrics 
import torch
import numpy as np

def z_score_outliers(sample, denoised_sample, threshold=3.0):
    diff = sample - denoised_sample
    mean = torch.mean(diff)
    std = torch.std(diff)
    z_scores = torch.abs((diff - mean) / std)
    outliers = (z_scores > threshold).cpu().numpy()
    return outliers

def modified_z_score_outliers(sample, denoised_sample, threshold=3.5):
    diff = sample - denoised_sample
    median = torch.median(diff)
    mad = torch.median(torch.abs(diff - median))
    modified_z_scores = 0.6745 * torch.abs(diff - median) / mad
    outliers = (modified_z_scores > threshold).cpu().numpy()
    return outliers

def iqr_outliers(sample, denoised_sample, threshold=1.5):
    diff = sample - denoised_sample
    q1 = torch.quantile(diff, 0.25)
    q3 = torch.quantile(diff, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = ((diff < lower_bound) | (diff > upper_bound)).cpu().numpy()
    
    # Calculate anomaly scores
    anomaly_scores = torch.abs(diff) / iqr
    anomaly_scores = anomaly_scores.cpu().numpy()
    
    return outliers, anomaly_scores



"""# Example usage:
sample = torch.randn(1000)
denoised_sample = torch.randn(1000)

z_score_out = z_score_outliers(sample, denoised_sample)
mod_z_score_out = modified_z_score_outliers(sample, denoised_sample)
iqr_out,_ = iqr_outliers(sample, denoised_sample)

print(f"Z-score outliers: {z_score_out.sum()}")
print(f"Modified Z-score outliers: {mod_z_score_out.sum()}")
print(f"IQR outliers: {iqr_out.sum()}")"""


import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d

def iqr_collective_outliers(sample, denoised_sample, threshold=1.5, window_size=10, density_threshold=0.5, smoothing_sigma=None):
    # Step 1: Calculate point-wise differences
    diff = sample - denoised_sample
    
    # Step 2: Identify initial outliers using IQR method
    q1 = torch.quantile(diff, 0.25)
    q3 = torch.quantile(diff, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    initial_outliers = ((diff < lower_bound) | (diff > upper_bound))
    
    # Step 3: Calculate outlier density using a sliding window
    outlier_density = torch.zeros_like(diff, dtype=torch.float32)
    for i in range(len(diff)):
        start = max(0, i - window_size // 2)
        end = min(len(diff), i + window_size // 2)
        outlier_density[i] = torch.mean(initial_outliers[start:end].float())
    
    # Step 4: Apply Gaussian smoothing to the density if smoothing_sigma is provided
    if smoothing_sigma is not None:
        smoothed_density = torch.from_numpy(
            gaussian_filter1d(outlier_density.cpu().numpy(), sigma=smoothing_sigma)
        ).to(diff.device)
    else:
        smoothed_density = outlier_density
    
    # Step 5: Identify collective outliers based on density
    collective_outliers = smoothed_density > density_threshold
    
    return collective_outliers.cpu().numpy()

"""# Example usage:
sample = torch.randn(1000)
denoised_sample = torch.randn(1000)

# Add some collective anomalies
sample[300:350] += 5
sample[600:630] += 4

# Detect collective outliers with and without smoothing
collective_outliers_smooth = iqr_collective_outliers(sample, denoised_sample, smoothing_sigma=2.0)
collective_outliers_no_smooth = iqr_collective_outliers(sample, denoised_sample, smoothing_sigma=None)

print(f"Collective outliers detected (with smoothing): {collective_outliers_smooth.sum()}")
print(f"Collective outliers detected (without smoothing): {collective_outliers_no_smooth.sum()}")"""
