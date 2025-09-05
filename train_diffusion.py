import numpy as np
import torch
import torch.nn.functional as F
import os
import gc

from monai.utils import set_determinism
from monai.engines import SupervisedTrainer
from monai.handlers import MeanSquaredError, from_engine

from generative.networks.schedulers import DDPMScheduler
from generative.networks.nets import DiffusionModelUNet

from ignite.contrib.handlers import ProgressBar

from simple_utils import get_wavelet_data, prepare_data
import random
from tqdm import tqdm
import argparse

# Set seeds for reproducibility
torch.manual_seed(8)
set_determinism(8)
random.seed(8)
np.random.seed(8)
torch.set_float32_matmul_precision('high')

def main(args):
    # Load and prepare data
    print("Loading wavelet data...")
    data = get_wavelet_data(args.data_dir)
    if not data:
        raise ValueError("No data found in the specified directory")
    
    print(f"Found {len(data)} samples")
    tensors = prepare_data(data)
    
    # Split data into train/val
    n_val = max(1, len(tensors) // 5)
    train_data = tensors[:-n_val]
    val_data = tensors[-n_val:]
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size_train,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size_train,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Set up model and training
    device = torch.device(args.device)
    
    # Initialize model
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=[args.channels] * 4,
        attention_levels=[False, False, True, True],
        num_res_blocks=2,
    ).to(device)
    
    # Initialize scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        schedule="linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    
    # Training loop
    for epoch in range(args.train_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.train_epochs}"):
            batch = batch.to(device)
            
            # Sample noise and timesteps
            noise = torch.randn_like(batch)
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (batch.shape[0],), device=device
            ).long()
            
            # Get noisy image
            noisy_image = scheduler.add_noise(original_samples=batch, noise=noise, timesteps=timesteps)
            
            # Predict noise
            noise_pred = model(noisy_image, timesteps)
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if args.save_path:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, args.save_path)
    
    print("Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/wavelet/fomo_60k/sub_1",
                      help="Directory containing wavelet data")
    parser.add_argument("--device", type=str, default="cpu",
                      help="Device to use for training (cpu or cuda)")
    parser.add_argument("--batch_size_train", type=int, default=1,
                      help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=1,
                      help="Number of data loading workers")
    parser.add_argument("--train_epochs", type=int, default=5,
                      help="Number of training epochs")
    parser.add_argument("--num_train_timesteps", type=int, default=100,
                      help="Number of timesteps in diffusion process")
    parser.add_argument("--channels", type=int, default=64,
                      help="Number of channels in UNet")
    parser.add_argument("--lr", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--save_path", type=str, default="models/diffusion_model.pth",
                      help="Path to save model checkpoints")
    
    args = parser.parse_args()
    main(args)
