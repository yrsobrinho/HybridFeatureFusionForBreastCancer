import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.dataloader import BreastMNISTLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )
        
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # input: (B, 64, 28, 28)
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64)
        )  # 28 -> 56
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32)
        )  # 56 -> 112
        self.up3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16)
        )  # 112 -> 224
        self.final_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.out_act = nn.Sigmoid()  # mantemos Sigmoid se as imagens estiverem em [0,1]

    def forward(self, x):
        # x shape (B, 64, 28, 28)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 56
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 112
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 224
        x = self.up3(x)
        x = self.final_conv(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    return device

def setup_directories():
    """Create and return directories for saving models and results."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    directories = {
        "project_dir": project_dir,
        "experiment_dir": os.path.join(project_dir, "experiments", "separate"),
        "features_dir": os.path.join(project_dir, "experiments", "separate", "features"),
        "models_dir": os.path.join(project_dir, "experiments", "separate", "models"),
        "results_dir": os.path.join(project_dir, "experiments", "separate", "results")
    }
    
    # Create autoencoder directory
    autoencoder_dir = os.path.join(directories["models_dir"], "autoencoder")
    os.makedirs(autoencoder_dir, exist_ok=True)
    directories["autoencoder_dir"] = autoencoder_dir
    
    # Create directory for saving visualizations
    visualizations_dir = os.path.join(directories["results_dir"], "autoencoder_visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    directories["visualizations_dir"] = visualizations_dir
    
    return directories

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_autoencoder(model, train_loader, optimizer, device, num_epochs=10, log_interval=50):

    model.train()
    history = {"train_loss": []}
    criterion = lambda recon, x: 0.5 * nn.MSELoss()(recon, x) + 0.5 * nn.L1Loss()(recon, x)
    
    for epoch in range(num_epochs):
        
        running_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i, (images, _) in enumerate(pbar):
                images = images.to(device)
                
                # Forward pass
                reconstructed, _ = model(images)
                loss = criterion(reconstructed, images)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                
                if (i + 1) % log_interval == 0:
                    pbar.set_postfix(loss=f"{running_loss / log_interval:.6f}")
                    running_loss = 0.0
        
        # Compute epoch loss
        epoch_loss = compute_epoch_loss(model, train_loader, criterion, device)
        history["train_loss"].append(epoch_loss)
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.6f}")
    
    return history

def compute_epoch_loss(model, dataloader, criterion, device):
    """
    Compute the average loss over the entire dataset.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            reconstructed, _ = model(images)
            loss = criterion(reconstructed, images)
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    
    model.train()
    return total_loss / total_samples

def visualize_reconstructions(model, test_loader, device, save_path):
    """
    Generate and save visualizations of original and reconstructed images.
    """
    model.eval()
    
    # Get a batch of test images
    images, _ = next(iter(test_loader))
    
    # Forward pass to get reconstructions
    with torch.no_grad():
        images = images.to(device)
        reconstructed, _ = model(images)
    
    # Convert to numpy arrays
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    # Plot a grid of original and reconstructed images
    n = min(8, images.shape[0])  # Display up to 8 examples
    plt.figure(figsize=(20, 4))
    
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title("Original")
        plt.axis("off")
        
        # Reconstructed
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(reconstructed[i][0], cmap='gray')
        plt.title("Reconstructed")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Reconstruction visualization saved to {save_path}")

def visualize_latent_space(model, test_loader, device, save_path):
    """
    Visualize the latent space using PCA or t-SNE.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    model.eval()
    all_latents = []
    all_labels = []
    
    # Extract latent representations
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            _, latent = model(images)
            # Convert 4D feature maps (B, C, H, W) to per-sample vectors (B, C)
            # using global average pooling to avoid PCA input dimension errors.
            pooled = F.adaptive_avg_pool2d(latent, (1, 1)).view(latent.size(0), -1)
            all_latents.append(pooled.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate all batches
    all_latents = np.concatenate(all_latents)
    all_labels = np.concatenate(all_labels)
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(all_latents)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=all_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title('PCA of Latent Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Latent space visualization saved to {save_path}")

def save_latent_maps(model, loader, device, save_path, n=8, avg_channels=True, save_npy=False):
    """
    Extract encoder latent maps (B, C, H, W) and save a grid of 28x28 maps.
    avg_channels: if True average across channels to get single-channel images.
    save_npy: if True save individual latent maps as .npy files in same directory.
    """
    model.eval()
    images, labels = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        latent = model.encode(images)  # expect (B, C, 28, 28)
    latent_cpu = latent.cpu().numpy()  # (B, C, H, W)
    B = latent_cpu.shape[0]
    m = min(n, B)
    if avg_channels:
        maps = latent_cpu.mean(axis=1)  # (B, H, W)
    else:
        maps = latent_cpu[:, 0, :, :]  # take first channel

    # Normalize each map to [0,1] for visualization
    maps_norm = []
    for i in range(m):
        arr = maps[i]
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        maps_norm.append(arr)
    maps_norm = np.stack(maps_norm)

    # Plot grid
    plt.figure(figsize=(10, 4))
    for i in range(m):
        ax = plt.subplot(1, m, i + 1)
        plt.imshow(maps_norm[i], cmap='gray')
        plt.title(f"Latent {i}")
        plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Latent 28x28 maps saved to {save_path}")

    if save_npy:
        out_dir = os.path.splitext(save_path)[0] + "_npy"
        os.makedirs(out_dir, exist_ok=True)
        for i in range(m):
            np.save(os.path.join(out_dir, f"latent_{i}.npy"), maps[i])
        print(f"Saved {m} latent .npy files to {out_dir}")

def main():
    print("=== Setting up training environment ===")
    device = setup_device()
    directories = setup_directories()
    
    print("\n=== Configuring hyperparameters ===")
    hyperparams = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 500
    }
    print(f"Hyperparameters: {hyperparams}")
    
    print("\n=== Loading datasets ===")
    loader = BreastMNISTLoader(batch_size=hyperparams["batch_size"], image_size=224)
    train_loader, test_loader, val_loader = loader.get_dataloaders()
    print(f"Dataset loaded with {len(train_loader)} training batches, {len(val_loader)} validation batches, {len(test_loader)} test batches")
    
    print("\n=== Initializing model ===")
    autoencoder = Autoencoder().to(device)
    print(f"Total trainable parameters: {count_parameters(autoencoder)}")
    
    print("\n=== Configuring optimizer ===")
    optimizer = optim.Adam(autoencoder.parameters(), lr=hyperparams["learning_rate"])
    print(f"Using Adam optimizer with learning rate: {hyperparams['learning_rate']}")
    
    print("\n=== Starting training process ===")
    
    # Debug tensor shapes before training
    sample_batch, _ = next(iter(train_loader))
    print(f"Input image shape: {sample_batch.shape}")
    
    # Test model with a single batch to verify tensor shapes
    with torch.no_grad():
        sample_batch = sample_batch.to(device)
        reconstructed, _ = autoencoder(sample_batch)
        print(f"Reconstructed image shape: {reconstructed.shape}")
        print(f"Expected image shape: {sample_batch.shape}")
        if reconstructed.shape != sample_batch.shape:
            print("WARNING: Shape mismatch - will be using interpolation to fix")
    
    # Continue with normal training
    history = train_autoencoder(
        model=autoencoder,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=hyperparams["epochs"]
    )
    
    print("\n=== Saving model ===")
    model_path = os.path.join(directories["autoencoder_dir"], "autoencoder.pth")
    torch.save(autoencoder.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    
    print("\n=== Generating visualizations ===")
    recon_path = os.path.join(directories["visualizations_dir"], "reconstructions.png")
    visualize_reconstructions(autoencoder, test_loader, device, recon_path)
    
    # Save 28x28 latent feature maps (averaged across channels) as an image grid
    latent_feature_map_path = os.path.join(directories["visualizations_dir"], "latent_feature_maps_28x28.png")
    save_latent_maps(autoencoder, test_loader, device, latent_feature_map_path, n=8, avg_channels=True, save_npy=False)
    
    latent_path = os.path.join(directories["visualizations_dir"], "latent_space.png")
    visualize_latent_space(autoencoder, test_loader, device, latent_path)
    
    print("\n=== Training complete ===")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    loss_path = os.path.join(directories["visualizations_dir"], "loss_curve.png")
    plt.savefig(loss_path)
    print(f"Loss curve saved to {loss_path}")

if __name__ == "__main__":
    main()