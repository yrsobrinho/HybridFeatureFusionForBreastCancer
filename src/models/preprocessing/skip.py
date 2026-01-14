import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.dataloader import BreastMNISTLoader, BCDRLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pytorch_msssim import ssim

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """Residual block para melhor fluxo de gradiente e extração de features."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChannelAttention(nn.Module):
    """Channel attention para focar nos canais mais importantes da latente."""
    def __init__(self, channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class Encoder(nn.Module):
    def __init__(self, input_channels=1, latent_channels=64):
        super().__init__()

        self.conv1 = DoubleConv(input_channels, 32)   # 224
        self.pool1 = nn.MaxPool2d(2)                  # 112

        self.conv2 = DoubleConv(32, 64)               # 112
        self.pool2 = nn.MaxPool2d(2)                  # 56

        self.conv3 = DoubleConv(64, 128)              # 56
        self.pool3 = nn.MaxPool2d(2)                  # 28

        self.bottleneck = DoubleConv(128, 256)        # 28

        # gargalo informacional (NÃO normalizado, NÃO ativado)
        self.to_latent = nn.Conv2d(256, latent_channels, 1)

        # projeção dos skips (enfraquece atalho)
        self.skip3_proj = nn.Conv2d(128, 64, 1)
        self.skip2_proj = nn.Conv2d(64, 32, 1)
        self.skip1_proj = nn.Conv2d(32, 16, 1)

    def forward(self, x):
        s1 = self.conv1(x)
        x = self.pool1(s1)

        s2 = self.conv2(x)
        x = self.pool2(s2)

        s3 = self.conv3(x)
        x = self.pool3(s3)

        x = self.bottleneck(x)
        z = self.to_latent(x)

        return z, self.skip1_proj(s1), self.skip2_proj(s2), self.skip3_proj(s3)
    
class Decoder(nn.Module):
    def __init__(self, latent_channels=64, output_channels=1):
        super().__init__()

        self.expand = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = DoubleConv(256 + 64, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv2 = DoubleConv(128 + 32, 64)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv3 = DoubleConv(64 + 16, 32)

        self.final = nn.Conv2d(32, output_channels, 1)

    def forward(self, z, s1, s2, s3):
        x = self.expand(z)

        x = self.up1(x)
        x = self.conv1(torch.cat([x, s3], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([x, s2], dim=1))

        x = self.up3(x)
        x = self.conv3(torch.cat([x, s1], dim=1))

        return self.final(x)


class Autoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_channels=64):
        super().__init__()
        self.encoder = Encoder(input_channels, latent_channels)
        self.decoder = Decoder(latent_channels, input_channels)

    def forward(self, x):
        z, s1, s2, s3 = self.encoder(x)
        x_hat = self.decoder(z, s1, s2, s3)
        return x_hat, z

    def encode(self, x):
        z, _, _, _ = self.encoder(x)
        return z

    def decode(self, z):
        return self.decoder(z, None, None, None)

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
        "experiment_dir": os.path.join(project_dir, "experiments"),
        "features_dir": os.path.join(project_dir, "experiments", "features"),
        "models_dir": os.path.join(project_dir, "experiments", "models"),
        "results_dir": os.path.join(project_dir, "experiments", "results")
    }
    
    autoencoder_dir = os.path.join(directories["models_dir"], "unet_autoencoder")
    os.makedirs(autoencoder_dir, exist_ok=True)
    directories["autoencoder_dir"] = autoencoder_dir
    
    visualizations_dir = os.path.join(directories["results_dir"], "unet_autoencoder_visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    directories["visualizations_dir"] = visualizations_dir
    
    return directories

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_autoencoder(model, train_loader, optimizer, device, num_epochs=10, log_interval=50):

    model.train()
    history = {"train_loss": []}
    l1 = nn.L1Loss()
    
    def criterion(recon, x):
        # MSE para fidelidade pixel-wise
        l1_loss = l1(recon, x)
        # SSIM para preservar estrutura e textura
        ssim_val = ssim(recon, x, data_range=1.0, size_average=True)
        # Combinar: 40% MSE + 60% SSIM (dar mais peso à estrutura)
        return l1_loss + 0.5 * (1 - ssim_val)
    
    # Scheduler mais suave com ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i, (images, _) in enumerate(pbar):
                images = images.to(device)

                reconstructed, latent = model(images)
                loss = criterion(reconstructed, images)

                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping para estabilidade
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

                running_loss += loss.item()
                
                if (i + 1) % log_interval == 0:
                    pbar.set_postfix(loss=f"{running_loss / log_interval:.6f}")
                    running_loss = 0.0
        
        epoch_loss = compute_epoch_loss(model, train_loader, criterion, device, epoch)
        history["train_loss"].append(epoch_loss)
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.6f}")

        scheduler.step(epoch_loss)
    
    return history

def compute_epoch_loss(model, dataloader, criterion, device, epoch=0):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            reconstructed, latent = model(images)
            loss = criterion(reconstructed, images)
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    
    model.train()
    return total_loss / total_samples

def visualize_reconstructions(model, test_loader, device, save_path):
    model.eval()
    images, _ = next(iter(test_loader))
    with torch.no_grad():
        images = images.to(device)
        reconstructed, latent = model(images)
  
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    latent = latent.cpu().numpy()
    
    n = min(8, images.shape[0])
    num_latent_channels = latent.shape[1]
    
    plt.figure(figsize=(20, 8))
    
    for i in range(n):
        ax = plt.subplot(3, n, i+1)
        plt.imshow(images[i][0], cmap='gray')
        if i == 0:
            plt.ylabel("Original", fontsize=12)
        plt.title(f"224x224")
        plt.axis("off")

        ax = plt.subplot(3, n, i+1+n)
        # Calcular a média de todos os canais latentes
        latent_mean = latent[i].mean(axis=0)
        plt.imshow(latent_mean, cmap='gray')
        if i == 0:
            plt.ylabel("Latent", fontsize=12)
        plt.title(f"{latent.shape[2]}x{latent.shape[3]}")
        plt.axis("off")
        
        # Reconstructed
        ax = plt.subplot(3, n, i+1+2*n)
        plt.imshow(reconstructed[i][0], cmap='gray')
        if i == 0:
            plt.ylabel("Reconstructed", fontsize=12)
        plt.title(f"224x224")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Reconstruction visualization saved to {save_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Autoencoder')
    parser.add_argument('--dataset', type=str, default='breastmnist', choices=['breastmnist', 'bcdr'],
                        help='Dataset to use: breastmnist (default) or bcdr')
    parser.add_argument('--bcdr_root', type=str, default='src/data',
                        help='Root directory for BCDR dataset (used with --dataset bcdr)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--latent_channels', type=int, default=64, help='Number of latent channels (1, 3, 64, etc.)')
    args = parser.parse_args()
    
    print("=== Setting up training environment ===")
    device = setup_device()
    directories = setup_directories()
    
    print("\n=== Configuring hyperparameters ===")
    hyperparams = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "latent_channels": args.latent_channels,
        "dataset": args.dataset
    }
    print(f"Hyperparameters: {hyperparams}")
    
    print(f"\n=== Loading {args.dataset.upper()} dataset ===")
    if args.dataset == 'bcdr':
        loader = BCDRLoader(
            batch_size=hyperparams["batch_size"], 
            image_size=224,
            root_dir=args.bcdr_root
        )
        dataset_name = 'bcdr'
        print(f"BCDR dataset from: {args.bcdr_root}")
    else:
        loader = BreastMNISTLoader(
            batch_size=hyperparams["batch_size"], 
            image_size=224
        )
        dataset_name = 'breastmnist'
        print("BreastMNIST dataset")
    
    train_loader, test_loader, val_loader = loader.get_dataloaders()
    print(f"Dataset loaded with {len(train_loader)} training batches, {len(val_loader)} validation batches, {len(test_loader)} test batches")
    
    print("\n=== Initializing model ===")
    autoencoder = Autoencoder(
        input_channels=1,
        latent_channels=args.latent_channels
    ).to(device)
    print(f"Autoencoder architecture: input_channels=1, latent_channels={args.latent_channels}")
    print(f"Total trainable parameters: {count_parameters(autoencoder)}")
    
    print("\n=== Configuring optimizer ===")
    optimizer = optim.Adam(autoencoder.parameters(), lr=hyperparams["learning_rate"])
    print(f"Using Adam optimizer with learning rate: {hyperparams['learning_rate']}")
    
    print("\n=== Starting training process ===")
    
    sample_batch, _ = next(iter(train_loader))
    print(f"Input image shape: {sample_batch.shape}")
    
    with torch.no_grad():
        sample_batch = sample_batch.to(device)
        reconstructed, _ = autoencoder(sample_batch)
        print(f"Reconstructed image shape: {reconstructed.shape}")
        print(f"Expected image shape: {sample_batch.shape}")
        if reconstructed.shape != sample_batch.shape:
            print("WARNING: Shape mismatch - will be using interpolation to fix")
    
    history = train_autoencoder(
        model=autoencoder,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=hyperparams["epochs"]
    )
    
    print("\n=== Saving model ===")
    model_path = os.path.join(directories["autoencoder_dir"], f"autoencoder_dif_{dataset_name}_latent{args.latent_channels}.pth")
    torch.save(autoencoder.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    
    print("\n=== Generating visualizations ===")
    recon_path = os.path.join(directories["visualizations_dir"], "reconstructions.png")
    visualize_reconstructions(autoencoder, test_loader, device, recon_path)
    
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