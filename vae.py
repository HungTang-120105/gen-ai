import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        # Conv layers: 1 -> 32 -> 64 -> 128
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # 28 -> 14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 14 -> 7
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 7 -> 3 (actually 4 due to stride)

        # Flatten: 128 * 4 * 4 = 2048
        self.fc_hidden = nn.Linear(128 * 4 * 4, 256)

        # Output layers for μ and log(σ²)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))  # (batch, 32, 14, 14)
        x = F.relu(self.conv2(x))  # (batch, 64, 7, 7)
        x = F.relu(self.conv3(x))  # (batch, 128, 4, 4)

        x = x.view(x.size(0), -1)  # Flatten: (batch, 2048)
        x = F.relu(self.fc_hidden(x))  # (batch, 256)

        mu = self.fc_mu(x)  # (batch, latent_dim)
        logvar = self.fc_logvar(x)  # (batch, latent_dim)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        # Fully connected layer to expand latent vector
        self.fc = nn.Linear(latent_dim, 256)
        self.fc_reshape = nn.Linear(256, 128 * 4 * 4)

        # Deconv layers: 128 -> 64 -> 32 -> 1
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 4 -> 8
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 8 -> 16
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # 16 -> 32 (but we crop to 28)

    def forward(self, z):
        # z: (batch, latent_dim)
        x = F.relu(self.fc(z))  # (batch, 256)
        x = F.relu(self.fc_reshape(x))  # (batch, 2048)
        x = x.view(x.size(0), 128, 4, 4)  # (batch, 128, 4, 4)

        x = F.relu(self.deconv1(x))  # (batch, 64, 8, 8)
        x = F.relu(self.deconv2(x))  # (batch, 32, 16, 16)
        x = torch.sigmoid(self.deconv3(x))  # (batch, 1, 32, 32)

        # Crop to 28x28 to match MNIST size
        x = x[:, :, :28, :28]

        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=20, beta=1.0):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
        self.beta = beta

    def reparameterize(self, mu, logvar):
        # Reparameterization trick: z = μ + ε·σ
        # where ε ~ N(0, 1)
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²))
        eps = torch.randn_like(std)  # Sample ε
        z = mu + eps * std  # z = μ + ε·σ
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss: Binary Cross Entropy
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # KL Divergence loss: β·KL where KL = -0.5·Σ(1 + log(σ²) - μ² - σ²)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        loss = BCE + self.beta * KLD

        return loss, BCE, KLD


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_bce = 0
    total_kld = 0

    for x, _ in tqdm(train_loader, desc="Training"):
        x = x.to(device)

        # Forward pass
        recon_x, mu, logvar, z = model(x)

        # Compute loss
        loss, bce, kld = model.loss_function(recon_x, x, mu, logvar)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bce += bce.item()
        total_kld += kld.item()

    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_bce = total_bce / num_batches
    avg_kld = total_kld / num_batches

    return avg_loss, avg_bce, avg_kld


def test(model, test_loader, device):
    model.eval()
    total_loss = 0
    total_bce = 0
    total_kld = 0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)

            recon_x, mu, logvar, z = model(x)
            loss, bce, kld = model.loss_function(recon_x, x, mu, logvar)

            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

    num_batches = len(test_loader)
    avg_loss = total_loss / num_batches
    avg_bce = total_bce / num_batches
    avg_kld = total_kld / num_batches

    return avg_loss, avg_bce, avg_kld


def main():
    # Hyperparameters
    latent_dim = 20
    beta = 1.0
    batch_size = 128
    num_epochs = 20
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load MNIST dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    print("Creating VAE model...")
    model = VAE(latent_dim=latent_dim, beta=beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        'train_loss': [],
        'train_bce': [],
        'train_kld': [],
        'test_loss': [],
        'test_bce': [],
        'test_kld': []
    }

    # Train
    print(f"Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_loss, train_bce, train_kld = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_bce, test_kld = test(model, test_loader, device)

        history['train_loss'].append(train_loss)
        history['train_bce'].append(train_bce)
        history['train_kld'].append(train_kld)
        history['test_loss'].append(test_loss)
        history['test_bce'].append(test_bce)
        history['test_kld'].append(test_kld)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, BCE: {train_bce:.4f}, KLD: {train_kld:.4f}")
        print(f"  Test  - Loss: {test_loss:.4f}, BCE: {test_bce:.4f}, KLD: {test_kld:.4f}")

    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), './vae_model.pth')

    # Plotting
    print("Generating plots...")
    plot_training_history(history)
    plot_reconstructions(model, test_loader, device)
    plot_latent_space(model, test_loader, device)

    print("Training completed!")


def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['test_loss'], label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)

    # BCE loss
    axes[1].plot(history['train_bce'], label='Train')
    axes[1].plot(history['test_bce'], label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Reconstruction Loss (BCE)')
    axes[1].legend()
    axes[1].grid(True)

    # KLD loss
    axes[2].plot(history['train_kld'], label='Train')
    axes[2].plot(history['test_kld'], label='Test')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('./training_history.png', dpi=150)
    print("Saved training history plot to training_history.png")
    plt.close()


def plot_reconstructions(model, test_loader, device, n_samples=10):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x[:n_samples].to(device)
        recon_x, _, _, _ = model(x)

        x = x.cpu()
        recon_x = recon_x.cpu()

        fig, axes = plt.subplots(2, n_samples, figsize=(15, 3))
        for i in range(n_samples):
            axes[0, i].imshow(x[i, 0], cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')

            axes[1, i].imshow(recon_x[i, 0], cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig('./reconstructions.png', dpi=150)
        print("Saved reconstruction plot to reconstructions.png")
        plt.close()


def plot_latent_space(model, test_loader, device):
    model.eval()
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            mu, _ = model.encoder(x)
            latent_vectors.append(mu.cpu())
            labels.append(y)

    latent_vectors = torch.cat(latent_vectors)
    labels = torch.cat(labels)

    # Plot first 2 dimensions using t-SNE would be ideal, but let's use 2D projection
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1],
                         c=labels, cmap='tab10', alpha=0.5, s=10)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('Latent Dim 0')
    plt.ylabel('Latent Dim 1')
    plt.title('Latent Space (First 2 Dimensions)')
    plt.tight_layout()
    plt.savefig('./latent_space.png', dpi=150)
    print("Saved latent space plot to latent_space.png")
    plt.close()


if __name__ == "__main__":
    main()
