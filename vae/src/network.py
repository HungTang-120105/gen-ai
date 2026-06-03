import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class Sampling(nn.Module): 
    def forward(self, z_mean, z_log_var):
        batch, dim = z_mean.shape
        epsilon = Normal(0,1).sample((batch, dim)).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var)*epsilon  
    

class Encoder(nn.Module):
    def __init__(self, image_size, embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride = 2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride = 2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride = 2, padding=1)
        
        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(
            128 * (image_size // 8) * (image_size // 8),
            embedding_dim
        )
        self.fc_log_var = nn.Linear(
            128 * (image_size // 8) * (image_size // 8),
            embedding_dim
        )
        
        self.sampling = Sampling()
        
    def forward(self, X):
        # X.shape = (batch_size, channel, H, W)
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = self.flatten(X)
        Z_mean = self.fc_mean(X)
        Z_log_var = self.fc_log_var(X)
        Z = self.sampling(Z_mean, Z_log_var)
        return (Z_mean, Z_log_var, Z)
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim, shape_before_flattening):
        super().__init__()
        self.fc = nn.Linear(
            embedding_dim,
            shape_before_flattening[0] *
            shape_before_flattening[1] *
            shape_before_flattening[2] 
        )
        
        self.reshape = lambda x : x.view(-1, *shape_before_flattening)
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, 3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            32, 1, 3, stride=2, padding=1, output_padding=1
        )
        
    def forward(self, X):
        X = self.fc(X)
        X = self.reshape(X)
        X = F.relu(self.deconv1(X))
        X = F.relu(self.deconv2(X))
        X = torch.sigmoid(self.deconv3(X))
        
        return X

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, X):
        Z_mean, Z_log_var, Z = self.encoder(X)
        reconstruction = self.decoder(Z)
        
        return Z_mean, Z_log_var, reconstruction
    
    def sample(self, num_samples: int = 1, device: torch.device = None, z: torch.Tensor = None) -> torch.Tensor:
        """Generate samples by decoding random latent vectors.

        If `z` is provided use it (shape [N, latent_dim]); otherwise draw from N(0,1).
        Returns decoded images tensor with shape [N, C, H, W].
        """
        if device is None:
            device = next(self.parameters()).device

        latent_dim = self.decoder.fc.in_features
        if z is None:
            z = torch.randn(num_samples, latent_dim, device=device)
        else:
            z = z.to(device)

        self.eval()
        with torch.no_grad():
            imgs = self.decoder(z)

        return imgs
        