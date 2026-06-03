import torch
import torch.nn as nn

def vae_gaussian_kl_loss(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)
    return KLD.mean()

def reconstruction_loss(reconstruction, x):
    bce_loss = nn.BCELoss()
    return bce_loss(reconstruction, x)

def vae_loss(y_pred, y_true):
    mu, logvar, reconstruction = y_pred
    recon_loss = reconstruction_loss(reconstruction, y_true)
    kld_loss = vae_gaussian_kl_loss(mu, logvar)
    return 100 * recon_loss + kld_loss