import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

import config
import network


def load_model(device):
    encoder = network.Encoder(config.IMAGE_SIZE, config.EMBEDDING_DIM).to(device)
    decoder = network.Decoder(
        config.EMBEDDING_DIM, config.SHAPE_BEFORE_FLATTENING
    ).to(device)
    vae = network.VAE(encoder, decoder).to(device)

    checkpoint = torch.load(config.MODEL_WEIGHTS_PATH, map_location=device)
    vae.load_state_dict(checkpoint["vae"])
    vae.eval()
    return vae


def save_samples(model, device, num_samples=16, nrow=4, output_name="samples.png"):
    samples = model.sample(num_samples=num_samples, device=device)
    samples = samples.cpu()

    grid = make_grid(samples, nrow=nrow, padding=2)

    plt.figure(figsize=(nrow * 2, (num_samples // nrow) * 2))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    out_path = Path(config.output_dir) / output_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {num_samples} samples to {out_path}")


def main():
    device = config.DEVICE
    vae = load_model(device)
    save_samples(vae, device, num_samples=16, nrow=4, output_name="normally_sampled.png")


if __name__ == "__main__":
    main()
