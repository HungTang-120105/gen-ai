import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

import config
import network


def build_test_loader():
	transform = transforms.Compose(
		[transforms.Pad(padding=2), transforms.ToTensor()]
	)
	testset = datasets.FashionMNIST(
		"data", train=False, download=True, transform=transform
	)
	return torch.utils.data.DataLoader(
		testset, batch_size=config.BATCH_SIZE, shuffle=False
	)


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


def save_reconstruction_pairs(model, loader, device, num_pairs=8):
	images, _ = next(iter(loader))
	images = images[:num_pairs].to(device)

	with torch.no_grad():
		_, _, reconstructions = model(images)

	images = images.cpu()
	reconstructions = reconstructions.cpu()

	fig, axes = plt.subplots(num_pairs, 2, figsize=(5, 2.2 * num_pairs))
	if num_pairs == 1:
		axes = [axes]

	for idx in range(num_pairs):
		axes[idx][0].imshow(images[idx].squeeze(0), cmap="gray")
		axes[idx][0].set_title("Original")
		axes[idx][0].axis("off")

		axes[idx][1].imshow(reconstructions[idx].squeeze(0), cmap="gray")
		axes[idx][1].set_title("Reconstruction")
		axes[idx][1].axis("off")

	plt.tight_layout()
	output_path = Path(config.output_dir) / "inference_pairs.png"
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved reconstruction pairs to {output_path}")


def main():
	device = config.DEVICE
	loader = build_test_loader()
	model = load_model(device)
	save_reconstruction_pairs(model, loader, device)


if __name__ == "__main__":
	main()
