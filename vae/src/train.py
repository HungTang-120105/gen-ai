import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

import config, network
from trainer import Trainer
from util import vae_loss

# define the transformation to be applied to the data
transform = transforms.Compose(
    [transforms.Pad(padding=2), transforms.ToTensor()]
)
# load the MNIST training data and create a dataloader
trainset = datasets.MNIST(
    "data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=config.BATCH_SIZE, shuffle=True
)
# load the MNIST test data and create a dataloader
testset = datasets.MNIST(
    "data", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=config.BATCH_SIZE, shuffle=True
)

# instantiate the encoder and decoder models
encoder = network.Encoder(config.IMAGE_SIZE, config.EMBEDDING_DIM).to(
    config.DEVICE
)
decoder = network.Decoder(
    config.EMBEDDING_DIM, config.SHAPE_BEFORE_FLATTENING
).to(config.DEVICE)
# pass the encoder and decoder to VAE class
vae = network.VAE(encoder, decoder)

# instantiate optimizer and scheduler
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=config.LR
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)

trainer = Trainer(
    model = vae,
    optimizer = optimizer,
    criterion = vae_loss,
    train_loader = train_loader,
    scheduler = scheduler,
    device = config.DEVICE,
    val_loader = test_loader,
)

trainer.fit(10)