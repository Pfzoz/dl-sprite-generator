from typing import Any
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset
from numpy.core.shape_base import block

from dlspritegen import datasets
from dlspritegen.models.discriminator import Discriminator
from dlspritegen.models.generator import Generator

# Labels: 0 = Front Facing Character, 1 = Monster/Enemy/Minion, 2 = Object, 3 = Item/Equipment, 4 = Side Facing Character

device: Any = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Using device: {device}")

BATCH_SIZE = 32
LEARNING_RATE = 0.3
NUM_EPOCHS = 300
LOSS_FUNCTION = nn.BCELoss()
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5))])
TRAIN_DATA = datasets.Pixel16Dataset(transform=TRANSFORM)
DATA_LOADER = DataLoader(TRAIN_DATA, BATCH_SIZE, shuffle=True)

LATENT_SPACE_SIZE = 100

discriminator = Discriminator().to(device=device)
generator = Generator(LATENT_SPACE_SIZE, 16 * 16 * 3).to(device=device)

DISCRIMINATOR_OPT = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
GENERATOR_OPT = optim.Adam(generator.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    loss_discriminator: Any = None
    loss_generator: Any = None
    generated_samples: Tensor = torch.Tensor()
    real_samples: Tensor = torch.Tensor()
    for i, real_samples in enumerate(DATA_LOADER):
        if real_samples.shape[0] != BATCH_SIZE:
            continue

        real_samples = real_samples.to(device=device)

        # Labels
        real_sample_labels = torch.ones((BATCH_SIZE, 1)).to(device=device) # N amount of 1s
        generated_sample_labels = torch.zeros((BATCH_SIZE, 1)).to(device=device) # N amount of 0s

        # Latent Samples
        latent_space_x = torch.randn((BATCH_SIZE, LATENT_SPACE_SIZE)).to(device=device)
        generated_samples = generator(latent_space_x)

        # Joining Samples
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_sample_labels, generated_sample_labels))

        # Training Discriminator
        discriminator.zero_grad() # Resets gradients
        output_discriminator = discriminator(all_samples)
        # print(output_discriminator.shape, all_samples_labels.shape)
        loss_discriminator = LOSS_FUNCTION(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        DISCRIMINATOR_OPT.step()

        # Training Generator

        latent_space_x = torch.randn((BATCH_SIZE, LATENT_SPACE_SIZE)).to(device=device)
        generator.zero_grad()
        generated_samples = generator(latent_space_x)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = LOSS_FUNCTION(output_discriminator_generated, real_sample_labels)
        loss_generator.backward()
        GENERATOR_OPT.step()

    print(f"Epoch: {epoch} Loss D.: {loss_discriminator} Loss G.: {loss_generator}")

    if epoch % 5 == 0:
        cpu_samples = generated_samples.cpu().detach()
        real_samples = real_samples.cpu().detach()
        plt.imshow(to_pil_image(cpu_samples[0]))
        plt.show()
        plt.imshow(to_pil_image(real_samples[0]))
        plt.show()
