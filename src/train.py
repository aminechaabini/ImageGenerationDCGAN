import torch
from torch.distributions import transforms
import models

LEARNING_RATE = 0.0005
BATCH_SIZE = 256
IMAGE_SIZE = 64
NUM_CLASSES = 10
EMBED_SIZE = 100
EPOCHS = 150
image_channels = 1
noise_channels = 256
gen_features = 64
disc_features = 64

device = torch.device("cuda")

#load dataset

data_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
])

gen_model  = models.Generator(noise_channels, image_channels, gen_features, NUM_CLASSES, EMBED_SIZE).to(device)
disc_model = models.Discriminator(image_channels, disc_features, NUM_CLASSES, IMAGE_SIZE).to(device)