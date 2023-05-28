import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_channels, img_size, num_filters, num_classes):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.model = nn.Sequential(
            # Input: (img_channels, img_size, img_size)
            nn.Conv2d(img_channels+1, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.label_emb = nn.Embedding(num_classes, img_size*img_size)

    def forward(self, x, labels):
        label_embed = self.label_emb(labels).view(labels[0], 1, self.img_size, self.img_size)
        latent_input = torch.cat([x, label_embed], dim=1)
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, img_size, num_filters, num_classes, embed_size):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # Input: (latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim+embed_size, num_filters * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_filters * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_filters, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.label_emb = nn.Embedding(num_classes, embed_size)

    def forward(self, noise, labels):
        labels_embed = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        latent_input = torch.cat((noise, labels_embed), dim=1)
        return self.model(latent_input)
