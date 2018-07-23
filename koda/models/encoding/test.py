import torch
import torch.nn as nn

from encoder import Encoder2 
from decoder import Decoder2 


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

autoencoder = autoencoder()
encoder = Encoder2()
decoder = Decoder2()

x = torch.randn(4, 1, 28, 28) 
latent = encoder(x)
out1 = decoder(latent)

out2 = autoencoder(x)

print(f'Original dimensions: {x.size()}')
print(f'Latent space has size {latent.size()}')
print(f'My model has size {out1.size()}')
print(f'AutoEncoder output size: {out2.size()}')
