import torch.nn as nn
import torch.nn.functional as F


class Decoder1(nn.Module):

    def __init__(self):
        super(Decoder1, self).__init__()
        self.convT1 = nn.ConvTranspose2d(8, 16, 3, stride=2)
        self.convT2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)
        self.convT3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.convT1(x))
        x = F.relu(self.convT2(x))
        x = F.tanh(self.convT3(x))
        return x


class Decoder2(nn.Module):

    def __init__(self):
        super(Decoder2, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x 

