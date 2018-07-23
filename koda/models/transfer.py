import torch.nn as nn
import torch.nn.functional as F


class TransferNet(nn.Module):

    def __init__(self, original_model):
        super(TransferNet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children()))
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1) 
