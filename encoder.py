import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace conv1 to accept 1-channel mel spectrograms instead of 3-channel RGB.
        # Instead of random reinit, average the pretrained RGB weights into 1 channel
        # so the new conv1 starts close to the pretrained distribution rather than
        # from scratch. This noticeably helps convergence speed and final accuracy
        # on small datasets.
        old_conv1 = model.conv1
        new_conv1 = nn.Conv2d(
            1, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        new_conv1.weight.data = old_conv1.weight.data.mean(dim=1, keepdim=True)
        model.conv1 = new_conv1

        self.encoder = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        return x