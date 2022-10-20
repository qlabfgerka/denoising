import torch
from nn.resnet import Resnet


class SecondBranch(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = Resnet()
        self.conv2d = torch.nn.Conv2d(
            32, 8, kernel_size=(1, 1), padding="same")
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.resnet(x, True)
        x = self.resnet(x, False)
        x = self.resnet(x, False)
        x = self.conv2d(x)
        x = self.softmax(x)
        return x
