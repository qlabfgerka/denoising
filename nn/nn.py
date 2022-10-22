import torch
from nn.secondbranch import SecondBranch


class NeuralNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.filter_layer = torch.nn.Conv2d(
            1, 8, kernel_size=(11, 11), padding="same", bias=False)
        self.second_branch = SecondBranch()

    def forward(self, x, r, g, b):
        r = self.filter_layer(r)
        g = self.filter_layer(g)
        b = self.filter_layer(b)

        x = self.second_branch(x)

        r = torch.multiply(x, r)
        g = torch.multiply(x, g)
        b = torch.multiply(x, b)

        r = torch.sum(r, dim=1, keepdim=True)
        g = torch.sum(g, dim=1, keepdim=True)
        b = torch.sum(b, dim=1, keepdim=True)

        return torch.cat((r, g, b), dim=1)
