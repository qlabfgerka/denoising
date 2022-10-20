import torch
from nn.firstbranch import FirstBranch
from nn.secondbranch import SecondBranch


class NeuralNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_branch = FirstBranch()
        self.second_branch = SecondBranch()

    def forward(self, x, r, g, b):
        r = self.first_branch(r)
        g = self.first_branch(g)
        b = self.first_branch(b)

        x = self.second_branch(x)

        r = x * r
        g = x * g
        b = x * b

        r = torch.sum(r, dim=1)
        g = torch.sum(g, dim=1)
        b = torch.sum(b, dim=1)

        return torch.stack((r, g, b), dim=1)
