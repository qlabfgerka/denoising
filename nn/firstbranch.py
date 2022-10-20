import torch


class FirstBranch(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.filter_layer = torch.nn.Conv2d(
            1, 8, kernel_size=(11, 11), padding="same", bias=False)

    def forward(self, x):
        x = self.filter_layer(x)

        return x
