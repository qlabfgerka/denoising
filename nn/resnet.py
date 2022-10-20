import torch


class Resnet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_conv2d = torch.nn.Conv2d(
            3, 32, kernel_size=(3, 3), padding="same")
        #self.batchnorm2d = torch.nn.BatchNorm2d()
        self.dropout2d = torch.nn.Dropout2d(p=0.5)
        self.relu = torch.nn.ReLU()
        self.second_conv2d = torch.nn.Conv2d(
            32, 32, kernel_size=(3, 3), padding="same")
        self.third_conv2d = torch.nn.Conv2d(
            3, 32, kernel_size=(1, 1), padding="same")

    def forward(self, x, first):
        y = x

        if first:
            x = self.first_conv2d(x)
        else:
            x = self.second_conv2d(x)

        #x = self.batchnorm2d(x)
        x = self.dropout2d(x)
        x = self.relu(x)
        x = self.second_conv2d(x)
        #x = self.batchnorm2d(x)
        x = self.dropout2d(x)

        if first:
            y = self.third_conv2d(y)

        x = x + y

        x = self.relu(x)

        return x
