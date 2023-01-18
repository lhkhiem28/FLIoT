import os, sys
from libs import *

class CNN3(nn.Module):
    def __init__(self, 
        image_size = 32, in_channels = 3, 
        num_classes = 2, 
    ):
        super(CNN3, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, out_channels = 32, 
                kernel_size = 5, padding = "same", 
            ), 
            nn.ReLU(), 
            nn.MaxPool2d(
                kernel_size = 2, stride = 2, 
            ), 
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 32, out_channels = 64, 
                kernel_size = 5, padding = "same", 
            ), 
            nn.ReLU(), 
            nn.MaxPool2d(
                kernel_size = 2, stride = 2, 
            )
        )
        self.denses = nn.Sequential(
            nn.Linear(
                int(((image_size/4)**2)*64), 512, 
            ), 
            nn.ReLU(), 
        )

        self.classifier = nn.Linear(
            512, num_classes, 
        )

    def forward(self, 
        x, 
    ):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.denses(x.view(x.shape[0], -1))

        output = self.classifier(x)

        return output