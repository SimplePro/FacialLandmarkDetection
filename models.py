import torch
import torch.nn as nn

from torchsummary import summary



IN_CHANNELS = "in_channels"
OUT_CHANNELS = "out_channels"
SEPARABLE = "separable"
RELU = "relu"
MAXPOOL = "maxpool"
SKIP = "conv"
NAME = "name"


class DepthwiseSeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=out_channels)

    
    def forward(self, x):
        out = self.pointwise(x)
        out = self.depthwise(out)

        return out


class ResidualBlock(nn.Module):

    def __init__(self, skip_layer, main_layer):
        super().__init__()

        self.skip_layer = nn.Sequential(
            *[self.get_block(block) for block in skip_layer]
        )

        self.main_layer = nn.Sequential(
            *[self.get_block(block) for block in main_layer]
        )

    
    def get_block(self, block):

        name = block[NAME]

        if name == SEPARABLE:
            return nn.Sequential(
                DepthwiseSeparableConv2d(
                    in_channels=block[IN_CHANNELS], out_channels=block[OUT_CHANNELS]
                ),
                nn.BatchNorm2d(block[OUT_CHANNELS])
            )

        elif name == RELU:
            return nn.ReLU()

        elif name == MAXPOOL:
            return nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        elif name == SKIP:
            return nn.Sequential(
                nn.Conv2d(in_channels=block[IN_CHANNELS], out_channels=block[OUT_CHANNELS], kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(block[OUT_CHANNELS])
            )


    def forward(self, x):
        skip = self.skip_layer(x)
        out = self.main_layer(x)

        return out + skip


class EntryFlow(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(
                skip_layer=[
                    {NAME: SKIP, IN_CHANNELS: 64, OUT_CHANNELS: 128},
                ],
                main_layer=[
                    {NAME: SEPARABLE, IN_CHANNELS: 64, OUT_CHANNELS: 128},
                    {NAME: RELU},
                    {NAME: SEPARABLE, IN_CHANNELS: 128, OUT_CHANNELS: 128},
                    {NAME: MAXPOOL}
                ]
            ),
            ResidualBlock(
                skip_layer=[
                    {NAME: SKIP, IN_CHANNELS: 128, OUT_CHANNELS: 256},
                ],
                main_layer=[
                    {NAME: SEPARABLE, IN_CHANNELS: 128, OUT_CHANNELS: 256},
                    {NAME: RELU},
                    {NAME: SEPARABLE, IN_CHANNELS: 256, OUT_CHANNELS: 256},
                    {NAME: MAXPOOL}
                ]
            ),
            ResidualBlock(
                skip_layer=[
                    {NAME: SKIP, IN_CHANNELS: 256, OUT_CHANNELS: 512},
                ],
                main_layer=[
                    {NAME: SEPARABLE, IN_CHANNELS: 256, OUT_CHANNELS: 512},
                    {NAME: RELU},
                    {NAME: SEPARABLE, IN_CHANNELS: 512, OUT_CHANNELS: 512},
                    {NAME: MAXPOOL}
                ]
            ),
        )


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res_blocks(out)

        return out


class MiddleFlow(nn.Module):

    def __init__(self, repeat_n=4):
        super().__init__()

        self.repeat_n = repeat_n

        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                skip_layer=[],
                main_layer=[
                    {NAME: RELU},
                    {NAME: SEPARABLE, IN_CHANNELS: 512, OUT_CHANNELS: 512},
                    {NAME: RELU},
                    {NAME: SEPARABLE, IN_CHANNELS: 512, OUT_CHANNELS: 512},
                    {NAME: RELU},
                    {NAME: SEPARABLE, IN_CHANNELS: 512, OUT_CHANNELS: 512}
                ]
            )
            for _ in range(repeat_n)
        ])

    
    def forward(self, x):
        out = x

        for i in range(self.repeat_n):
            out = self.res_blocks[i](out)

        return out


class ExitFlow(nn.Module):

    def __init__(self):
        super().__init__()

        self.res_block = ResidualBlock(
            skip_layer=[
                {NAME: SKIP, IN_CHANNELS: 512, OUT_CHANNELS: 1024}
            ],
            main_layer=[
                {NAME: RELU},
                {NAME: SEPARABLE, IN_CHANNELS: 512, OUT_CHANNELS: 512},
                {NAME: RELU},
                {NAME: SEPARABLE, IN_CHANNELS: 512, OUT_CHANNELS: 1024},
                {NAME: MAXPOOL}
            ]
        )

        self.conv = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=1024, out_channels=1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            DepthwiseSeparableConv2d(in_channels=1024, out_channels=1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        out = self.res_block(x)
        out = self.conv(out)

        return out

    
class Xception(nn.Module):

    def __init__(self, middle_repeat_n):
        super().__init__()

        self.entry_flow = EntryFlow()
        self.middle_flow = MiddleFlow(repeat_n=middle_repeat_n)
        self.exit_flow = ExitFlow()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 68*2)
        )

    def forward(self, x):
        out = self.entry_flow(x)
        out = self.middle_flow(out)
        out = self.exit_flow(out)
        out = self.fc(out)

        return out


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xception = Xception(middle_repeat_n=8).to(device)

    summary(xception, (3, 512, 512))
    
    # dummy = torch.randn((16, 3, 512, 512)).to(device)

    # out = xception(dummy)
    # print(out.shape)