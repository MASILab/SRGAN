import math
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(16)
        self.block3 = ResidualBlock(16)
        #self.block4 = ResidualBlock(16)
        #self.block5 = ResidualBlock(16)
        #self.block6 = ResidualBlock(16)
        self.block7 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16)
        )
        block8 = [UpsampleBLock(16, 1) for _ in range(upsample_block_num)]
        block8.append(nn.Conv3d(16, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        print(x.shape)
        block1 = self.block1(x)
        print('block1',block1.size())
        block2 = self.block2(block1)
        print('block2',block2.size())
        #block3 = self.block3(block2)
        print('block3', block3.size())
        #print(block3.size())
        #block4 = self.block4(block3)
        #print(block4.size())
        #block5 = self.block5(block4)
        #print(block5.size())
        #block6 = self.block6(block5)
        block7 = self.block7(block3)
        print('block7', block7.size())
        block8 = self.block8(block1 + block7)
        print('block8', block8.size())

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),

            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),

            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),

            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(128, 256, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        print('batch size',batch_size)
        print('D input', x.size())
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        print(channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        #self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        #x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
