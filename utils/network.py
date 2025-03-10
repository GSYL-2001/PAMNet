import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.blocks.PE_InceptionAttnBlock import PE_InceptionAttnBlock
from utils.blocks.PE_InceptionBlock import PE_InceptionBlock
from utils.blocks.PE_ResidualBlock import PE_ResidualBlock
from utils.blocks.ResidualBlock import ResidualBlock


class Bias_Tanh(nn.Module):
    def forward(self, x):
        return torch.tanh(x) + 0.5


# 定义残差网络
class ResNet(nn.Module):
    def __init__(self, block, layers, output_num=10, in_channel=11, activation=None):
        super(ResNet, self).__init__()
        self.in_channels = 128

        self.conv1 = nn.Conv1d(in_channel, 128, kernel_size=1, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.layers = []
        for i, layer in enumerate(layers):
            self.layers.append(
                self._make_layer(block,
                                 self.in_channels * (2 if i > 0 else 1),
                                 layer,
                                 stride=(2 if i > 0 else 1)
                                 ).to(torch.device('cuda'))
            )
        self.layers = nn.Sequential(*self.layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.in_channels, output_num)
        self.last_activation = activation

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1),
                nn.AvgPool1d(kernel_size=2, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.last_activation is not None:
            x = self.last_activation(x)

        return x


def pe_resnet14(output_num=10, in_channel=36 + 1):
    return ResNet(PE_ResidualBlock, [2, 2, 2], output_num, in_channel, None)


def resnet14(output_num=10, in_channel=36 + 1):
    return ResNet(ResidualBlock, [2, 2, 2], output_num, in_channel, None)


def incption_resnet14(output_num=10, in_channel=36 + 1):
    return ResNet(PE_InceptionBlock, [2, 2, 2], output_num, in_channel, None)


def incption_attn_resnet14(output_num=10, in_channel=36 + 1):
    return ResNet(PE_InceptionAttnBlock, [2, 2, 2], output_num, in_channel, None)


