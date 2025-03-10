import torch
import torch.nn as nn



class PE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PE_ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.first = True
    def forward(self, x):
        if self.first:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.first = False
            self.conv1 = nn.Conv1d(self.in_channels+x.shape[2], self.out_channels, kernel_size=3, stride=self.stride, padding=1).to(device)
            self.I = torch.eye(x.shape[2]).to(device).unsqueeze(0)
        identity = x

        # print(x.shape)
        out = torch.cat((x, self.I.repeat(x.shape[0], 1, 1)), dim=-2)
        # print(out.shape)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

