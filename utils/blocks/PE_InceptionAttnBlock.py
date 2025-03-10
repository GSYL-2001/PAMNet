import torch
import torch.nn as nn


class Inception_Attn_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_kernels=4):
        super(Inception_Attn_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.attn_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=1, kernel_size=(1, 1))
        self.attn_softmax = nn.Softmax(dim=-1)
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv1d(in_channels, out_channels, stride=stride, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        self.first = True

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1)
        # print(x.shape)
        if self.first:
            self.first = False
            self.attn_conv = nn.Conv2d(in_channels=self.out_channels+res.shape[2], out_channels=1, kernel_size=(1, 1)).to(x.device)
            self.I = torch.eye(res.shape[2], device=x.device).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, self.num_kernels).to(x.device)
        # print(x.shape)
        # print(x.shape)
        # print(res.shape)
        # print(self.out_channels)
        # print(res.shape)
        # print(self.I.shape)
        attn = self.attn_softmax(self.attn_conv(torch.cat((res, self.I.repeat(x.shape[0], 1, 1, 1)), dim=-3)))
        # print("attn.shape = ", attn.shape)
        res = res*attn
        # print(res.shape)
        res = torch.sum(res, dim=-1)
        return res

class PE_InceptionAttnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PE_InceptionAttnBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Inception_Attn_Conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.downsample = downsample
        self.first = True
    def forward(self, x):
        if self.first:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.first = False
            self.conv1 = Inception_Attn_Conv(self.in_channels+x.shape[2], self.out_channels, stride=self.stride).to(device)
            self.I = torch.eye(x.shape[2]).to(device).unsqueeze(0)
        identity = x
        out = torch.cat((x, self.I.repeat(x.shape[0], 1, 1)), dim=-2)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out
