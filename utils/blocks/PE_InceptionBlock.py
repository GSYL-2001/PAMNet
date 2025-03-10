import torch
import torch.nn as nn



class Inception_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_kernels=4, init_weight=True):
        super(Inception_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv1d(in_channels, out_channels, stride=stride, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print(x.shape)
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class PE_InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PE_InceptionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        # self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = Inception_Conv(out_channels, out_channels) # nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.first = True
    def forward(self, x):
        if self.first:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.first = False
            self.conv1 = Inception_Conv(self.in_channels+x.shape[2], self.out_channels, stride=self.stride).to(device) # nn.Conv1d(self.in_channels+x.shape[2], self.out_channels, kernel_size=3, stride=self.stride, padding=1).to(device)
            self.I = torch.eye(x.shape[2]).to(device).unsqueeze(0)
        identity = x

        # print(x.shape)
        out = torch.cat((x,self.I.repeat(x.shape[0], 1, 1)),dim=-2)
        # print(out.shape)
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
