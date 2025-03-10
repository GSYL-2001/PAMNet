import math
import sys

import torch
import numpy as np


class onehot:
    def __init__(self):
        self.pe = [[0 for i in range(36)] for i in range(36)]
        for i in range(36):
            self.pe[i][i] = 1
        self.pe = torch.from_numpy(np.array(self.pe)).unsqueeze(0).float()

    def __call__(self, data):
        self.pe = self.pe.to(data.device)
        # print(data.device, file=sys.stderr)
        return torch.cat((data, self.pe.expand([data.shape[0], -1, -1])), dim=1)

    def channle(self):
        return 36


class sincos:
    def __init__(self):
        self.pe = []
        self.pe.append([])
        for i in range(36):
            self.pe[0].append(math.sin(1 + i / 10000))
        self.pe = torch.from_numpy(np.array(self.pe)).unsqueeze(0).float()

    def __call__(self, data):
        self.pe = self.pe.to(torch.device(data.device))
        return torch.cat((data, self.pe.expand([data.shape[0], -1, -1])), dim=1)

    def channle(self):
        return 1


class relative:
    def __init__(self):
        self.pe = []
        self.pe.append([])
        for i in range(36):
            self.pe[0].append(i / 35)
        self.pe = torch.from_numpy(np.array(self.pe)).unsqueeze(0).float()

    def __call__(self, data):
        self.pe = self.pe.to(torch.device(data.device))
        return torch.cat((data, self.pe.expand([data.shape[0], -1, -1])), dim=1)

    def channle(self):
        return 1


class PE:
    def __init__(self, name):
        self.name = name
        self.e = None
        if self.name.lower() == 'onehot':
            self.e = onehot()
        elif self.name.lower() == 'sincos':
            self.e = sincos()
        elif self.name.lower() == 'relative':
            self.e = relative()

    def __call__(self, data):
        if self.e is not None:
            return self.e(data)
        return data

    def channle(self):
        if self.e is not None:
            return self.e.channle()
        return 0


class Noise:
    @torch.no_grad()
    def __init__(self, flag, level, data):
        self.flag = flag
        self.level = level
        self.std = torch.std(data, dim=[0, 1], keepdim=True)

    def __call__(self, data):
        noise = self.level * self.std * torch.randn_like(data)
        return noise + data


class Noise_and_PE:
    def __init__(self, args, y_data):
        self.noise = Noise(args.add_noise, args.noise_level, y_data)
        self.pe = PE(args.position_encoding)

    def __call__(self, data):
        return self.pe(self.noise(data))

    def PE_Channle(self):
        return self.pe.channle()
