import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.arguments import get_args
from utils.dataset import get_data, split_data
from utils.noise_and_pe import Noise_and_PE

args = get_args()

tid = args.id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
y, x = get_data("RW_for_invModel.csv")
y_normal, x_normal = get_data(f"data/NormalData_{tid}_invModel.csv")
y = (y - y_normal[1]) / (y_normal[0] - y_normal[1])
x = (x - x_normal[1]) / (x_normal[0] - x_normal[1])
x1, y1, x2, y2, _, _ = split_data(x, y)
x2 = torch.from_numpy(x2).float().to(device)
y2 = torch.from_numpy(y2).float().to(device)
y2 = torch.unsqueeze(y2, dim=1)
y1 = torch.from_numpy(y1).float().to(device)
y1 = torch.unsqueeze(y1, dim=1)
x_normal = [torch.from_numpy(x_normal[0]).float().to(device),
            torch.from_numpy(x_normal[1]).float().to(device)]

model = torch.load(f"models/inversion_model_{tid}.pth").to(device)
# print(model.state_dict())
model.eval()

# # print(model.parameters())
# num = sum(p.numel() for p in model.parameters())
# print(num)
# print([p.numel() for p in model.parameters()])
# print([p for p in model.parameters()])

# did_ = int(92683)
# print(did_)
#
# xx = x[did_]
# xx = xx * (x_normal[0] - x_normal[1]) + x_normal[1]
# print(xx)
# xx = model(
#     torch.cat((y[did_:did_+1], torch.eye(36).to(device).unsqueeze(0).expand([1, -1, -1])), dim=1)
# )
# xx = xx * (x_normal[0] - x_normal[1]) + x_normal[1]
# print(xx)
# exit(0)

dataloader = DataLoader(TensorDataset(x2, y2), batch_size=1)

loop = tqdm(dataloader)
max_mae = 0
avg_mae = 0
max_mre = 0
avg_mre = 0
test_num = 0
__re = torch.tensor([]).to(device)
NaP = Noise_and_PE(args, torch.concatenate((y1, y2), dim=0))
with torch.no_grad():
    for x, y in loop:
        test_num += 1
        x_pred = model(NaP(y))
        ae = torch.abs(x_pred - x)
        mae = torch.mean(ae).item()
        avg_mae += mae
        max_mae = max(max_mae, mae)
        x = x * (x_normal[0] - x_normal[1]) + x_normal[1]
        x_pred = x_pred * (x_normal[0] - x_normal[1]) + x_normal[1]
        re = (torch.abs(x_pred - x) / torch.abs(x))
        _re = torch.mean(re, dim=1).flatten()
        __re = torch.concatenate((__re, _re), dim=0)

        mre = torch.mean(re).item()
        max_mre = max(max_mre, mre)
        avg_mre += mre
        loop.set_postfix(max_mae=max_mae, avg_mae=avg_mae / test_num,
                         max_mre=max_mre, avg_mre=avg_mre / test_num)

__re = __re.cpu().numpy()
# print(__re.shape)

bins = np.arange(start=0, stop=0.4, step=0.05)
# print(bins)
_, _, patches = plt.hist(__re, bins=bins)

for i in range(len(patches)):
    # 计算每个条形的高度 (频次)
    height = patches[i].get_height()

    # 在条形顶部添加文本，显示频次
    plt.text(
        patches[i].get_x() + patches[i].get_width() / 2,  # X位置，条形中心
        height,  # Y位置，条形顶部
        f'{int(height)}',  # 显示频次（整数）
        ha='center',  # 水平居中
        va='bottom',  # 垂直对齐到条形顶部
        color='black'  # 文字颜色
    )
plt.savefig(f"plots/{tid}_hist.png")