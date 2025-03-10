import numpy as np
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from utils import network
from utils.dataset import get_data, Normalization01, split_data
from time import time
import torch
from utils.arguments import get_args
from torch.utils.data import DataLoader, TensorDataset

from utils.noise_and_pe import Noise_and_PE

args = get_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
train_id = time() if args.id is None else args.id

print(args.data_path)
# 数据处理
y, x = get_data(args.data_path)
x1, y1, x2, y2, x3, y3 = split_data(x, y)
y1, (y_mx, y_mn) = Normalization01(y1)
x1, (x_mx, x_mn) = Normalization01(x1)
mx = np.concatenate((y_mx, x_mx))
mn = np.concatenate((y_mn, x_mn))
x1 = torch.from_numpy(x1).to(device).float()
y1 = torch.from_numpy(y1).to(device).float()
x2 = torch.from_numpy(x2).to(device).float()
y2 = torch.from_numpy(y2).to(device).float()
y1 = torch.unsqueeze(y1, dim=1)
y2 = torch.unsqueeze(y2, dim=1)
x_mn = torch.from_numpy(x_mn).to(device).float()
y_mn = torch.from_numpy(y_mn).to(device).float()
x_mx = torch.from_numpy(x_mx).to(device).float()
y_mx = torch.from_numpy(y_mx).to(device).float()
x2 = (x2 - x_mn) / (x_mx - x_mn)
y2 = (y2 - y_mn) / (y_mx - y_mn)
NaP = Noise_and_PE(args, torch.concatenate((y1, y2), dim=0))
_ = np.concatenate((mx[np.newaxis, :], mn[np.newaxis, :]), axis=0)
np.savetxt(f"data/NormalData_{train_id}_invModel.csv", _, delimiter=',')
model = None
if args.model == 'pe_resnet':
    model = network.pe_resnet14(10, 1+NaP.PE_Channle()).to(device)
elif args.model == 'resnet':
    model = network.resnet14(10, 1+NaP.PE_Channle()).to(device)
elif args.model == 'pamnet':
    model = network.incption_attn_resnet14(10, 1+NaP.PE_Channle()).to(device)
loss_calculater = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
train_loder = DataLoader(
    dataset=TensorDataset(x1, y1),
    batch_size=args.batch_size,
    shuffle=True
)
test_loder = DataLoader(
    dataset=TensorDataset(x2, y2),
    batch_size=args.batch_size,
    shuffle=True
)


# 模型训练与保存
train_loss_all = []
train_mae_all = []
test_loss_all = []
test_mae_all = []
for epoch in range(args.epochs):
    loop = tqdm(train_loder)
    loop.set_description(f"{epoch}/{args.epochs}")
    train_loop = 0
    train_loss = 0
    train_mae = 0
    model.train()
    for x, y in loop:
        train_loop += 1
        # print(pe.shape)
        # print(y.shape)
        output = model(NaP(y))
        # output = model(y)
        # print(output.shape)
        # print(x.shape)
        dx = output-x
        loss = torch.mean(torch.square(dx))
        mae = torch.mean(torch.abs(dx))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_mae += mae.item()
        test_loop = 1
        test_loss = 0
        test_mae = 0
        if train_loop == loop.total:
            model.eval()
            with torch.no_grad():
                test_loop = 0
                for xx, yy in test_loder:
                    # ooutput = model(yy)
                    ooutput = model(NaP(yy))
                    dx = xx - ooutput
                    loss = torch.mean(torch.square(dx))
                    mae = torch.mean(torch.abs(dx))
                    test_loop += 1
                    test_loss += loss.item()
                    test_mae += mae.item()
                train_loss_all.append(train_loss/train_loop)
                train_mae_all.append(train_mae/train_loop)
                test_loss_all.append(test_loss/test_loop)
                test_mae_all.append(test_mae/test_loop)
            model.train()
        loop.set_postfix(test_loss=test_loss / test_loop,
                         test_mae=test_mae / test_loop,
                         train_loss=train_loss / train_loop,
                         train_mae=train_mae / train_loop
                         )
    scheduler.step()
torch.save(model, f=f"./models/inversion_model_{train_id}.pth")

# 绘图与绘图数据保存
import matplotlib.pyplot as plt
def make_plot(data, names, val_name, model_name):
    for i,d in enumerate(data):
        plt.plot(d, label=f"{names[i]}_{val_name}")
    plt.title(f"{val_name} for {model_name}")
    plt.xlabel('epoch')
    plt.ylabel(f'{val_name}')
    plt.legend()
    plt.savefig(f"./plots/inv_{train_id}_{val_name}.png")
    plt.cla()

make_plot([train_loss_all, test_loss_all],
          ["train", "test"],
          "loss",
          "inversion model"
          )

make_plot([train_mae_all, test_mae_all],
          ["train", "test"],
          "mae",
          "inversion model"
          )


np.savetxt(f'loss_{train_id}.csv',
           np.array([train_loss_all, test_loss_all,
                     train_mae_all, test_mae_all]),
           delimiter=','
           )




