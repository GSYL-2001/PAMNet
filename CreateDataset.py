import random
from disba import PhaseDispersion as PD
import numpy as np

# import pandas
t = 100000
n = 10
thk = [2]
depth = [2]
for i in range(1, 10):
    thk.append(thk[i - 1] * 1.1)
    depth.append(depth[i - 1] + thk[i])
thk.append(300)
depth.append(300 + depth[len(depth) - 1])
print(depth)
Vs_half_space = 1500  # m/s
Vp_half_space = Vs_half_space / 0.557  # m/s
rho_half_space = 2.5  # g/cm3
Vs_min = 150  # m/s
Vs_max = 350  # m/s
Vs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Vp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
rho = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tau = 0.9
lambda_s_min = 0.02
lambda_s_max = 0.3
data_all = []

# 预训练模型的随机数种子
# random.seed(9419823)
# np.random.seed(345433)
# 反演模型的随机数种子
random.seed(99824)
np.random.seed(4353)


def go():
    Vs[0] = random.uniform(Vs_min, Vs_max)
    for i in range(1, n):
        rd = random.uniform(0, 1)
        if rd <= tau:
            lambda_s = random.uniform(lambda_s_max, lambda_s_min)
            Vs[i] = Vs[i - 1] * (1 + lambda_s)
        elif rd <= tau + (1 - tau) / 2:
            vs_l = Vs[i - 1] * (1 + lambda_s_max)
            vs_r = min(1200, Vs[i - 1] + 300)
            Vs[i] = random.uniform(vs_l, vs_r)
        else:
            vs_l = Vs[i - 1] * (1 + lambda_s_min)
            vs_r = max(150, Vs[i - 1] - 300)
            Vs[i] = random.uniform(vs_l, vs_r)
    Vs[10] = Vs_half_space
    for i in range(n):
        Vp[i] = Vs[i] / (0.5684 * ((0.005 * depth[i]) ** 0.163))
    Vp[10] = Vp_half_space
    for i in range(n):
        rho[i] = 1.2475 + 0.399 * (0.001 * Vp[i]) - 0.026 * ((0.001 * Vp[i]) * (0.001 * Vp[i]))
    rho[10] = rho_half_space
    for i in range(len(rho)):
        rho[i] *= 1000

    layers = []
    for i in range(11):
        layers.append([thk[i] / 1000, Vp[i] / 1000, Vs[i] / 1000, rho[i]])

    layers = np.array(layers)
    # print(layers)
    pd = PD(*layers.T)
    try:
        cpr = pd(np.flip(np.array([1 / (1.12 ** k) for k in range(36)])), mode=0, wave="rayleigh")
    except Exception as e:
        return
    # print(cpr.velocity)
    y = cpr.velocity
    y = np.flip(y, axis=0)
    # print(y.shape)
    # print(len(Vs))
    data_all.append(np.concatenate([y * 1000, Vs])[0:66])


tau = 0.933
while len(data_all) < 100000:
    go()
    if len(data_all) % 10000 == 0:
        print(len(data_all))

tau = 0.85
while len(data_all) < 140000:
    go()
    if len(data_all) % 10000 == 0:
        print(len(data_all))

# tau = 0.75
# while len(data_all) < 170000:
#     go()
#     if len(data_all) % 10000 == 0:
#         print(len(data_all))
#
# tau = 0.95
# while len(data_all) < 200000:
#     go()
#     if len(data_all) % 10000 == 0:
#         print(len(data_all))

# import matplotlib.pyplot as plt
#
# id = np.random.rand(9) * len(data_all)
# id = id.tolist()
# for i in range(9):
#     id[i] = int(id[i])
#
# import layer
#
# layer.layer_properties = []
# for i in id:
#     layer.layer_properties.append(data_all[i][36:46])
# layer.Main()
# # for i in id:
# #     print(list(data_all[int(i)][32:42]))
#
# colors = ["red", "blue", "green", "orange", "purple", "pink", "cyan", "magenta", "gray"]
# for ii, i in enumerate(id):
#     plt.plot(data_all[i][0:36], [1.12 ** (j) for j in range(36)], color=colors[ii])
#     print(colors[ii], data_all[i][36:46])
# # plt.plot(data_all[-1][0:60])
# # plt.plot(data_all[123][0:60])
# # plt.plot(data_all[13][0:60])
# # plt.plot(data_all[235][0:60])
# plt.gca().invert_yaxis()
# plt.semilogx()
#
# plt.show()

data_all = np.array(data_all)
print(data_all.shape)
np.savetxt("RWdata.csv", data_all, delimiter=',', newline="\n")
# dataframe = pandas.DataFrame(data=data_all)
