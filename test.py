import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
# 定义超参数
LR = 0.01
BATCH_SIZE = 32
EPOCH = 10
# 生成数据
# x.size() : 1000 * 1
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim = 1)
# y = x^2 + 0.1 * ...
y = x.pow(2) + 0.1  * torch.normal(torch.zeros(x.size()))
# 绘制数据图像
plt.scatter(x.numpy(), y.numpy())
plt.show()