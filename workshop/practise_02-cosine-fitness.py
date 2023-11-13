import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

################################################
#生成训练数据，即标记
#取消下面注释
# x = np.linspace(-2*np.pi, 2*np.pi, 400)
# y = np.cos(x)

# X = x.reshape(400, -1)
# Y = y.reshape(400, -1)

###############################################
#准备 tensor dataset
#请练习输入 1

################################################
#定义模型
#请练习输入 2


################################################
#定义一个模型变量
#请练习输入 3


################################################
#定义顺势函数与优化器
#请练习输入 4

################################################
#开始循环训练
#请练习输入 5

###############################################
#Test
#请练习输入 6


###############################################
#请练习输入 7 附加题
#生成 -4PI - 4PI 的测试数据，并对其进行测试

###############################################
#显示预测与期望结果
#取消下面注释
# plt.figure(figsize=(12, 7), dpi=160)
# plt.plot(x, y, label='True', marker='X')
# plt.plot(x, predict.detach().numpy(), label='Predict', marker='o')
# plt.xlabel('x', size=15)
# plt.ylabel('cos(x)', size=15)
# plt.xticks(rotation=30, size=15)
# plt.yticks(size=15)

# plt.show()
