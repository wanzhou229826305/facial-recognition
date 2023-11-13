import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

################################################
#Prepare data
x = np.linspace(-2*np.pi, 2*np.pi, 400)
y = np.cos(x)

X = x.reshape(400, -1)
Y = y.reshape(400, -1)


dataset = TensorDataset(torch.tensor(X, dtype=torch.float),
                        torch.tensor(Y, dtype=torch.float))
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

################################################
#Model defintion
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_features=1, out_features=10),
                                nn.ReLU(),
                                nn.Linear(10, 100),
                                nn.ReLU(),
                                nn.Linear(100, 10),
                                nn.ReLU(),
                                nn.Linear(10, 1))

    def forward(self, x):
        return self.net(x)


net = Net()

cost = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

################################################
#Train Loop
for epoch in range(10):
    loss = None
    for batch_x, batch_y in dataloader:
        predict = net(batch_x)
        loss = cost(predict, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print('step: {0}, loss:{1}'.format(epoch+1, loss.mean()))

###############################################
#Test
test_x = np.linspace(-4*np.pi, 4*np.pi, 800)
test_y = np.cos(test_x)

test_x_reshape = test_x.reshape(800, -1)
test_x_tensor = torch.tensor(test_x_reshape,dtype=torch.float)
predict = net(test_x_tensor)
###############################################
#Show
plt.figure(figsize=(12, 7), dpi=160)
plt.plot(test_x, test_y, label='True', marker='X')
plt.plot(test_x, predict.detach().numpy(), label='Predict', marker='o')
plt.xlabel('x', size=15)
plt.ylabel('cos(x)', size=15)
plt.xticks(rotation=30, size=15)
plt.yticks(size=15)

plt.show()
