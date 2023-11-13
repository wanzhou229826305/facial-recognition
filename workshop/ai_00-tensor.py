import torch
import numpy


#创建 [2,2] tensor
t = torch.randn([2,2])
print(t)
t2 = torch.tensor([1,2])
print(t2.shape)

#从 numpy 传教 tensor
arr = numpy.array([[1,2,3],[4,5,6],[7,8,9]])
arr2 = numpy.array([[1,2,3],[4,5,6]])
t3 = torch.from_numpy(arr)
print(t3)

#将tensor 转成 numpy
t4 = torch.tensor([1,2])
t4.numpy()

#创建2个tensor, 并计算空间距离
# Example tensors
tensor1 = torch.tensor([[1., 2.]])
tensor2 = torch.tensor([[3., 4.]])

#tensor相加
#tensor相减
#tensor相乘
#点乘
#矩阵相乘

# Calculate Euclidean distance (P2 distance)
# 方法1，tensor相减，再求模
euclidean_dist_by_norm = torch.norm(tensor1 - tensor2, p=2)
print(euclidean_dist_by_norm)

# 方法2，直接计算空间距离
euclidean_dist_by_cdist = torch.cdist(tensor1,tensor2, p=2)
print(euclidean_dist_by_cdist)

#创建2个tensor, 并计算余弦距离
# Example tensors
tensor1 = torch.tensor([[1., 2.]])
tensor2 = torch.tensor([[3., 4.]])

# Calculate cosine distance
cosine_dist = 1 - torch.cosine_similarity(tensor1.flatten(), tensor2.flatten(), dim=0)

print(cosine_dist)