# pip install pillow
from PIL import Image, ImageEnhance,ImageFilter

# pip install matplotlib
import matplotlib.pyplot as plt

# pip install torch torchvision torchaudio 
import torch, torchvision

#pip install numpy
import numpy as np

# pip install tqdm
import tqdm 

#pip install opencv-python
import cv2

#create numpy
arr = np.array([[1,2,3],[4,5,6]])
print(arr)
print(arr)

# Create tensor
t1 = torch.from_numpy(arr)
print(t1)
print(t1.shape)

#Update tensor
t1[1,2] = 9
print(t1)

print('Hello, PyTorch, Hello world')