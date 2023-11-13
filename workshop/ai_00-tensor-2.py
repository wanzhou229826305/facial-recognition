import torch
import numpy


w = torch.tensor([1.],requires_grad=True)
x = torch.tensor([2.],requires_grad=False)
a = torch.add(x,w)
b = torch.add(w,1)
y = torch.mul(a,b)

#f=(x+w)*(w+1) = w2 + (x+1)w + x 
#df(w) = 2w + x +1
y.backward()
print(w.grad)
print(x.grad)

# -------------
w = torch.tensor([1.],requires_grad=True)
x = torch.tensor([2.],requires_grad=True)

for i in range(3):
    a = torch.add(x,w)
    b = torch.add(w,1)
    y = torch.mul(a,b)

    y.backward()
    print(w.grad)
    w.grad.zero_()

# --------------
a = torch.rand(3,4)
b = torch.rand(4)
x = a+b
