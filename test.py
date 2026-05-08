import torch
a = torch.randint(0 ,10 ,(2,3))
x = torch.ones((a.shape[0],1)) * 2
print(x)