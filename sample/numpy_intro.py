import numpy as np
import torch

a = torch.ones(5)

# make a numpy array out of a tensor
b = a.numpy()

# beware! a and b point to the same memory location, thus a and b will be the same after running this
a.add_(1)

# create tensor out of numpy array
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)