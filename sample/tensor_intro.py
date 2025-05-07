import torch

x = torch.rand(4, 4)
y = torch.rand(4, 4)

# these are the same
z = x + y
z = torch.add(x, y)
# subtract
z = x - y
z = torch.sub(x, y)
# multiply
z = x * y
z = torch.mul(x, y)
# divide
z = x / y
z = torch.div(x, y)

# modify y and add x's elements to it piecewise
# in pytorch, every function with a trailing _ will modify the variable it is applied on
y.add_(x)

# slice
# all the rows, only the first column
#print(x[:, 0])
# only the first row, all the columns
#print(x[0, :])
print(x)
# converting to a 1D tensor with 16 elements only works because x has 16 total elements
y = x.view(16)
print(y)