import torch

# Use float64 for accuracy
dtype = torch.float64
eps = 1e-6

# Define the function
def f(x):
    return x

# Create scalar tensor with grad tracking
x = torch.tensor(1.2345, dtype=dtype, requires_grad=True)

# Compute loss and autograd gradient
y = f(x)
y.backward()
grad_autograd = x.grad.item()

# Finite difference
x_val = x.detach()
f_plus = f(x_val + eps)
f_minus = f(x_val - eps)
grad_fd = (f_plus - f_minus) / (2 * eps)

# Print results
print(f"x = {x_val}")
print(f"Autograd gradient:         {grad_autograd:.12f}")
print(f"Finite difference gradient:{grad_fd:.12f}")
print(f"Absolute error:            {abs(grad_autograd - grad_fd):.2e}")
