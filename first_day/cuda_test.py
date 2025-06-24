import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(3, 3).to(device)
y = torch.randn(3, 3).to(device)
z = x + y

print(z)

