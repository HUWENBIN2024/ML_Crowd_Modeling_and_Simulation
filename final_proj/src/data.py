import torch
import torchvision

batch_size = 128

# Mnist
train_data = torch.utils.data.DataLoader(
       torchvision.datasets.MNIST('../data',
                     train=True,
                     transform=torchvision.transforms.ToTensor(),
                     download=True),
                     batch_size=batch_size,
                     shuffle=True)

test_data = torch.utils.data.DataLoader(
       torchvision.datasets.MNIST('../data',
                     train=False,
                     transform=torchvision.transforms.ToTensor(),
                     download=True),
                     batch_size=batch_size,
                     shuffle=True)

# Cifar 10

# Swiss Roll
