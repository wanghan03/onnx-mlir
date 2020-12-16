import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import math
import torchvision.transforms as transforms
from PyRuntime import ExecutionSession
import sys
import onnx
import time




model = 'approximate_mnist_gelu.onnx.so'
session = ExecutionSession(model, "run_main_graph")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=3,drop_last=True)
correct = 0
total = 0

begin=time.perf_counter()
for i in range(30):
    for data in testloader:
        images, labels = data
        images = images.view(128,28*28)
        images=images.numpy()
        outputs = torch.tensor(session.run(images)[0])
end=time.perf_counter()
print(end-begin)

# model = 'super-approximate_mnist_gelu.onnx.so'
# session = ExecutionSession(model, "run_main_graph")

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5), (0.5))])
# testset = torchvision.datasets.MNIST(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128,
#                                          shuffle=False, num_workers=3,drop_last=True)
# correct = 0
# total = 0

# begin=time.perf_counter()
# for i in range(30):
#     for data in testloader:
#         images, labels = data
#         images = images.view(128,28*28)
#         images=images.numpy()
#         outputs = torch.tensor(session.run(images)[0])
# end=time.perf_counter()
# print(end-begin)

# model = 'accurate_mnist_gelu.onnx.so'
# session = ExecutionSession(model, "run_main_graph")

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5), (0.5))])
# testset = torchvision.datasets.MNIST(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128,
#                                          shuffle=False, num_workers=3,drop_last=True)
# correct = 0
# total = 0

# begin=time.perf_counter()
# for i in range(30):
#     for data in testloader:
#         images, labels = data
#         images = images.view(128,28*28)
#         images=images.numpy()
#         outputs = torch.tensor(session.run(images)[0])
# end=time.perf_counter()
# print(end-begin)