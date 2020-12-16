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




model1 = 'accurate_mnist_gelu.onnx.so'
session1 = ExecutionSession(model1, "run_main_graph")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=3,drop_last=True)
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.view(128,28*28)
        images=images.numpy()
        outputs = torch.tensor(session1.run(images)[0])
        _, predicted = torch.max(outputs.data, axis = 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %f %%' % (
    100 * correct / total))

model2 = 'super-approximate_mnist_gelu.onnx.so'
session2 = ExecutionSession(model2, "run_main_graph")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=3,drop_last=True)
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.view(128,28*28)
        images=images.numpy()
        outputs = torch.tensor(session2.run(images)[0])
        _, predicted = torch.max(outputs.data, axis = 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %f %%' % (
    100 * correct / total))

model3 = 'approximate_mnist_gelu.onnx.so'
session3 = ExecutionSession(model3, "run_main_graph")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=3,drop_last=True)
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.view(128,28*28)
        images=images.numpy()
        outputs = torch.tensor(session3.run(images)[0])
        _, predicted = torch.max(outputs.data, axis = 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %f %%' % (
    100 * correct / total))

begin=time.perf_counter()
for i in range(30):
    for data in testloader:
        images, labels = data
        images = images.view(128,28*28)
        images=images.numpy()
        outputs = torch.tensor(session1.run(images)[0])
end=time.perf_counter()
print(end-begin)

begin=time.perf_counter()
for i in range(30):
    for data in testloader:
        images, labels = data
        images = images.view(128,28*28)
        images=images.numpy()
        outputs = torch.tensor(session2.run(images)[0])
end=time.perf_counter()
print(end-begin)

begin=time.perf_counter()
for i in range(30):
    for data in testloader:
        images, labels = data
        images = images.view(128,28*28)
        images=images.numpy()
        outputs = torch.tensor(session3.run(images)[0])
end=time.perf_counter()
print(end-begin)