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
# architecture parameters
n_hidden = 128
n_labels = 10
image_pixels = 28 * 28

training_epochs = 2
batch_size = 128
q = torch.full((1,128),0.5)
y = torch.full((1,128),0.044715)
z = torch.full((1,128),0.7978845608)
r = torch.full((1,128),1)
t= torch.full((1,128),3) 
def gelu(x):
# 	return 0.5 * x * (torch.tanh((0.044715 * x ** 3 + x) * 0.7978845608)) 
	return  q * x *((torch.tanh(((torch.pow(x,t) * y) + x) * z)) + r)
class Network(nn.Module):
    def __init__(self, inplace=False):
        super(Network, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.fc1 = nn.Linear(image_pixels, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_labels)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)




net = Network()
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(training_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.view(-1,28*28))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
import time
begin=time.perf_counter()
for i in range(1):
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.view(-1,28*28))
            _, predicted = torch.max(outputs.data, axis = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
end=time.perf_counter()
print(end-begin)
print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))



input_names = [ "actual_input_1" ]
output_names = ["my_output"]
dummy_input = torch.rand(128, 28 * 28)
torch.onnx.export(net, dummy_input, "mnist_gelu.onnx", verbose=True, input_names=input_names, output_names=output_names)