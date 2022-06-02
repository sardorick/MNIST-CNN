
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3, 32, 5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(32, 16, 5)
        self.fc1=nn.Linear(64, 28, 28, 128)
        self.fc2=nn.Linear(120, 64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x=x.view(x.shape[0], -1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        x=F.log_softmax(x, dim=1)
        return x

model = CNN()
print(model)