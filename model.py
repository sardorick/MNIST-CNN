
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=nn.Conv2d(1, 32, 5, padding=1, stride=1)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(32, 16, 5)
        self.fc1=nn.Linear(16*4*4, 128)
        self.fc2=nn.Linear(128, 64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(x.shape[0], -1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        #x=F.softmax(x, dim=1)
        return x

model = CNN()
print(model)