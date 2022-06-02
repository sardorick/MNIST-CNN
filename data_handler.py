import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
print(device)

def load_data():
    # Define a transform to normalize the data (Preprocessing)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5)) ])

    # Download and load the training data
    trainset    = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset    = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=True)
    return transform, trainset, trainloader, testset, testloader

