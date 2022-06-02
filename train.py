# import matplotlib.pyplot as plt
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from data_handler import load_data
from model import CNN
model = CNN()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
transform, trainset, trainloader, testset, testloader = load_data()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
num_epochs = 10
print_every = 40

def fit(model, trainloader, testloader, criterion, optimizer, num_epochs, print_every):
    for epoch in range(num_epochs):
        current_loss = 0
        print(f"Epoch: {epoch+1}/{num_epochs}")

        for i, (images, labels) in enumerate(iter(trainloader)):

            images.resize_(images.size()[0], 784)
            
            optimizer.zero_grad()
            
            output = model.forward(images)   # 1) Forward pass
            loss = criterion(output, labels) # 2) Compute loss
            loss.backward()                  # 3) Backward pass
            optimizer.step()                 # 4) Update model
            
            current_loss += loss.item()
            
            if i % print_every == 0:
                loss_value = current_loss/print_every
                print(f"\tIteration: {i}\t Loss: {current_loss/print_every:.4f}")
                current_loss = 0

        model.eval()
        acc_list = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(iter(testloader)):
                    images.resize_(images.size()[0], 784)
                    probability = F.softmax(model(images), dim=1)
                    pred = probability.argmax(dim=1)
                    acc = (pred == labels).sum() / len(labels) * 100
                    acc_list.append(acc)
        print(f'Mean accuracy is {np.array(acc_list).mean()} after epoch {epoch+1}')

        model.train()

trained = fit(model, trainloader, testloader, criterion, optimizer, num_epochs, print_every)