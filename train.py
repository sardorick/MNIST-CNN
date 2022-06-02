import numpy as np
import matplotlib.pyplot as plt

import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from data_handler import load_data
from model import CNN
model = CNN()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
transform, trainset, trainloader, testset, testloader = load_data()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
num_epochs = 20
print_every = 40

start_time = time.time()
def fit(model, trainloader, testloader, criterion, optimizer, num_epochs, print_every):
    train_losses = []
    test_losses = []
    accuracy = []
    benchmark = 0.95
    for epoch in range(num_epochs):
        train_epoch_loss = []
        current_loss = 0
        print(f"Epoch: {epoch+1}/{num_epochs}")

        for i, (images, labels) in enumerate(iter(trainloader)):
            
            optimizer.zero_grad()
            
            output = model.forward(images)   # 1) Forward pass
            loss = criterion(output, labels) # 2) Compute loss
            loss.backward()                  # 3) Backward pass
            optimizer.step()                 # 4) Update model
            
            train_epoch_loss.append(loss.item())
            

        model.eval()
        acc_list = []
        with torch.no_grad():
            test_epoch_loss = []
            acc_epoch = []
            for i, (images_test, labels_test) in enumerate(iter(testloader)):
                    probability = F.softmax(model(images_test), dim=1)
                    pred = probability.argmax(dim=1)
                    losst = criterion(probability, labels_test)
                    test_epoch_loss.append(losst.item())
                    acc = (pred == labels_test).sum() / len(labels_test) * 100
                    acc_epoch.append(acc.item())

        mean_acc=sum(acc_epoch)/len(acc_epoch)
        accuracy.append(mean_acc)

        test_loss_mean=sum(test_epoch_loss)/len(test_epoch_loss)
        test_losses.append(test_loss_mean)

        train_loss_mean=sum(train_epoch_loss)/len(train_epoch_loss)
        train_losses.append(train_loss_mean)


        if benchmark < mean_acc:
            torch.save(model.state_dict(),'model_trained.pth')
            state_dict = torch.load('model_trained.pth')
            print(state_dict.keys())
            print(model.load_state_dict(state_dict))
            benchmark = mean_acc
        model.train()

        print(f'Mean epoch loss for train: {train_loss_mean}')
        print(f'Mean epoch loss for test: {test_loss_mean}')
        print(f'Accuracy on epoch: {mean_acc}')

    final_time = time.time()- start_time
    print(f'Training time took: {final_time}')

    x_axis=list(range(num_epochs))
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(x_axis,accuracy, label='Accuracy')
    plt.legend()
    plt.show()



trained = fit(model, trainloader, testloader, criterion, optimizer, num_epochs, print_every)