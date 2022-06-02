import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from model import CNN


from data_handler import load_data
transform, trainset, trainloader, testset, testloader = load_data()

trained_model = torch.load('model_trained.pth')
model = CNN()
model.load_state_dict(trained_model)


def view_classify(img, ps):

    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

images, labels = next(iter(testloader))

model.eval()
for img in images:
    with torch.no_grad():
        logits = model(img.reshape(1, *img.shape))
    ps = F.softmax(logits)
    view_classify(img, ps)
    plt.show()