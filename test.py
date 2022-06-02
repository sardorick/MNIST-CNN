import matplotlib.pyplot as plt
import numpy as np
import torch
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

# images, labels = next(iter(testloader))
# images.resize_(images.shape[0], 1, 784)
# logits = model.forward(images[0,:])
# ps = F.softmax(logits, dim=1)
# view_classify(images[0].view(1, 28, 28), ps)