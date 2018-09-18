import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from matplotlib import pyplot as plt
import numpy as np

data_transforms = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(227),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_data_dir = '/Users/evnw/Research/Cats_v_Dogs/data/train_by_class'

test_dataset = datasets.ImageFolder(test_data_dir, data_transforms)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

dataset_sizes = len(test_dataset)
classes = test_dataset.classes

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

net = AlexNet()
net.load_state_dict(torch.load('alex_cat_dog_iter10000.pt'))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))



dataiter = iter(testloader)
images, labels = dataiter.next()

""" sample display
# print images
imshow(torchvision.utils.make_grid(images))
plt.show()
outputs = net(images)
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
"""

correct = 0
total = 0
with torch.no_grad():
    count = 0
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        count += 1
        if count%100 == 0:
            print(count)
        if(count == 2000):
            break

print('Accuracy of the network on the 2000 test images: %d %%' % (
    100 * correct / total))