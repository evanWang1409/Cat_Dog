import torch
import torchvision
import torchvision.transforms as transforms

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


test_data_dir = '/Users/evnw/Research/Cats_v_Dogs/data/train-by-class'
test_dataset = datasets.ImageFolder(train_data_dir,
                                          data_transforms)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                             shuffle=True, num_workers=4)

dataset_sizes = len(train_dataset)

classes = train_dataset.classes
print(classes)

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

net = AlexNet()

net.load_state_dict(torch.load('alex_cat_dog_iter100000.pt'))


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))



dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
#imshow(torchvision.utils.make_grid(images))
#plt.show()
outputs = net(images)
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))