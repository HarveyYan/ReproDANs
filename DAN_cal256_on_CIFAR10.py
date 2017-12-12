from VGG13 import MYVGG13
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import os
from mydataset import MYCIFAR10
import numpy as np

'''
Hyperparameters
'''
torch.manual_seed(1234)
dataset_identifier = {
    'MNIST': 0,
    'CIFAR10': 1
}
image_size = {
    'MNIST': (28, 28),
    'CIFAR10': (32, 32)
}
batch_size = 100
n_iters = 10 ** 5
validation_ratio = 0.1

# pad_vgg options pad all images to 224*224
# an easy way out, that we could simply use the original vgg model
def get_data(dataset_name, pad_vgg=False):
    pad = []
    if pad_vgg is True:
        size = image_size[dataset_name]
        pad.append(int((224 - size[1]) / 2))  # padding to the left
        pad.append(int((224 - size[1]) / 2))  # padding to the right
        pad.append(int((224 - size[0]) / 2))  # padding to the top
        pad.append(int((224 - size[0]) / 2))  # padding to the bottom
        transform = transforms.Compose(
            [
                transforms.Pad(pad),
                # transforms.Scale(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    train_dataset = MYCIFAR10(ratio=validation_ratio,
                                              root='./data',
                                              train=True,
                                              transform=transform,
                                              download=True)

    validat_dataset = train_dataset.get_validat_set()

    test_dataset = dsets.CIFAR10(root='./data',
                                                 train=False,
                                                 transform=transform,
                                                 download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    validate_loader = torch.utils.data.DataLoader(dataset=validat_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, validate_loader, test_loader, len(train_dataset), len(test_dataset)



def train(model, train_loader, validate_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())#, lr=0.1)
    # num_epochs is the upper bound
    num_epochs = int(n_iters / (len_train / batch_size))
    iter = 0
    best_acc = 0
    best_path = None
    failure = 0

    validat_loss_acc = []
    out_file = open('./Result/DAN_noise_on_CIFAR10.txt', 'a')
    out_file.write('TRAINING PHASE\n')
    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter += 1

            if iter % 500 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through validate dataset
                for images, labels in validate_loader:
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                    else:
                        images = Variable(images)
                    outputs = model(images)
                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    # Total correct predictions
                    if torch.cuda.is_available():
                        correct += (predicted.cpu() == labels.cpu()).sum()
                    else:
                        correct += (predicted == labels).sum()

                accuracy = 100 * correct / total
                # Print Loss
                print('Iteration: {}. Val_Loss: {}. Val_Accuracy: {}'.format(iter, loss.data[0], accuracy))
                out_file.write('Iteration: {}. Val_Loss: {}. Val_Accuracy: {}'.format(iter, loss.data[0], accuracy) + '\n')
                validat_loss_acc.append([loss.data[0],accuracy])

                if accuracy >= best_acc:
                    best_acc = accuracy
                    best_path = model.state_dict()
                    failure = 0
                else:
                    failure += 1

        if failure >= 3:
            break
    out_file.close()
    np.save('./Result/DAN_noise_on_CIFAR10.npy', np.array(validat_loss_acc))
    model.load_state_dict(best_path)
    return model


def _test(model, test_loader):
    correct = 0
    total = 0
    out_file = open('./Result/DAN_noise_on_CIFAR10.txt', 'a')
    out_file.write('TESTING PHASE\n')
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = Variable(images.cuda())
        else:
            images = Variable(images)
        outputs = model(images)
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # Total correct predictions
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    # Print Loss
    print('Test Data Accuracy: {}'.format(accuracy))
    out_file.write('Test Data Accuracy: {}'.format(accuracy) + '\n')
    out_file.close()


def get_model():
    model = MYVGG13('CIFAR10', pretrained=False, num_classes=10)
    # this step is imperative
    if torch.cuda.is_available():
        model.cuda()
    return model


# LOAD DATA
train_loader, validate_loader, test_loader, len_train, len_test = get_data('CIFAR10')

# LOAD MODEL
model = get_model()
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        module.add_controller()
model = train(model, train_loader, validate_loader)
_test(model, test_loader)