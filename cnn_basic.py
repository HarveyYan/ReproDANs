import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import functional
import os
import numpy as np
from mydataset import MYCIFAR10
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

    train_dataset = eval('MY' + dataset_name)(ratio=validation_ratio,
                                              root='./data',
                                              train=True,
                                              transform=transform,
                                              download=True)

    validat_dataset = train_dataset.get_validat_set()

    test_dataset = eval('dsets.' + dataset_name)(root='./data',
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


class ContConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super(ContConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        comb_weight = self.weight.mm(self.old_filters.view(self.out_channels, -1)).view(self.out_channels,
                                                                                        self.in_channels,
                                                                                        *self.kernel_size)
        return functional.conv2d(input, comb_weight, self.bias, self.stride,
                                 self.padding, self.dilation, self.groups)

    def add_controller(self):

        '''

                w = Variable(
            torch.Tensor(s).copy_(conv.weight.data))  # Copy the elements from original weights and save them to w
        w = w.view(s[0], -1)  # flatten the weights
        # print('w1',w)
        self.w = w.detach().cuda()  # Returns a new Variable, detached from the current graph
        '''
        # save old filters and bias, place controller module parameters

        self.old_filters = Variable(self.weight.data).detach()
        self.old_bias = Variable(self.bias.data).detach()

        # old_filters = Variable(torch.FloatTensor(list(self.weight.size())).copy_(self.weight.data))
        # self.old_filters = old_filters.detach().cuda()
        # old_bias = Variable(torch.FloatTensor(list(self.bias.size())).copy_(self.bias.data))
        # self.old_bias = old_bias.detach().cuda()

        # self.old_filters = self.weight.clone().detach()
        # self.old_bias = self.bias.clone().detach()
        # self.old_bias.requires_grad.detach()
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.eye(self.out_channels, self.out_channels).cuda())
            self.bias = nn.Parameter(torch.randn(self.out_channels).cuda())
        else:
            self.weight = nn.Parameter(torch.eye(self.out_channels))
            self.bias = nn.Parameter(torch.randn(self.out_channels))


# TODO, expand to VGG model
class CNNModel(nn.Module):

    def __init__(self, dataset_id, load_state):
        super(CNNModel, self).__init__()
        self.dataset_id = dataset_id
        self.load_state = load_state
        self.make_layer()

    def make_layer(self):
        # load_state = True means we are going to experiment on transfer learning;
        # otherwise, it's only an initial learning
        if self.load_state:
            conv_layer = ContConv2d
        else:
            conv_layer = nn.Conv2d

        # Convolution 1
        if self.dataset_id == 0:
            # MNIST dataset, in channel is 1
            self.cnn1 = conv_layer(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        else:
            # CIFAR dataset, in channel is 3
            self.cnn1 = conv_layer(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = conv_layer(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        if self.dataset_id == 0:
            self.fc1 = nn.Linear(32 * 4 * 4, 10)
        else:
            self.fc1 = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2 
        out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out

    def load_state_dict(self, state_dict):
        # super(CNNModel, self).load_state_dict(state_dict)

        own_state = self.state_dict()
        for name, param in state_dict.items():

            # only copy the filters in convolutional layers
            if not name.lower().startswith('cnn'):
                continue

            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))

            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                raise

        # for stationary model structure, there shouldn't be missing states
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))
        # add controller modules here
        # for module in self.modules():
        #     if isinstance(module, ContConv2d):
        #         module.add_controller()

    def add_controller(self):
        cont_params = []
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                for name, param in module.named_parameters():
                    if name == 'weight':
                        param.require_grad = False
                        cont_param = nn.Parameter(torch.randn(param.size()[0], param.size()[0]))
                        comb_param = cont_param.mm(param.view(param.size()[0], -1)).view(*param.size())
                        module._parameters['weight'] = comb_param
                        cont_params.append(cont_param)
        return cont_params

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])

    def save(self, best_path, name):
        torch.save({'state_dict': best_path}, './model_params/'+name+'.pth')


def get_model(dataset_id, load_state=True):
    model = CNNModel(dataset_id, load_state)
    if load_state:
        if os.path.exists('./model_params/param.pth.tar'):
            model.load('./model_params/param.pth.tar')
        else:
            raise RuntimeError('./model_params/param.pth.tar' + ' does not exist')

    if torch.cuda.is_available():
        model.cuda()
    return model


def train(model, train_loader, validate_loader, save_param = False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # num_epochs is the upper bound
    num_epochs = int(n_iters / (len_train / batch_size))
    iter = 0
    best_acc = 0
    best_path = None
    failure = 0

    validat_loss_acc = []
    out_file = open('./Result/Simple_CNN_Outcome.txt', 'a')
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
    # if not os.path.exists('./model_params/param.pth.tar'):
    #     model.save(best_path)

    if save_param:
        for dataset_name, dataset_id in dataset_identifier.items():
            if dataset_id == model.dataset_id:
                break
        model.save(best_path, dataset_name)

    np.save('./Result/Simple_CNN_MNIST_CIFAR10.npy', np.array(validat_loss_acc))
    model.load_state_dict(best_path)
    return model


def _test(model, test_loader):
    correct = 0
    total = 0
    out_file = open('./Result/Simple_CNN_Outcome.txt', 'a')
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


if __name__ == '__main__':
    # should get dataset data from argparse
    dataset_name = 'CIFAR10'
    train_loader, validate_loader, test_loader, len_train, len_test = get_data(dataset_name)
    init_model = get_model(dataset_identifier[dataset_name], load_state=False)
    model = train(init_model, train_loader, validate_loader)
    _test(model, test_loader)
