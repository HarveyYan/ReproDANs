import numpy as np
import torch
import torchvision
from torchvision.models import vgg13
from torch.autograd import Variable

np.random.seed(1234)
vgg_b_legend = [64, 64, 128, 128, 256, 256, 512, 512, 512, 512]
num_conv_layers = 10


# class VGG(nn.Module):
#
#     def __init__(self, features, num_classes=1000):
#         super(VGG, self).__init__()
#         self.features = features
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes),
#         )
#         self._initialize_weights()
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
#
#
# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)
#
#
# class VGG_B(torch.nn.Module):
#     def __init__(self, features, num_classes=1000):
#         super(VGG_B, self).__init__()
#         self.features = make_layer_with_controller()
#         self.classifier = torch.nn.Sequential(
#             torch.nn.Linear(512 * 7 * 7, 4096),
#             torch.nn.ReLU(True),
#             torch.nn.Dropout(),
#             torch.nn.Linear(4096, 4096),
#             torch.nn.ReLU(True),
#             torch.nn.Dropout(),
#             torch.nn.Linear(4096, num_classes),
#         )
#         self._initialize_weights()
#
#     def make_layer(self):
#         pass



# configuration B:
# 'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
# initialization of vgg13 assembles the network for us, directly

def _test():
    print('dummy example')
    input = torch.FloatTensor(np.abs(np.random.randn(1, 3, 224, 224))) # batch size is 1, 3 channels, with height and width of 224
    model = vgg13(pretrained=False)
    # model.register_forward_hook()
    # type(model) # presumably nn.Module
    print(model) # for illustration purpose
    #model.eval()
    output = model.forward(torch.autograd.Variable(input))  # must add torch.autograd.Variable, otherwise behaviour unpredictable
    print(output)

def data_preprocess():
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes

def vgg_net(pretrained = False, type='original', freeze_kernels = False):
    # Todo, specify cuda device id
    if type == 'original':
        return vgg13(pretrained = False)#.cuda()
    elif type == 'CIFAR10':
        model = vgg13(pretrained=False)
        controller_params = []
        if freeze_kernels:
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    for name, param in module.named_parameters():
                        # This is to assume that the model has been properly trained or using a
                        # pre-trained model. We don't make further change to the kernel values.
                        # Instead we combine the kernels linearly using weights in cont_param
                        if name == 'weight':
                            param.require_grad = False
                            # kernels with size [out_channels, in_channels, height, width]

                            cont_param = torch.nn.Parameter(torch.randn(param.size()[0], param.size()[0]))
                            comb_param = flatten_combine_deflat(cont_param, param)
                            module._parameters['weight'] = comb_param
                            controller_params.append(cont_param)
        # 224 = 32*7, height and width for CIFAR images are of 32*32
        # size = 1 # 32/32
        # classifiers are trained from scratch
        classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 1 * 1, 4096), #modified
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 10), #modified number of classes
        )
        model.classifier = classifier
        model._initialize_weights()
        return model, controller_params

def flatten_combine_deflat(cont_param, base_param):
    size = base_param.size()
    return cont_param.mm(base_param.view(size[0],-1)).view(*size)

# Todo, use modulelist, and write own vgg
def controller(network):
    parameters = network.parameters()
    for i, param in enumerate(parameters):
        if i %2 == 0:
            controller_param = torch.nn.Parameter()
            new_param = combine_kernel(controller_param, param)



def train(network, train_loader):
    criterion = torch.nn.CrossEntropyLoss()
    # check sourcecode to know details of Adam Optimizer
    optimizer = torch.optim.Adam(network.parameters())

    running_loss = 0.0
    for epoch in range(10):
        for i, (inputs, labels) in enumerate(train_loader, 0): # start at the first data
            # inputs are of batch size 4, 3*32*32
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() # perform a single optimization step

            running_loss += loss.data[0]
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    save_model(network, optimizer)
    return network

def test(network, test_loader):
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        outputs = network(Variable(images))
        _, index = torch.max(outputs.data, 1)
        total += labels.size(0) # or shape[0]
        correct += (index == labels).sum()
    print('Accuracy of %d test examples is: %d %%'%(i, 100*correct/total))

def save_model(network, optimizer):
    torch.save({'net_state':network.state_dict(), 'optim_state':optimizer.state_dict()}, './model_param/param.pth.tar')

if __name__ == "__main__":
    train_loader, test_loader, classes = data_preprocess()
    network, cont_params = vgg_net(type = 'CIFAR10', freeze_kernels= True)
    train(network, train_loader)
    test(network, test_loader)
