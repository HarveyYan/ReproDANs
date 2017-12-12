import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
from cnn_basic import ContConv2d
import math

# pad all image to size 224
class MYVGG13(nn.Module):

    legend = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    model_urls = {
        'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    }

    dataset_identifier = {
        'MNIST': 0,
        'CIFAR10': 1
    }

    # pre-trained option allows us to use parameters from the imagenet
    def __init__(self, dataset, pretrained = False, num_classes = 1000, batch_norm = True):
        super(MYVGG13, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.batch_norm = batch_norm

        self.features = self.make_layers()
        self.classifiers = self.make_classifier(dataset)

        if pretrained:
            if batch_norm:
                self.load_state_dict(model_zoo.load_url(self.model_urls['vgg13_bn']))
            else:
                self.load_state_dict(model_zoo.load_url(self.model_urls['vgg13']))

    def make_layers(self):
        layers = []
        in_channels = 3
        for v in self.legend:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = ContConv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def make_classifier(self, dataset):
        if dataset == 'CIFAR10':
            return nn.Sequential(
                nn.Linear(512 * 1 * 1, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, self.num_classes),
            )
        else:
            return nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, self.num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifiers(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ContConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def load_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            # only copy the filters in convolutional layers
            if name.lower().startswith('classifier'):
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
        # for controlled modules, we should expect classifiers to be missing
        # missing = set(own_state.keys()) - set(state_dict.keys())
        # if len(missing) > 0:
        #     raise KeyError('missing keys in state_dict: "{}"'.format(missing))

        # DON'T DO IT HERE!!!!!!!!!!!!!!!!!!!
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

    def save(self, best_path):
        torch.save({'state_dict': best_path}, './model_params/param.pth.tar')

