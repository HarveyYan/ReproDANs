import torchvision.datasets as dsets
import torchvision.transforms as transforms

class MYCIFAR10(dsets.CIFAR10):

    def __init__(self, ratio, *args, **kwargs):
        super(MYCIFAR10, self).__init__(*args, **kwargs)

        if self.train:
            instances = self.train_data.shape[0]
            self.validat_data = self.train_data[:int(instances*ratio)]
            self.train_data = self.train_data[int(instances*ratio):]

            self.validat_labels = self.train_labels[:int(instances*ratio)]
            self.train_labels = self.train_labels[int(instances*ratio):]

        else:
            pass

    def get_validat_set(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        validat = dsets.CIFAR10(root=self.root,
                                train=False,
                                transform=transform,
                                download=False)

        validat.train_data = self.validat_data
        validat.train_labels = self.validat_labels
        del self.validat_data
        del self.validat_labels

        return validat