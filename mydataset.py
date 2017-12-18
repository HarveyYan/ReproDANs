import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import torch

np.random.seed(1234)


class MYCIFAR10(dsets.CIFAR10):

    def __init__(self, ratio, *args, **kwargs):
        super(MYCIFAR10, self).__init__(*args, **kwargs)

        if self.train:
            instances = self.train_data.shape[0]
            self.validat_data = self.train_data[:int(instances * ratio)]
            self.train_data = self.train_data[int(instances * ratio):]

            self.validat_labels = self.train_labels[:int(instances * ratio)]
            self.train_labels = self.train_labels[int(instances * ratio):]

        else:
            pass

    def get_validat_set(self):

        # transform = transforms.Compose(
        # [transforms.ToTensor(),
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        validat = dsets.CIFAR10(root=self.root,
                                train=False,
                                transform=self.transform,
                                download=False)

        validat.train_data = self.validat_data
        validat.train_labels = self.validat_labels
        del self.validat_data
        del self.validat_labels

        return validat


# exclude 257.clutter
class MYCAL256():
    dir_path = '256_ObjectCategories/'

    def __init__(self, batch_size, test_split_ratio=0.2, valid_split_ratio=0.2):
        self.batch_size = batch_size
        self.test_split_ratio = test_split_ratio
        self.valid_split_ratio = valid_split_ratio

        self.train_data = []
        self.test_data = []
        self.valid_data = []
        self.train_labels = []
        self.test_labels = []
        self.valid_labels = []

        self.legend = {}

        if not os.path.exists(self.dir_path):
            raise RuntimeError('Caltech256 dataset not found in ' + self.dir_path)
        categories = os.listdir(self.dir_path)[:-1]  # in alphabetical order, from 001 to 256
        for category in categories:
            # map labels to objects
            self.legend[category.split('.')[0]] = category.split('.')[1]
            # file names, comprised of labels and id
            files = np.array(os.listdir(self.dir_path + category))
            # if 'greg' in files or 'RENAME2' in files:
            #     print(category)
            #     exit()
            total = files.shape[0]
            permute = np.random.permutation(np.arange(0, total))
            self.test_data = np.concatenate((self.test_data, files[permute[:int(total * test_split_ratio)]]))
            self.valid_data = np.concatenate((self.valid_data, files[permute[int(total * test_split_ratio)]:int(
                total * (test_split_ratio + valid_split_ratio))]))
            self.train_data = np.concatenate(
                (self.train_data, files[permute[int(total * (test_split_ratio + valid_split_ratio)):]]))

        self.test_data = self.test_data[np.random.permutation(np.arange(0, len(self.test_data)))]
        for file in self.test_data:
            self.test_labels.append(int(file[:3])-1)
        self.valid_data = self.valid_data[np.random.permutation(np.arange(0, len(self.valid_data)))]
        for file in self.valid_data:
            self.valid_labels.append(int(file[:3])-1)
        self.train_data = self.train_data[np.random.permutation(np.arange(0, len(self.train_data)))]
        for file in self.train_data:
            try:
                self.train_labels.append(int(file[:3])-1)
            except ValueError:
                print(file)

    def get_train_loader(self, parallel=True):
        return CAL256Iter(self.train_data, self.train_labels, self.batch_size, self.legend, parallel=parallel, mode='train')

    def get_test_loader(self, parallel=True):
        return CAL256Iter(self.test_data, self.test_labels, self.batch_size, self.legend, parallel=parallel, mode='test')

    def get_valid_loader(self, parallel=True):
        return CAL256Iter(self.valid_data, self.valid_labels, self.batch_size, self.legend, parallel=parallel, mode='valid')

class CAL256Iter():
    dir_path = '256_ObjectCategories/'
    data_transforms = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def __init__(self, data, labels, batch_size, legend, parallel, mode):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.legend = legend
        self.parallel = parallel
        self.loaded_data = []
        self.mode = mode
        if parallel is True:
            for file in self.data:
                sub_folder = file.split('_')[0] + '.' + self.legend[file.split('_')[0]] + '/'
                img = Image.open(self.dir_path + sub_folder + file)
                img = self.data_transforms(img.convert('RGB'))
                self.loaded_data.append(img)

    def get_generator(self):
        if not self.parallel:
            if self.mode == 'train' or self.mode=='valid':
                while True:
                    batch = torch.FloatTensor(self.batch_size, 3, 64, 64)
                    batch_labels = torch.LongTensor(self.batch_size)
                    for i, file in enumerate(self.data):
                        sub_folder = file.split('_')[0] + '.' + self.legend[file.split('_')[0]] + '/'
                        img = Image.open(self.dir_path + sub_folder + file)
                        img = self.data_transforms(img.convert('RGB'))
                        batch[i % self.batch_size] = img
                        batch_labels[i % self.batch_size] = self.labels[i]
                        if (i + 1) % self.batch_size == 0:
                            yield (batch, batch_labels)
                    if not (i - 1) % self.batch_size == 0:
                        yield (batch[:(i - 1) % self.batch_size], batch_labels[:(i - 1) % self.batch_size])
            else:
                batch = torch.FloatTensor(self.batch_size, 3, 64, 64)
                batch_labels = torch.LongTensor(self.batch_size)
                for i, file in enumerate(self.data):
                    sub_folder = file.split('_')[0] + '.' + self.legend[file.split('_')[0]] + '/'
                    img = Image.open(self.dir_path + sub_folder + file)
                    img = self.data_transforms(img.convert('RGB'))
                    batch[i % self.batch_size] = img
                    batch_labels[i % self.batch_size] = self.labels[i]
                    if (i + 1) % self.batch_size == 0:
                        yield (batch, batch_labels)
                if not (i - 1) % self.batch_size == 0:
                    yield (batch[:(i - 1) % self.batch_size], batch_labels[:(i - 1) % self.batch_size])

        else:
            if self.mode == 'train' or self.mode=='valid':
                while True:
                    batch = torch.FloatTensor(self.batch_size, 3, 64, 64)
                    batch_labels = torch.LongTensor(self.batch_size)
                    for i, img in enumerate(self.loaded_data):
                        batch[i % self.batch_size] = img
                        batch_labels[i % self.batch_size] = self.labels[i]
                        if (i + 1) % self.batch_size == 0:
                            yield (batch, batch_labels)
                    if not (i - 1) % self.batch_size == 0:
                        yield (batch[:(i - 1) % self.batch_size], batch_labels[:(i - 1) % self.batch_size])
            else:
                batch = torch.FloatTensor(self.batch_size, 3, 64, 64)
                batch_labels = torch.LongTensor(self.batch_size)
                for i, file in enumerate(self.data):
                    sub_folder = file.split('_')[0] + '.' + self.legend[file.split('_')[0]] + '/'
                    img = Image.open(self.dir_path + sub_folder + file)
                    img = self.data_transforms(img.convert('RGB'))
                    batch[i % self.batch_size] = img
                    batch_labels[i % self.batch_size] = self.labels[i]
                    if (i + 1) % self.batch_size == 0:
                        yield (batch, batch_labels)
                if not (i - 1) % self.batch_size == 0:
                    yield (batch[:(i - 1) % self.batch_size], batch_labels[:(i - 1) % self.batch_size])


if __name__ == "__main__":
    cal256 = MYCAL256(100)
    train_loader = cal256.get_train_loader(True).get_generator()
    for batch, batch_labels in train_loader:
        print(batch.size(), batch_labels.size())
