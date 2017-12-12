import matplotlib.pyplot as plt
import numpy as np


dan_noise_on_cifar10 = np.load('./Result/DAN_noise_on_CIFAR10.npy')
dan_imagenet_on_cifar10 = np.load('./Result/DAN_imagenet_on_CIFAR10.npy')
vggb_s_on_cifar10 = np.load('./Result/VGGB_S_on_CIFAR10.npy')

print(dan_noise_on_cifar10.shape)
print(dan_imagenet_on_cifar10.shape)
print(vggb_s_on_cifar10.shape)

x = np.linspace(1,19,19,dtype=int)
plt.xlabel('epochs')
plt.ylabel('accuracy(%)')
plt.xticks(x)
plt.yticks(np.arange(10,100,10))
plt.plot(x, dan_noise_on_cifar10[:,1], label='DAN_noise on CIFAR10')
plt.plot(x, np.lib.pad(dan_imagenet_on_cifar10[:,1],(0,8),'constant',constant_values = (0,dan_imagenet_on_cifar10[-1,-1])), label='DAN_imagenet on CIFAR10')
plt.plot(x, np.lib.pad(vggb_s_on_cifar10[:, 1], (0, 1), 'constant', constant_values=(0, vggb_s_on_cifar10[-1, -1])), label='VGGB_S on CIFAR10')
plt.legend()
plt.title('Validation Accuracy')
plt.savefig('./graph/valid_acc.png')