import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from math import inf

import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import v2

from data_classes.data_tissue import TissueMNISTDataset
from data_classes.data_chest import ChestDataset


class Dataset:

    def __init__(self,
                 dataset_name,
                 data_dir='./data',
                 noise_type=None,
                 noise_rate=None,
                 random_seed=1,
                 device=torch.device('cuda')
                 ):
        self.dataset_name = dataset_name
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.device = device
        self.random_seed = random_seed
        self.train_sampler = None
        self.test_sampler = None

        if self.dataset_name == "cifar10":
            cifar_mean = [0.4914, 0.4822, 0.4465]
            cifar_std = [0.2023, 0.1994, 0.2010]
            transform_pipe = [transforms.RandomHorizontalFlip(),
                              transforms.RandomCrop(32, 4),
                              transforms.ToTensor(),
                              transforms.Normalize(cifar_mean, cifar_std)]

            self.train_set = Cifar10(root=data_dir,
                                     train=True,
                                     transform=transforms.Compose(transform_pipe),
                                     download=True)

            self.test_set = Cifar10(root=data_dir,
                                    train=False,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(),
                                         transforms.Normalize(cifar_mean, cifar_std)]),
                                    download=True)
            self.num_classes = 10
            self.input_size = 32 * 32 * 3
            self.is_noisy = []
            if noise_type is not None:
                self.clean_labels = torch.tensor(self.train_set.cifar10.targets)
                train_noisy_labels_tensor, is_noisy_labels = self.make_labels_noisy()
                self.is_noisy = is_noisy_labels[:]
                self.train_set.cifar10.targets = train_noisy_labels_tensor.detach()

        elif self.dataset_name == "cifar100":
            cifar_mean = [0.507, 0.487, 0.441]
            cifar_std = [0.267, 0.256, 0.276]
            transform_pipe = [transforms.RandomHorizontalFlip(),
                              transforms.RandomCrop(32, 4),
                              transforms.ToTensor(),
                              transforms.Normalize(cifar_mean, cifar_std), ]
            self.train_set = Cifar100(root=data_dir,
                                      train=True,
                                      transform=transforms.Compose(transform_pipe),
                                      download=True)
            self.test_set = Cifar100(root=data_dir,
                                     train=False,
                                     transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.Normalize(cifar_mean,
                                                                                        cifar_std)]),
                                     download=True)
            self.num_classes = 100
            self.input_size = 32 * 32 * 3
            if noise_type is not None:
                self.clean_labels = torch.tensor(self.train_set.cifar100.targets)
                train_noisy_labels_tensor, is_noisy_labels = self.make_labels_noisy()
                self.is_noisy = is_noisy_labels[:]
                self.train_set.cifar100.targets = train_noisy_labels_tensor.detach()

        elif self.dataset_name == "tissue":
            # Define transformations for the dataset

            transform = transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])

            s = 128
            self.num_classes = 8
            self.input_size = s * s * 3
            self.train_set = TissueMNISTDataset(split='train', size=s, transform=transform, download=True, 
                                                noise_type=noise_type, noise_rate=noise_rate)
            self.test_set = TissueMNISTDataset(split='test', size=s, transform=transform, download=True)

            self.clean_labels = torch.tensor(self.train_set.clean_labels)

        elif self.dataset_name == "chest":
            s = 128
            train_transforms = v2.Compose([
                v2.Resize(132),
                v2.RandomResizedCrop(size=(s, s), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                # v2.RandomRotation(degrees=(-20, 20)),
                v2.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
                v2.RandomErasing(p=0.5, scale=(0.1,0.15)),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

            ])

            test_transforms = v2.Compose([
                v2.Resize((s,s)),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
            
            self.num_classes = 3
            self.input_size = s * s * 3
            self.train_set = ChestDataset(split='train', transform=train_transforms)
            self.test_set = ChestDataset(split='test', transform=test_transforms)

            self.clean_labels = torch.tensor(self.train_set.clean_labels)

        print("Train set Size: ", len(self.train_set))
        print("Test set Size: ", len(self.test_set))

    def set_target(self, key, target):
        if self.dataset_name == "cifar10":
            self.train_set.cifar10.targets[key] = target
        elif self.dataset_name == "cifar100":
            self.train_set.cifar100.targets[key] = target
        elif self.dataset_name == "tissue":
            self.train_set.targets[key] = target
        elif self.dataset_name == "clothing1m":
            pass # TODO: to be implemented
        else:
            raise Exception("Not handled")

    def get_targets(self):
        if self.dataset_name == "cifar10":
            return self.train_set.cifar10.targets
        elif self.dataset_name == "cifar100":
            return self.train_set.cifar100.targets
        elif self.dataset_name == "tissue":
            return self.train_set.targets
        elif self.dataset_name == "clothing1m":
            return None # TODO: to be implemented
        else:
            raise Exception("Not handled")      
         
    # ------------------------------------------------------------------------------------------------------------------
    # Taken from https://github.com/xiaoboxia/CDR/blob/6665a8ba265f0f60291ed7775042575db05bed61/utils.py
    def make_labels_noisy(self):
        clean_labels_np = self.clean_labels.detach().numpy()
        clean_labels_np = clean_labels_np[:, np.newaxis]
        m = clean_labels_np.shape[0]
        noisy_labels = clean_labels_np.copy()

        is_noisy = m * [None]
        if self.noise_rate is None:
            raise ValueError("Noise rate needs to be specified ....")

        if self.noise_type == "symmetric":
            noise_matrix = self.compute_noise_transition_symmetric()

        elif self.noise_type == "instance":
            noise_matrix = self.compute_noise_transition_instance()
        
        elif self.noise_type == "pairflip":
            noise_matrix = self.compute_noise_transition_pairflip()


        print('Size of noise transition matrix: {}'.format(noise_matrix.shape))

        if self.noise_type == "symmetric" or self.noise_type == "pairflip":
            assert noise_matrix.shape[0] == noise_matrix.shape[1]
            assert np.max(clean_labels_np) < noise_matrix.shape[0]
            assert_array_almost_equal(noise_matrix.sum(axis=1), np.ones(noise_matrix.shape[1]))
            assert (noise_matrix >= 0.0).all()

            flipper = np.random.RandomState(self.random_seed)
            for idx in np.arange(m):
                i = clean_labels_np[idx]
                flipped = flipper.multinomial(1, noise_matrix[i, :][0], 1)[0]
                noisy_labels[idx] = np.where(flipped == 1)[0]
                is_noisy[idx] = (noisy_labels[idx] != i)[0]
        elif self.noise_type == "instance":
            l = [i for i in range(self.num_classes)]
            for idx in np.arange(m):
                noisy_labels[idx] = np.random.choice(l, p=noise_matrix[idx])
                is_noisy[idx] = (noisy_labels[idx] != clean_labels_np[idx])[0]

        # noise_or_not = (noisy_labels != clean_labels_np)
        actual_noise_rate = (noisy_labels != clean_labels_np).mean()
        assert actual_noise_rate > 0.0
        print('Actual_noise_rate : {}'.format(actual_noise_rate))
        return torch.tensor(np.squeeze(noisy_labels)), is_noisy


    # ------------------------------------------------------------------------------------------------------------------
    # Taken from https://github.com/xiaoboxia/CDR/blob/6665a8ba265f0f60291ed7775042575db05bed61/utils.py
    def compute_noise_transition_symmetric(self):
        noise_matrix = np.ones((self.num_classes, self.num_classes))
        noise_matrix = (self.noise_rate / (self.num_classes - 1)) * noise_matrix

        if self.noise_rate > 0.0:
            # 0 -> 1
            noise_matrix[0, 0] = 1. - self.noise_rate
            for i in range(1, self.num_classes - 1):
                noise_matrix[i, i] = 1. - self.noise_rate
            noise_matrix[self.num_classes - 1, self.num_classes - 1] = 1. - self.noise_rate
            # print(noise_matrix)
        return noise_matrix

    # ------------------------------------------------------------------------------------------------------------------
    # Taken from https://github.com/xiaoboxia/CDR/blob/6665a8ba265f0f60291ed7775042575db05bed61/tools.py
    def compute_noise_transition_instance(self):
        clean_labels = self.clean_labels
        norm_std = 0.1
        np.random.seed(int(self.random_seed))
        torch.manual_seed(int(self.random_seed))
        torch.cuda.manual_seed(int(self.random_seed))

        noise_matrix = []
        flip_distribution = stats.truncnorm((0 - self.noise_rate) / norm_std,
                                            (1 - self.noise_rate) / norm_std,
                                            loc=self.noise_rate,
                                            scale=norm_std)
        flip_rate = flip_distribution.rvs(clean_labels.shape[0])

        W = np.random.randn(self.num_classes, self.input_size, self.num_classes)
        W = torch.FloatTensor(W).to(self.device)
        for i, (image, label, _) in enumerate(self.train_set):
            # 1*m *  m*10 = 1*10 = A.size()
            image = image.detach().to(self.device)
            A = image.view(1, -1).mm(W[label]).squeeze(0)
            A[label] = -inf
            A = flip_rate[i] * F.softmax(A, dim=0)
            A[label] += 1 - flip_rate[i]
            noise_matrix.append(A)
        noise_matrix = torch.stack(noise_matrix, 0).cpu().numpy()
        return noise_matrix

    # ------------------------------------------------------------------------------------------------------------------
    #https://github.com/tmllab/PES/blob/54662382dca22f314911488d79711cffa7fbf1a0/common/NoisyUtil.py
    def compute_noise_transition_pairflip(self):

        noise_matrix = np.eye(self.num_classes)

        if self.noise_rate > 0.0:
            # 0 -> 1
            noise_matrix[0, 0], noise_matrix[0,1] = 1. - self.noise_rate, self.noise_rate
            for i in range(1, self.num_classes - 1):
                noise_matrix[i, i], noise_matrix[i, i+1] = 1. - self.noise_rate, self.noise_rate
            noise_matrix[self.num_classes - 1, self.num_classes - 1], noise_matrix[self.num_classes-1, 0] = 1. - self.noise_rate, self.noise_rate
            # print(noise_matrix)
        return noise_matrix


# ----------------------------------------------------------------------------------------------------------------
class Cifar10(Dataset):
    def __init__(self, root, train, transform, download):
        self.cifar10 = datasets.CIFAR10(root=root,
                                        download=download,
                                        train=train,
                                        transform=transform)
        self.is_noisy = []

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)


class Cifar100(Dataset):
    def __init__(self, root, train, transform, download):
        self.cifar100 = datasets.CIFAR100(root=root,
                                          download=download,
                                          train=train,
                                          transform=transform)

    def __getitem__(self, index):
        data, target = self.cifar100[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar100)
            

if __name__ == "__main__":

    ci = Cifar100(root='./data',
                  train=True,
                  transform=None,
                  download=True)
    print(ci.cifar100.targets)
