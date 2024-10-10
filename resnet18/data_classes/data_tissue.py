import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO, Evaluator
import numpy as np
from collections import Counter

# Download TissueMNIST dataset
class TissueMNISTDataset(medmnist.TissueMNIST):
    def __init__(self, split, size=None, transform=None, download=True, noise_type="symmetric", noise_rate=40):
        super().__init__(split=split, size=size, transform=transform, download=download)

        self.targets = np.array(self.labels.flatten(), dtype=int)

        self.clean_labels = self.targets.copy()
 
        if split == "train":
            if noise_type != "clean":
                loaded_targets = torch.load(f'../targets_tissue_{noise_type}_{noise_rate}.pt')
                self.targets = np.array(loaded_targets)
            actual_noise_rate = (self.targets != self.clean_labels).mean()
            print("Actual Noise Rate: ", actual_noise_rate)

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img, self.targets[index], index

    def __len__(self):
        return len(self.imgs)
    
   

if __name__ == "__main__":    
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
    print("Start")
    train_set = TissueMNISTDataset(split='test', transform=transform, download=True)
    print(train_set.targets)
    # print(Counter(train_set.targets))
