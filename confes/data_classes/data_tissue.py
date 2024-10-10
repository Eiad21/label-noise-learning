import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO, Evaluator
import numpy as np

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
        # if split == "train":
        #     self.noisify_data(noise_percentage=0.6, confused_classes=[6,0,5,4,3,2,0,6])

        # num_noisy_samples = np.sum(self.clean_labels != self.targets)
        # total_samples = len(self.targets)
        
        # # Calculate the percentage of noisy samples
        # true_noise_percentage = (num_noisy_samples / total_samples) * 100
        # print(f"True noise percentage: {true_noise_percentage:.2f}%")

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img, self.targets[index], index, self.clean_labels[index]

    def __len__(self):
        return len(self.imgs)
    
    def noisify_data(self, noise_percentage, confused_classes):
        """
        Adds label noise to simulate mislabeled samples.

        :param noise_percentage: Percentage of labels to corrupt (between 0 and 1).
        :param confused_classes: An array of 8 elements where each index `i` represents the class 
                                 that class `i` can be confused with.
        """
        print("custom noise ", noise_percentage)
        assert 0 <= noise_percentage <= 1, "Noise percentage must be between 0 and 1."
        assert len(confused_classes) == 8, "confused_classes must be an array of 8 elements."

        # Determine the number of samples to corrupt based on the noise percentage
        num_samples = len(self.targets)
        num_noisy_samples = int(noise_percentage * num_samples)

        # Randomly select indices to apply noise
        noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)

        # Apply noise by changing the labels based on confused_classes mapping
        for idx in noisy_indices:
            original_class = self.targets[idx]
            self.targets[idx] = confused_classes[original_class]

    def balance_classes(self):
        label_counts = np.bincount(self.targets)
        min_count = np.min(label_counts)
        
        indices_to_keep = []
        for label in np.unique(self.labels):
            label_indices = np.where(self.targets == label)[0]
            if len(label_indices) > min_count:
                label_indices = np.random.choice(label_indices, min_count, replace=False)
            indices_to_keep.extend(label_indices)
        
        self.imgs = self.imgs[indices_to_keep]
        self.targets = self.targets[indices_to_keep]

    def over_under(self):
        n = len(self.imgs)  # Total number of samples
        classes, _ = np.unique(self.targets, return_counts=True)  # Unique classes and their counts
        c = len(classes)  # Number of unique classes
        target_count = n // c  # Target samples per class

        # Create new lists for balanced images and targets
        new_imgs = []
        new_targets = []

        for cls in classes:
            # Get all indices of this class
            cls_indices = np.where(self.targets == cls)[0]

            if len(cls_indices) > target_count:
                # Drop samples randomly if class has more than target_count
                cls_indices = np.random.choice(cls_indices, target_count, replace=False)
            elif len(cls_indices) < target_count:
                # Augment samples by duplicating if class has fewer than target_count
                num_to_add = target_count - len(cls_indices)
                cls_indices = np.concatenate([cls_indices, np.random.choice(cls_indices, num_to_add, replace=True)])

            # Add selected samples to new list
            new_imgs.extend(self.imgs[cls_indices])
            new_targets.extend(self.targets[cls_indices])

        # Update imgs and targets to balanced versions
        self.imgs = np.array(new_imgs)
        self.targets = np.array(new_targets)

        # # Update clean_labels to match the new targets
        self.clean_labels = self.targets
        print("Over-under applied")

        label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

        # Iterate through the dataset
        for label in self.targets:
            # Get the label for the current sample
            label_counts[label] += 1

        print(label_counts)

if __name__ == "__main__":    
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    train_set = TissueMNISTDataset(split='train', transform=transform, download=True)
    print(train_set[0][0].shape)
    # train_set.over_under()
    # train_dataloader = torch.utils.data.DataLoader(dataset=train_set,
                                                # batch_size=128)

    # for image, label, index in train_dataloader:
    #     print(label[0])
    #     break

    # train_set.balance_classes()
    # label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

    # # Iterate through the dataset
    # for i, (image, label, _, clean) in enumerate(train_set):
    #     # Get the label for the current sample
    #     label_counts[clean] += 1

    # print(label_counts)