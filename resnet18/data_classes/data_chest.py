import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import matplotlib.image as mpimg
import nibabel as nib

import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2

DATASET_DIR = "../data"     # Base directory
class_names = ['normal', 'viral pneumonia', 'bacterial pneumonia']  # List of class names
# train_resolution = 224  # Target resolution for training

# Function to load images and labels
def load_combined_data(dataset_dir, split):
    images = []
    labels = []

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, split, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)


            images.append(img_path)
            labels.append(class_index)  # Assign an integer label (0, 1, or 2)

    return np.array(images), np.array(labels)

# Load the dataset
# images, labels = load_combined_data(DATASET_DIR, 'train', train_resolution)

# print(f"Loaded {len(images)} images.")
# print(f"Shape of images: {images.shape}")  # Should be (num_images, height, width, channels)
# print(f"Labels: {np.unique(labels)}")


class ChestDataset(data.Dataset):
    """
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        split (string): If train, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    
    def __init__(self, split="train", transform=None, 
                 noise_type=None, noise_rate=0.2):
        self.root = os.path.expanduser(DATASET_DIR)
        self.transform = transform
        self.split = split  # training set or test set
        self.dataset='chest'
        self.noise_type=noise_type
        self.nb_classes=3

        # load the numpy arrays
        self.images, self.labels = load_combined_data(self.root, self.split)
        self.targets = self.labels
        self.clean_labels = self.labels

        self.train_noisy_labels = self.labels
        # if noise_type !='clean':
        #     # noisify train data
        #     self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
        #     self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state, nb_classes=self.nb_classes)
        #     self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
        #     _train_labels=[i[0] for i in self.train_labels]
        #     self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)
        

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.split == "train" and self.noise_type !='clean':
            return img, self.train_noisy_labels[index], index
        else:
            return img, self.labels[index], index

    def __len__(self):
        return len(self.images)
        
if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.ToTensor(),            # Convert image to tensor
    transforms.Resize((224, 224)),    # Resize to 224x224
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1))   # Repeat channels to make 3-channel image
    transforms.Lambda(lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1))
])
    
    train_dataset = ChestDataset(transform=transform, split='train')
    print(train_dataset[0][0])
    # test_dataset = ChestDataset(transform=transform, split='test')
    # print(len(train_dataset))
    # print(len(test_dataset))
    # i = 0
    # print(train_dataset[i][0].shape)
    # print(train_dataset[0][0].shape)
    # print(train_dataset[4971][0].shape)
    # for a,b,c in train_dataset:
    #     if a.shape[0] != 3:
    #         print(a.shape)
    #         print(i)
    #     i += 1
    # print(dataset[0][0].shape)