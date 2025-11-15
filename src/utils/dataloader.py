from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from medmnist import BreastMNIST, INFO

from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import INFO
from medmnist.dataset import BreastMNIST
import numpy as np

class BreastMNISTLoader:
    def __init__(self, batch_size, image_size, use_rgb=False):
        """
        Args:
            batch_size: Batch size for dataloaders
            image_size: Target image size
            use_rgb: If True, uses 3-channel RGB. If False, uses 1-channel grayscale
        """
        self.batch_size = batch_size
        self.image_size = image_size
        self.use_rgb = use_rgb
        self.info = INFO['breastmnist']
        self.DataClass = BreastMNIST

        if use_rgb:
            # RGB mode (3 channels) - for autoencoders trained with RGB
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])

            self.eval_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])
        else:
            # Grayscale mode (1 channel)
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])

            self.eval_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])

    def get_dataloaders(self):
        data_train = self.DataClass(split='train', transform=self.train_transform, download=True, size=self.image_size)
        data_test  = self.DataClass(split='test',  transform=self.eval_transform, download=True, size=self.image_size)
        data_val   = self.DataClass(split='val',   transform=self.eval_transform, download=True, size=self.image_size)

        targets = np.array(data_train.labels).flatten() 
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[targets]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(dataset=data_train, batch_size=self.batch_size, sampler=sampler)
        test_loader  = DataLoader(dataset=data_test,  batch_size=self.batch_size, shuffle=False)
        val_loader   = DataLoader(dataset=data_val,   batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, val_loader