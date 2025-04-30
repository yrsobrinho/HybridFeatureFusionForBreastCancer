from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from medmnist import BreastMNIST, INFO

from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import INFO
from medmnist.dataset import BreastMNIST
import numpy as np

class BreastMNISTLoader:
    def __init__(self, batch_size=128, image_size=28):
        self.batch_size = batch_size
        self.image_size = image_size
        self.info = INFO['breastmnist']
        self.DataClass = BreastMNIST

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        self.eval_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

    def get_dataloaders(self):
        data_train = self.DataClass(split='train', transform=self.train_transform, download=True, size=self.image_size)
        data_test  = self.DataClass(split='test',  transform=self.eval_transform, download=True, size=self.image_size)
        data_val   = self.DataClass(split='val',   transform=self.eval_transform, download=True, size=self.image_size)

        # Calculate class weights for WeightedRandomSampler
        targets = np.array(data_train.labels).flatten() 
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[targets]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(dataset=data_train, batch_size=self.batch_size, sampler=sampler)
        test_loader  = DataLoader(dataset=data_test,  batch_size=self.batch_size, shuffle=False)
        val_loader   = DataLoader(dataset=data_val,   batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, val_loader