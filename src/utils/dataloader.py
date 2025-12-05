from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms
from medmnist import BreastMNIST, INFO
import os
from PIL import Image
import torch
import numpy as np

class BreastMNISTLoader:
    def __init__(self, batch_size, image_size):
        self.batch_size = batch_size
        self.image_size = image_size
        self.info = INFO['breastmnist']
        self.DataClass = BreastMNIST


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
    
class BCDRDataset(Dataset):
    def __init__(self, root_dir, train=True, data_augmentation=True, image_size=224):
        self.image_paths = []
        self.labels = []
        
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        
        base_transform = [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize
        ]

        augmentation_transform = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15)
        ]

        if train and data_augmentation:
            self.transform = transforms.Compose(augmentation_transform + base_transform)
        else:
            self.transform = transforms.Compose(base_transform)

        for label in ['0', '1']:
            folder_path = os.path.join(root_dir, label)
            if not os.path.exists(folder_path):
                continue
            images = sorted(os.listdir(folder_path)) 
            self.image_paths.extend([os.path.join(folder_path, img_name) for img_name in images])
            self.labels.extend([int(label)] * len(images))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  
        return image, label


class BCDRLoader:
    def __init__(self, batch_size, image_size, root_dir='data/BCDR'):
        self.batch_size = batch_size
        self.image_size = image_size
        self.root_dir = root_dir
        
    def get_dataloaders(self):
        train_dir = os.path.join(self.root_dir, 'train')
        val_dir = os.path.join(self.root_dir, 'val')
        test_dir = os.path.join(self.root_dir, 'test')
        
        data_train = BCDRDataset(
            root_dir=train_dir,
            train=True,
            data_augmentation=True,
            image_size=self.image_size
        )
        
        data_val = BCDRDataset(
            root_dir=val_dir,
            train=False,
            data_augmentation=False,
            image_size=self.image_size
        )
        
        data_test = BCDRDataset(
            root_dir=test_dir,
            train=False,
            data_augmentation=False,
            image_size=self.image_size
        )

        if len(data_train) > 0:
            targets = np.array([label for _, label in data_train])
            class_counts = np.bincount(targets)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[targets]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
        else:
            sampler = None

        train_loader = DataLoader(
            dataset=data_train,
            batch_size=self.batch_size,
            sampler=sampler if sampler else None,
            shuffle=(sampler is None)
        )
        
        val_loader = DataLoader(
            dataset=data_val,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        test_loader = DataLoader(
            dataset=data_test,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_loader, test_loader, val_loader
    
    def get_train_loader(self):
        train_loader, _, _ = self.get_dataloaders()
        return train_loader
    
    def get_val_loader(self):
        _, _, val_loader = self.get_dataloaders()
        return val_loader
    
    def get_test_loader(self):
        _, test_loader, _ = self.get_dataloaders()
        return test_loader