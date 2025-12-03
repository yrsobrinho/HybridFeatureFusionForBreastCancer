from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms
from medmnist import BreastMNIST, INFO
import os
from PIL import Image
import torch
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
    
class BCDRDataset(Dataset):
    def __init__(self, root_dir, train=True, data_augmentation=True, image_size=224, use_rgb=True):
        """
        BCDR Dataset Loader
        
        Args:
            root_dir: Path to dataset root (e.g., 'data/BCDR/train' or 'data/BCDR/test')
            train: Whether this is training set (affects augmentation)
            data_augmentation: Whether to apply data augmentation
            image_size: Target image size (default 224 to match BreastMNIST)
            use_rgb: If True, use RGB (3 channels). If False, convert to grayscale
        """
        self.image_paths = []
        self.labels = []
        self.use_rgb = use_rgb
        
        # Adjust normalization based on RGB vs grayscale
        if use_rgb:
            normalize = transforms.Normalize(mean=[0.5186, 0.5187, 0.5185], std=[0.1957, 0.1957, 0.1957])
        else:
            normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        
        base_transform = [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1) if not use_rgb else transforms.Lambda(lambda x: x),
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
    """
    BCDR Dataset Loader with train/val/test splits
    Follows the same interface as BreastMNISTLoader for easy swapping
    """
    def __init__(self, batch_size, image_size, use_rgb=True, root_dir='data/BCDR'):
        """
        Args:
            batch_size: Batch size for dataloaders
            image_size: Target image size
            use_rgb: If True, uses 3-channel RGB. If False, uses 1-channel grayscale
            root_dir: Root directory containing train/val/test folders
        """
        self.batch_size = batch_size
        self.image_size = image_size
        self.use_rgb = use_rgb
        self.root_dir = root_dir
        
    def get_dataloaders(self):
        """
        Returns train_loader, test_loader, val_loader
        
        Expected directory structure:
        root_dir/
          train/
            0/  (benign images)
            1/  (malignant images)
          val/
            0/
            1/
          test/
            0/
            1/
        """
        train_dir = os.path.join(self.root_dir, 'train')
        val_dir = os.path.join(self.root_dir, 'val')
        test_dir = os.path.join(self.root_dir, 'test')
        
        # Create datasets
        data_train = BCDRDataset(
            root_dir=train_dir,
            train=True,
            data_augmentation=True,
            image_size=self.image_size,
            use_rgb=self.use_rgb
        )
        
        data_val = BCDRDataset(
            root_dir=val_dir,
            train=False,
            data_augmentation=False,
            image_size=self.image_size,
            use_rgb=self.use_rgb
        )
        
        data_test = BCDRDataset(
            root_dir=test_dir,
            train=False,
            data_augmentation=False,
            image_size=self.image_size,
            use_rgb=self.use_rgb
        )
        
        # Calculate class weights for balanced sampling (train only)
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
        
        # Create dataloaders
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