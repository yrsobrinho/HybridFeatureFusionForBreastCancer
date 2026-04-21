from pathlib import Path

from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms
from medmnist import BreastMNIST, INFO
import os
from PIL import Image
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INBREAST_ROOT = PROJECT_ROOT / "src" / "data" / "INbreast" / "png"
BUSUCLM_ROOT = PROJECT_ROOT / "src" / "data" / "BUS-UCLM" / "png"

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
                print(f"Warning: Folder {folder_path} does not exist")
                continue
            
            all_files = os.listdir(folder_path)
            # Filter only image files
            images = sorted([f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            
            if len(images) == 0:
                print(f"Warning: No images found in {folder_path}")
                continue
                
            self.image_paths.extend([os.path.join(folder_path, img_name) for img_name in images])
            self.labels.extend([int(label)] * len(images))
            print(f"Loaded {len(images)} images from {folder_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  
        return image, label


class BCDRLoader:
    def __init__(self, batch_size, image_size, root_dir='src/data/png'):
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
            print(f"Warning: BCDR train dataset is empty at {train_dir}")

        train_loader = DataLoader(
            dataset=data_train,
            batch_size=self.batch_size,
            sampler=sampler if sampler else None,
            shuffle=(sampler is None and len(data_train) > 0)
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

class BUSBRADataset(Dataset):
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
            transforms.RandomRotation(degrees=15),
        ]

        if train and data_augmentation:
            self.transform = transforms.Compose(augmentation_transform + base_transform)
        else:
            self.transform = transforms.Compose(base_transform)

        for label in ['0', '1']:
            folder_path = os.path.join(root_dir, label)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} does not exist")
                continue
            
            all_files = os.listdir(folder_path)
            # Filter only image files
            images = sorted([f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            
            if len(images) == 0:
                print(f"Warning: No images found in {folder_path}")
                continue
                
            self.image_paths.extend([os.path.join(folder_path, img_name) for img_name in images])
            self.labels.extend([int(label)] * len(images))
            print(f"Loaded {len(images)} images from {folder_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  
        return image, label


class BUSBRALoader:
    def __init__(self, batch_size, image_size, root_dir='src/data/BUSBRA'):
        self.batch_size = batch_size
        self.image_size = image_size
        self.root_dir = root_dir
        
    def get_dataloaders(self):
        train_dir = os.path.join(self.root_dir, 'train')
        val_dir = os.path.join(self.root_dir, 'val')
        test_dir = os.path.join(self.root_dir, 'test')
        
        data_train = BUSBRADataset(
            root_dir=train_dir,
            train=True,
            data_augmentation=True,
            image_size=self.image_size
        )
        
        data_val = BUSBRADataset(
            root_dir=val_dir,
            train=False,
            data_augmentation=False,
            image_size=self.image_size
        )
        
        data_test = BUSBRADataset(
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
            shuffle=(sampler is None and len(data_train) > 0)
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


class INbreastDataset(Dataset):
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
            transforms.RandomRotation(degrees=15),
        ]

        if train and data_augmentation:
            self.transform = transforms.Compose(augmentation_transform + base_transform)
        else:
            self.transform = transforms.Compose(base_transform)

        for label in ['0', '1']:
            folder_path = os.path.join(root_dir, label)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} does not exist")
                continue
            
            all_files = os.listdir(folder_path)
            # Filter only image files
            images = sorted([f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            
            if len(images) == 0:
                print(f"Warning: No images found in {folder_path}")
                continue
                
            self.image_paths.extend([os.path.join(folder_path, img_name) for img_name in images])
            self.labels.extend([int(label)] * len(images))
            print(f"Loaded {len(images)} images from {folder_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  
        return image, label


class INbreastLoader:
    def __init__(self, batch_size, image_size, root_dir=str(INBREAST_ROOT)):
        self.batch_size = batch_size
        self.image_size = image_size
        self.root_dir = root_dir
        
    def get_dataloaders(self):
        train_dir = os.path.join(self.root_dir, 'train')
        test_dir = os.path.join(self.root_dir, 'test')
        val_dir = os.path.join(self.root_dir, 'val')
        
        data_train = INbreastDataset(
            root_dir=train_dir,
            train=True,
            data_augmentation=True,
            image_size=self.image_size
        )

        data_val= INbreastDataset(
            root_dir=val_dir,
            train=False,
            data_augmentation=False,
            image_size=self.image_size
        )
    
        data_test = INbreastDataset(
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
            print(f"Warning: INbreast train dataset is empty at {train_dir}")

        train_loader = DataLoader(
            dataset=data_train,
            batch_size=self.batch_size,
            sampler=sampler if sampler else None,
            shuffle=(sampler is None and len(data_train) > 0)
        )

        test_loader = DataLoader(
            dataset=data_test,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        val_loader = DataLoader(
            dataset=data_val,
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


class BUSUCLMDataset(Dataset):
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
            transforms.RandomRotation(degrees=15),
        ]

        if train and data_augmentation:
            self.transform = transforms.Compose(augmentation_transform + base_transform)
        else:
            self.transform = transforms.Compose(base_transform)

        for label in ['0', '1']:
            folder_path = os.path.join(root_dir, label)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} does not exist")
                continue
            
            all_files = os.listdir(folder_path)
            # Filter only image files
            images = sorted([f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            
            if len(images) == 0:
                print(f"Warning: No images found in {folder_path}")
                continue
                
            self.image_paths.extend([os.path.join(folder_path, img_name) for img_name in images])
            self.labels.extend([int(label)] * len(images))
            print(f"Loaded {len(images)} images from {folder_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  
        return image, label


class BUSUCLMLoader:
    def __init__(self, batch_size, image_size, root_dir=str(BUSUCLM_ROOT)):
        self.batch_size = batch_size
        self.image_size = image_size
        self.root_dir = root_dir
        
    def get_dataloaders(self):
        train_dir = os.path.join(self.root_dir, 'train')
        val_dir = os.path.join(self.root_dir, 'val')
        test_dir = os.path.join(self.root_dir, 'test')
        
        data_train = BUSUCLMDataset(
            root_dir=train_dir,
            train=True,
            data_augmentation=True,
            image_size=self.image_size
        )
        
        data_val = BUSUCLMDataset(
            root_dir=val_dir,
            train=False,
            data_augmentation=False,
            image_size=self.image_size
        )
        
        data_test = BUSUCLMDataset(
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
            print(f"Warning: BUS-UCLM train dataset is empty at {train_dir}")

        train_loader = DataLoader(
            dataset=data_train,
            batch_size=self.batch_size,
            sampler=sampler if sampler else None,
            shuffle=(sampler is None and len(data_train) > 0)
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


class BreastLesionsUSGDataset(Dataset):
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
            transforms.RandomRotation(degrees=15),
        ]

        if train and data_augmentation:
            self.transform = transforms.Compose(augmentation_transform + base_transform)
        else:
            self.transform = transforms.Compose(base_transform)

        for label in ['0', '1']:
            folder_path = os.path.join(root_dir, label)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} does not exist")
                continue
            
            all_files = os.listdir(folder_path)
            # Filter only image files
            images = sorted([f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            
            if len(images) == 0:
                print(f"Warning: No images found in {folder_path}")
                continue
                
            self.image_paths.extend([os.path.join(folder_path, img_name) for img_name in images])
            self.labels.extend([int(label)] * len(images))
            print(f"Loaded {len(images)} images from {folder_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  
        return image, label


class BreastLesionsUSGLoader:
    def __init__(self, batch_size, image_size, root_dir='src/data/BreastLesionsUSG'):
        self.batch_size = batch_size
        self.image_size = image_size
        self.root_dir = root_dir
        
    def get_dataloaders(self):
        train_dir = os.path.join(self.root_dir, 'train')
        val_dir = os.path.join(self.root_dir, 'val')
        test_dir = os.path.join(self.root_dir, 'test')
        
        data_train = BreastLesionsUSGDataset(
            root_dir=train_dir,
            train=True,
            data_augmentation=True,
            image_size=self.image_size
        )
        
        data_val = BreastLesionsUSGDataset(
            root_dir=val_dir,
            train=False,
            data_augmentation=False,
            image_size=self.image_size
        )
        
        data_test = BreastLesionsUSGDataset(
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
            print(f"Warning: BreastLesionsUSG train dataset is empty at {train_dir}")

        train_loader = DataLoader(
            dataset=data_train,
            batch_size=self.batch_size,
            sampler=sampler if sampler else None,
            shuffle=(sampler is None and len(data_train) > 0)
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


class BreastMNISTSingleImageLoader(BreastMNISTLoader):
    def __init__(self, image_size):
        super().__init__(batch_size=1, image_size=image_size)

    def get_single_image_loader(self):
        data_train = self.DataClass(split='train', transform=self.eval_transform, download=True, size=self.image_size)
        single_image_dataset = torch.utils.data.Subset(data_train, [0])  # Select the first image
        single_image_loader = DataLoader(single_image_dataset, batch_size=1, shuffle=False)
        return single_image_loader