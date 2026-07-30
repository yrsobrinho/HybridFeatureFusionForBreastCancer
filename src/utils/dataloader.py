from pathlib import Path

from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms
from medmnist import BreastMNIST, INFO
import os
from PIL import Image
import torch
import numpy as np

from src.utils.case_split import BUSUCLM_TEST_CASES, busuclm_case_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INBREAST_ROOT = PROJECT_ROOT / "src" / "data" / "INbreast" / "png"
BUSUCLM_DATA_ROOT = PROJECT_ROOT / "src" / "data" / "BUS-UCLM"
BUSUCLM_ROOT = BUSUCLM_DATA_ROOT / "png"

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
    def __init__(self, root_dir=None, samples=None, train=True, data_augmentation=True, image_size=224):
        """Either scan root_dir/{0,1}, or take an explicit [(path, label)] list.

        The samples form is what the case-wise split feeds in, since that split
        is computed from INFO.csv rather than from a directory layout.
        """
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

        if samples is not None:
            for image_path, label in sorted(samples):
                self.image_paths.append(image_path)
                self.labels.append(int(label))
            return

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
    """Loads BUS-UCLM with a case-wise (patient-wise) split computed at runtime.

    root_dir may point either at the dataset root (images/ + INFO.csv), in which
    case the split is computed here from the case codes, or at a pre-split
    png/{train,val,test}/{0,1} tree, which is read as-is. The second form keeps
    callers that still pass the png/ path working unchanged.
    """

    def __init__(self, batch_size, image_size, root_dir=str(BUSUCLM_DATA_ROOT),
                 test_cases=BUSUCLM_TEST_CASES, val_frac=0.15, seed=42):
        self.batch_size = batch_size
        self.image_size = image_size
        self.root_dir = root_dir
        self.test_cases = test_cases
        self.val_frac = val_frac
        self.seed = seed

    def _build_datasets(self):
        # A pre-split tree wins, so the legacy png/ path keeps its old meaning.
        if os.path.isdir(os.path.join(self.root_dir, 'train')):
            sources = {name: {'root_dir': os.path.join(self.root_dir, name)}
                       for name in ['train', 'val', 'test']}
        else:
            split = busuclm_case_split(self.root_dir, self.test_cases, self.val_frac, self.seed)
            sources = {name: {'samples': split[name]} for name in ['train', 'val', 'test']}

        return {
            name: BUSUCLMDataset(
                train=(name == 'train'),
                data_augmentation=(name == 'train'),
                image_size=self.image_size,
                **source
            )
            for name, source in sources.items()
        }

    def get_dataloaders(self):
        datasets = self._build_datasets()
        data_train, data_val, data_test = datasets['train'], datasets['val'], datasets['test']

        if len(data_train) > 0:
            targets = np.array(data_train.labels)
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
            print(f"Warning: BUS-UCLM train dataset is empty for root {self.root_dir}")

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