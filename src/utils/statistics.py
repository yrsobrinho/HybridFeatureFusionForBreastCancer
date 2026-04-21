"""
Dataset Statistics Calculator
Calculates total samples, positive and negative samples for all datasets
"""

import os
from pathlib import Path

import numpy as np
from src.utils.dataloader import BreastMNISTLoader, BCDRLoader, BUSBRALoader, BUSUCLMLoader, INbreastLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INBREAST_ROOT = PROJECT_ROOT / 'src' / 'data' / 'INbreast' / 'png'
BUSUCLM_ROOT = PROJECT_ROOT / 'src' / 'data' / 'BUS-UCLM' / 'png'


def calculate_dataset_stats(loader_name, train_loader, test_loader, val_loader=None):
    """Calculate and print statistics for a dataset"""
    
    print(f"\n{'='*60}")
    print(f"{loader_name} Dataset Statistics")
    print(f"{'='*60}")
    
    # Training set
    train_labels = [label.item() if hasattr(label, 'item') else label 
                    for _, label in train_loader.dataset]
    train_total = len(train_labels)
    train_positive = sum(1 for label in train_labels if label == 1)
    train_negative = sum(1 for label in train_labels if label == 0)
    
    print(f"\nTRAIN SET:")
    print(f"  Total samples:     {train_total}")
    print(f"  Positive (class 1): {train_positive} ({train_positive/train_total*100:.2f}%)")
    print(f"  Negative (class 0): {train_negative} ({train_negative/train_total*100:.2f}%)")
    
    # Validation set (if exists)
    if val_loader is not None:
        val_labels = [label.item() if hasattr(label, 'item') else label 
                      for _, label in val_loader.dataset]
        val_total = len(val_labels)
        val_positive = sum(1 for label in val_labels if label == 1)
        val_negative = sum(1 for label in val_labels if label == 0)
        
        print(f"\nVALIDATION SET:")
        print(f"  Total samples:     {val_total}")
        print(f"  Positive (class 1): {val_positive} ({val_positive/val_total*100:.2f}%)")
        print(f"  Negative (class 0): {val_negative} ({val_negative/val_total*100:.2f}%)")
    
    # Test set
    test_labels = [label.item() if hasattr(label, 'item') else label 
                   for _, label in test_loader.dataset]
    test_total = len(test_labels)
    test_positive = sum(1 for label in test_labels if label == 1)
    test_negative = sum(1 for label in test_labels if label == 0)
    
    print(f"\nTEST SET:")
    print(f"  Total samples:     {test_total}")
    print(f"  Positive (class 1): {test_positive} ({test_positive/test_total*100:.2f}%)")
    print(f"  Negative (class 0): {test_negative} ({test_negative/test_total*100:.2f}%)")
    
    # Overall statistics
    if val_loader is not None:
        overall_total = train_total + val_total + test_total
        overall_positive = train_positive + val_positive + test_positive
        overall_negative = train_negative + val_negative + test_negative
    else:
        overall_total = train_total + test_total
        overall_positive = train_positive + test_positive
        overall_negative = train_negative + test_negative
    
    print(f"\nOVERALL:")
    print(f"  Total samples:     {overall_total}")
    print(f"  Positive (class 1): {overall_positive} ({overall_positive/overall_total*100:.2f}%)")
    print(f"  Negative (class 0): {overall_negative} ({overall_negative/overall_total*100:.2f}%)")
    print(f"{'='*60}\n")
    
    return {
        'train': {'total': train_total, 'positive': train_positive, 'negative': train_negative},
        'val': {'total': val_total, 'positive': val_positive, 'negative': val_negative} if val_loader else None,
        'test': {'total': test_total, 'positive': test_positive, 'negative': test_negative},
        'overall': {'total': overall_total, 'positive': overall_positive, 'negative': overall_negative}
    }


def main():
    """Main function to calculate statistics for all datasets"""
    
    all_stats = {}
    
    # BreastMNIST Dataset
    print("\n" + "="*60)
    print("Loading BreastMNIST Dataset...")
    print("="*60)
    try:
        breastmnist_loader = BreastMNISTLoader(batch_size=16, image_size=28)
        train_loader, test_loader, val_loader = breastmnist_loader.get_dataloaders()
        all_stats['BreastMNIST'] = calculate_dataset_stats(
            'BreastMNIST', train_loader, test_loader, val_loader
        )
    except Exception as e:
        print(f"Error loading BreastMNIST: {e}\n")
    
    # BCDR Dataset
    print("\n" + "="*60)
    print("Loading BCDR Dataset...")
    print("="*60)
    try:
        bcdr_loader = BCDRLoader(
            batch_size=16,
            image_size=224,
            root_dir='src/data/BCDR/png'
        )
        train_loader, test_loader, val_loader = bcdr_loader.get_dataloaders()
        all_stats['BCDR'] = calculate_dataset_stats(
            'BCDR', train_loader, test_loader, val_loader
        )
    except Exception as e:
        print(f"Error loading BCDR: {e}\n")
    
    # BUSBRA Dataset
    print("\n" + "="*60)
    print("Loading BUSBRA Dataset...")
    print("="*60)
    try:
        busbra_loader = BUSBRALoader(
            batch_size=16,
            image_size=224,
            root_dir='src/data/BUSBRA/png'
        )
        train_loader, test_loader, val_loader = busbra_loader.get_dataloaders()
        all_stats['BUSBRA'] = calculate_dataset_stats(
            'BUSBRA', train_loader, test_loader, val_loader
        )
    except Exception as e:
        print(f"Error loading BUSBRA: {e}\n")
    
    # INbreast Dataset
    print("\n" + "="*60)
    print("Loading INbreast Dataset...")
    print("="*60)
    try:
        inbreast_loader = INbreastLoader(
            batch_size=16,
            image_size=224,
            root_dir='src/data/INbreast/png'
        )
        dataloaders = inbreast_loader.get_dataloaders()
        if len(dataloaders) == 3:
            train_loader, val_loader, test_loader = dataloaders
        else:
            train_loader, test_loader = dataloaders
            val_loader = None
        
        all_stats['INbreast'] = calculate_dataset_stats(
            'INbreast', train_loader, test_loader, val_loader
        )
    except Exception as e:
        print(f"Error loading INbreast: {e}\n")

    # BUSUCLM Dataset
    print("\n" + "="*60)
    print("Loading BUSUCLM Dataset...")
    print("="*60)
    try:
        busuclm_loader = BUSUCLMLoader(
            batch_size=16,
            image_size=224,
            root_dir='src/data/BUS-UCLM/png'
        )
        train_loader, test_loader, val_loader = busuclm_loader.get_dataloaders()
        all_stats['BUSUCLM'] = calculate_dataset_stats(
            'BUSUCLM', train_loader, test_loader, val_loader
        )
    except Exception as e:
        print(f"Error loading BUSUCLM: {e}\n")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - All Datasets")
    print("="*60)
    for dataset_name, stats in all_stats.items():
        print(f"\n{dataset_name}:")
        print(f"  Train: {stats['train']['total']} samples "
              f"(Pos: {stats['train']['positive']}, Neg: {stats['train']['negative']})")
        if stats['val']:
            print(f"  Val:   {stats['val']['total']} samples "
                  f"(Pos: {stats['val']['positive']}, Neg: {stats['val']['negative']})")
        print(f"  Test:  {stats['test']['total']} samples "
              f"(Pos: {stats['test']['positive']}, Neg: {stats['test']['negative']})")
        print(f"  Total: {stats['overall']['total']} samples "
              f"(Pos: {stats['overall']['positive']}, Neg: {stats['overall']['negative']})")
    
    print("\n" + "="*60)
    print("Statistics calculation completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
