import os
import datetime
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt


def train_epoch(model, dataloader, criterion, optimizer, device, extract_features=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.squeeze().to(device)
        
        optimizer.zero_grad()
        
        if extract_features:
            feats = model.extract_features(images)
            if hasattr(model, 'fc3'):
                outputs = model.fc3(feats)
            else:
                outputs = model(images)
        else:
            outputs = model(images)
            
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            "loss": running_loss / (pbar.n + 1),
            "acc": correct / total
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate_model(model, dataloader, criterion, device, extract_features=False):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.squeeze().to(device)
            
            if extract_features:
                feats = model.extract_features(images)
                if hasattr(model, 'fc3'):
                    outputs = model.fc3(feats)
                else:
                    outputs = model(images)
            else:
                outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() if outputs.shape[1] > 1 else torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, precision, recall, f1, auc


def test_model(model, dataloader, criterion, device, hyperparams, 
               output_file="test_results.txt", extract_features=False, 
               compute_detailed_metrics=True):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.squeeze().to(device)
            
            if extract_features:
                feats = model.extract_features(images)
                if hasattr(model, 'fc3'):
                    outputs = model.fc3(feats)
                else:
                    outputs = model(images)
            else:
                outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() if outputs.shape[1] > 1 else torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    tp = sum(1 for true, pred in zip(all_labels, all_predictions) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(all_labels, all_predictions) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(all_labels, all_predictions) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(all_labels, all_predictions) if true == 1 and pred == 0)
    
    if compute_detailed_metrics:
        precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
        
        precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0
        
        precision_macro = (precision_0 + precision_1) / 2
        recall_macro = (recall_0 + recall_1) / 2
        f1_macro = (f1_0 + f1_1) / 2
        
        total_tp = tp + tn
        total_fp_fn = fp + fn
        precision_micro = total_tp / (total_tp + total_fp_fn) if (total_tp + total_fp_fn) > 0 else 0.0
        recall_micro = total_tp / (total_tp + total_fp_fn) if (total_tp + total_fp_fn) > 0 else 0.0
        f1_micro = accuracy
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    
    avg_loss = total_loss / len(dataloader)
    
    with open(output_file, "a") as f:
        f.write(f"\n[Model Testing - {datetime.datetime.now()}]\n")
        f.write(f"Hyperparameters: {hyperparams}\n")
        f.write(f"Test Loss: {avg_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        
        if compute_detailed_metrics:
            f.write(f"Macro Precision: {precision_macro:.4f}\n")
            f.write(f"Macro Recall: {recall_macro:.4f}\n")
            f.write(f"Macro F1: {f1_macro:.4f}\n")
            f.write(f"Micro Precision: {precision_micro:.4f}\n")
            f.write(f"Micro Recall: {recall_micro:.4f}\n")
            f.write(f"Micro F1: {f1_micro:.4f}\n")
        
        f.write("-" * 50 + "\n")
    
    # Print results
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    if compute_detailed_metrics:
        print(f"Macro - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
        print(f"Micro - Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1: {f1_micro:.4f}")
    
    return avg_loss, accuracy, precision, recall, f1, auc


def extract_and_save_features(model, dataloader, device, output_dir, split_name, extract_features=False):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    feats_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extract {split_name}"):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                imgs, labels = batch[0], batch[1]
            else:
                imgs = batch
                labels = None
            
            imgs = imgs.to(device)
            
            if extract_features and hasattr(model, 'extract_features'):
                feats = model.extract_features(imgs)
            else:
                feats = model(imgs)
            
            feats_list.append(feats.cpu().numpy())
            
            if labels is not None:
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                labels_list.append(labels)
    
    # Concatenate and save
    feats = np.concatenate(feats_list, axis=0)
    np.save(os.path.join(output_dir, f"{split_name}_features.npy"), feats)
    
    if len(labels_list) > 0:
        labs = np.concatenate(labels_list, axis=0)
        np.save(os.path.join(output_dir, f"{split_name}_labels.npy"), labs)
    
    print(f"Saved {split_name} features -> {output_dir}")
    print(f"  Features shape: {feats.shape}")
    if len(labels_list) > 0:
        print(f"  Labels shape: {labs.shape}")


def plot_roc_curve(labels, probs, save_path, title="ROC Curve"):
    try:
        auc = roc_auc_score(labels, probs)
        fpr, tpr, thresholds = roc_curve(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"ROC curve saved to {save_path}")
    except Exception as e:
        print(f"Warning: Could not plot ROC curve: {e}")


def save_checkpoint(model, optimizer, epoch, val_acc, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, 0.0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint.get('epoch', 0)
                val_acc = checkpoint.get('val_acc', 0.0)
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                epoch = 0
                val_acc = 0.0
            else:
                model.load_state_dict(checkpoint)
                epoch = 0
                val_acc = 0.0
        else:
            model.load_state_dict(checkpoint)
            epoch = 0
            val_acc = 0.0
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return epoch, val_acc
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, 0.0


def training_loop(model, train_loader, val_loader, criterion, optimizer, device,
                  num_epochs, save_dir, model_name="model", extract_features=False,
                  save_best_only=True, save_all_epochs=False, save_frequency=1):
    """
    Training loop with flexible checkpoint saving.
    
    Args:
        save_best_only: If True, only saves the best model based on validation accuracy
        save_all_epochs: If True, saves checkpoint at every epoch (overrides save_best_only)
        save_frequency: Save checkpoint every N epochs (only used if save_all_epochs=True)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create checkpoints subdirectory if saving all epochs
    if save_all_epochs:
        checkpoints_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
    
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': []
    }
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, extract_features
        )
   
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate_model(
            model, val_loader, criterion, device, extract_features
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
              f"Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, "
              f"F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, f"best_{model_name}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model saved (val_acc: {best_val_acc:.4f}) at {best_model_path}")
        
        # Save all epochs if requested
        if save_all_epochs and (epoch + 1) % save_frequency == 0:
            epoch_checkpoint_path = os.path.join(checkpoints_dir, f"{model_name}_epoch{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, val_acc, epoch_checkpoint_path)
            print(f"  Checkpoint saved at epoch {epoch+1}")
        
        # Legacy behavior: save_best_only=False saves all epochs
        if not save_best_only and not save_all_epochs:
            model_path = os.path.join(save_dir, f"{model_name}_epoch{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, val_acc, model_path)
    
    # Save final model
    if save_all_epochs or not save_best_only:
        final_model_path = os.path.join(save_dir, f"final_{model_name}.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"\n✓ Final model saved at {final_model_path}")
    
    return history
