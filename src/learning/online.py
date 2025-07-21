import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.dataloader import BreastMNISTLoader
from src.models.quantum import QuantumModel
from src.models.classical import ClassicalModel
from src.utils.extraction import QuantumFeatureExtractor, ClassicalFeatureExtractor
from src.utils.methods import DimensionAdaptiveAttention, DimensionAdaptiveSelfAttention, DimensionAdaptiveAlignmentLoss
from src.config.hyperparams import HyperparameterConfig
from src.utils.schedulers import CosineAnnealingAlpha
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# Base path
BASE_PATH = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection"

# Directories
output_dir = f"{BASE_PATH}/experiments/online"
os.makedirs(output_dir, exist_ok=True)

# Load hyperparameters
config = HyperparameterConfig()
common_hyperparams, quantum_hyperparams, classical_hyperparams, fusion_hyperparams = config.get_all_params()

# Load data
loader = BreastMNISTLoader(batch_size=common_hyperparams["batch_size"], image_size=28)
train_loader, test_loader, val_loader = loader.get_dataloaders()

# Initialize complete models for Phase 1 (Knowledge Distillation)
teacher_model = ClassicalModel(num_classes=common_hyperparams["num_classes"]).to(device)
student_model = QuantumModel(n_qubits=quantum_hyperparams["n_qubits"], 
                           n_layers=quantum_hyperparams["n_layers"], 
                           num_classes=common_hyperparams["num_classes"]).to(device)

# Loss functions
criterion = nn.CrossEntropyLoss()
distillation_loss = nn.KLDivLoss(reduction="batchmean")

# Optimizers for Phase 1
teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=classical_hyperparams["learning_rate"])
student_optimizer = optim.Adam(student_model.parameters(), lr=quantum_hyperparams["learning_rate"])

def evaluate_models(teacher, student, data_loader, phase_name, is_warmup=False):
    """Evaluate both teacher and student models"""
    teacher.eval()
    student.eval()
    
    teacher_correct = 0
    student_correct = 0
    total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Convert single channel to RGB for teacher if needed
            teacher_images = images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images
            
            # Teacher predictions
            teacher_outputs = teacher(teacher_images)
            teacher_preds = teacher_outputs.argmax(dim=1)
            teacher_correct += (teacher_preds == labels).sum().item()
            
            # Student predictions
            if not is_warmup:
                student_outputs = student(images)
                student_preds = student_outputs.argmax(dim=1)
                student_correct += (student_preds == labels).sum().item()
            
            total += labels.size(0)
            if not is_warmup:
                val_loss += criterion(student_outputs, labels).item()

    teacher_acc = teacher_correct / total
    student_acc = student_correct / total
    avg_loss = val_loss / len(data_loader)
    
    print(f"{phase_name} - Teacher Acc: {teacher_acc:.4f}, Student Acc: {student_acc:.4f}, Loss: {avg_loss:.4f}")
    return teacher_acc, student_acc, avg_loss

best_teacher_accuracy = 0.0
best_teacher_epoch = 0

best_student_accuracy = 0.0
best_epoch = 0

print("=" * 60)
print("PHASE 1: KNOWLEDGE DISTILLATION (Teacher → Student)")
print("=" * 60)

# Warmup configuration
warmup_epochs = 5 
total_epochs = common_hyperparams["epochs_phase1"]

teacher_model_path = os.path.join(output_dir, "best_teacher_model.pth")
if os.path.exists(teacher_model_path):
    print("Loading best teacher model...")
    teacher_checkpoint = torch.load(teacher_model_path, map_location=device, weights_only=True)
    teacher_model.load_state_dict(teacher_checkpoint['teacher_model'])
    teacher_model.eval()
    
    # Freeze teacher model
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    best_teacher_accuracy = teacher_checkpoint['teacher_accuracy']
    best_teacher_epoch = teacher_checkpoint['epoch']
    
    print(f"Teacher loaded - Accuracy: {best_teacher_accuracy:.4f}")
    
    # Skip warmup epochs
    warmup_epochs = 0
    print("WARMUP SKIPPED - Starting directly with joint training")
    
else:
    print(f"WARM UP IS STARTING - Teacher model not found")

alpha_scheduler = CosineAnnealingAlpha(
    alpha_max=common_hyperparams["alpha_max"],
    alpha_min=common_hyperparams["alpha_min"],
    T_max=total_epochs,
    warmup_epochs=warmup_epochs
)

# print("\n" + "="*50)
# print("COSINE ANNEALING ALPHA SCHEDULE")
# print("="*50)
# alpha_scheduler.plot_schedule(total_epochs)
# print("="*50)

print(f"Warmup Plan: {warmup_epochs} epochs teacher-only, then {total_epochs - warmup_epochs} epochs joint training")

for epoch in range(total_epochs):
    print(f"\nPhase 1 - Epoch {epoch + 1}/{total_epochs}")

    # current_alpha = alpha_scheduler.get_alpha(epoch)
    current_alpha = common_hyperparams["alpha"]  

    # Determine training mode
    is_warmup = epoch < warmup_epochs

    if is_warmup:
        print("WARMUP MODE: Training Teacher Only")
    elif epoch == warmup_epochs:
        print("Teacher model frozen after warmup training.")
    else:
        print("JOINT MODE: Training Teacher + Student with KD")

        # Normal freezing process
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.eval() 
        
    
    if not is_warmup:
        teacher_model.eval()
    else:
        teacher_model.train()
    
    student_model.train()

    total_teacher_loss = 0.0
    total_student_loss = 0.0
    total_kd_loss = 0.0
    total_combined_loss = 0.0
    student_ce_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Convert single channel to RGB for teacher if needed
        teacher_images = images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images
        
        if is_warmup:
            teacher_optimizer.zero_grad()
            teacher_outputs = teacher_model(teacher_images)
            teacher_ce_loss = criterion(teacher_outputs, labels)
            
            teacher_ce_loss.backward()
            teacher_optimizer.step()
            
            # Accumulate losses
            total_teacher_loss += teacher_ce_loss.item()
            
        else:
            # JOINT TRAINING: Student with KD from frozen teacher
            student_optimizer.zero_grad()
            
        
            with torch.no_grad():
                teacher_outputs = teacher_model(teacher_images)
                teacher_ce_loss = criterion(teacher_outputs, labels)
            
            # Student forward pass
            student_outputs = student_model(images)
            student_ce_loss = criterion(student_outputs, labels)
            
            # Knowledge Distillation loss
            teacher_probs = torch.softmax(teacher_outputs / common_hyperparams["temperature"], dim=1)
            student_log_probs = torch.log_softmax(student_outputs / common_hyperparams["temperature"], dim=1)
            kd_loss = distillation_loss(student_log_probs, teacher_probs) * (common_hyperparams["temperature"] ** 2)
            
            combined_loss = current_alpha * kd_loss + (1 - current_alpha) * student_ce_loss
            
            # Backward pass
            combined_loss.backward()
            student_optimizer.step()
            
            # Accumulate losses
            total_teacher_loss += teacher_ce_loss.item()
            total_student_loss += student_ce_loss.item()
            total_kd_loss += kd_loss.item()
            total_combined_loss += combined_loss.item()
        
        if batch_idx % 5 == 0:
            
            if is_warmup:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} [WARMUP] - "
                      f"Teacher CE: {teacher_ce_loss.item():.4f}, Student CE (eval): {student_ce_loss:.4f}")
            else:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} [JOINT] - "
                      f"Combined: {combined_loss.item():.4f}, Teacher CE: {teacher_ce_loss.item():.4f}, "
                      f"Student CE: {student_ce_loss.item():.4f}, KD: {kd_loss.item():.4f}")
    
    # Epoch summary
    num_batches = len(train_loader)
    if is_warmup:
        print(f"Phase 1 - Epoch {epoch + 1} [WARMUP] Summary:")
        print(f"  Teacher Loss: {total_teacher_loss / num_batches:.4f}")
        print(f"  Student Loss (eval only): {total_student_loss / num_batches:.4f}")
    else:
        print(f"Phase 1 - Epoch {epoch + 1} [JOINT] Summary:")
        print(f"  Combined Loss: {total_combined_loss / num_batches:.4f}")
        print(f"  Teacher CE Loss: {total_teacher_loss / num_batches:.4f}")
        if not is_warmup:
            print(f"  Student CE Loss (eval): {total_student_loss / num_batches:.4f}")
        print(f"  KD Loss: {total_kd_loss / num_batches:.4f}")
        print(f"  Alpha: {current_alpha:.3f}")
    
    # Validation
    teacher_acc, student_acc, avg_loss = evaluate_models(teacher_model, student_model, val_loader, "Validation", is_warmup=is_warmup)
    
    if is_warmup and teacher_acc > best_teacher_accuracy:
        best_teacher_accuracy = teacher_acc
        best_teacher_epoch = epoch + 1
        print(f"NEW BEST TEACHER! Accuracy: {teacher_acc:.4f} (Warmup Epoch {best_teacher_epoch})")
        
        # Save best teacher model from warmup
        torch.save({
            'teacher_model': teacher_model.state_dict(),
            'teacher_accuracy': teacher_acc,
            'student_accuracy': student_acc,  # For reference
            'epoch': best_teacher_epoch,
            'warmup_phase': True,
            'model_type': 'best_teacher_warmup',
            'hyperparams': {
                'classical_lr': classical_hyperparams["learning_rate"],
                'num_classes': common_hyperparams["num_classes"],
                'batch_size': common_hyperparams["batch_size"]
            }
        }, os.path.join(output_dir, "best_teacher_model.pth"))

    # Check if this is the best student model (only after warmup)
    if not is_warmup and student_acc > best_student_accuracy:
        best_student_accuracy = student_acc
        best_epoch = epoch + 1
        print(f"NEW BEST STUDENT MODEL! Accuracy: {student_acc:.4f} (Epoch {best_epoch})")

print(f"\n{'='*60}")
print("PHASE 1 COMPLETE - SUMMARY")
print(f"{'='*60}")

# PHASE 2: FEATURE EXTRACTION + FUSION + CLASSIFICATION 
print("\n" + "=" * 60)
print("PHASE 2: FEATURE EXTRACTION + FUSION + CLASSIFICATION")
print("=" * 60)

# Create feature extractors using the pre-trained models
quantum_extractor = QuantumFeatureExtractor(
    quantum_model=student_model,
    output_size=student_model.output_size,
    patch_size=student_model.patch_size,
    n_qubits=student_model.n_qubits
).to(device)

classical_extractor = ClassicalFeatureExtractor(teacher_model).to(device)

# Freeze the pre-trained models to only extract features
for param in quantum_extractor.quantum_model.parameters():
    param.requires_grad = False
for param in classical_extractor.classical_model.parameters():
    param.requires_grad = False

# Get output sizes
classical_extractor.eval()
quantum_extractor.eval()

with torch.no_grad():
    try:
        sample_batch = next(iter(train_loader))
        sample_images = sample_batch[0][:2].to(device)
        
        if sample_images.shape[1] == 1:
            sample_images = sample_images.repeat(1, 3, 1, 1)
            
        quantum_features = quantum_extractor(sample_images)
        quantum_output_size = quantum_features.shape[1]
        
        classical_features = classical_extractor(sample_images)
        classical_output_size = classical_features.shape[1]
        
    except Exception as e:
        print(f"Error with sample batch, using dummy input: {e}")
        dummy_input = torch.randn(2, 3, 28, 28).to(device)
        quantum_features = quantum_extractor(dummy_input)
        quantum_output_size = quantum_features.shape[1]
        classical_features = classical_extractor(dummy_input)
        classical_output_size = classical_features.shape[1]

print(f"Quantum feature size: {quantum_output_size}")
print(f"Classical feature size: {classical_output_size}")

# Define all fusion methods to test
fusion_methods = ['concat', 'add', 'attention', 'self_attention']
best_results = {}

def evaluate_fusion_model(quantum_ext, classical_ext, fusion_components, data_loader, method_name, split_name):
    """Evaluate a fusion model on a specific dataset split"""
    quantum_ext.eval()
    classical_ext.eval()
    
    # Set all fusion components to eval
    for component in fusion_components.values():
        if component is not None:
            component.eval()
    
    correct = 0
    total = 0
    val_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            
            # Extract features
            quantum_features = quantum_ext(images)
            classical_features = classical_ext(images)
            
            # Apply fusion based on method
            if method_name == 'concat':
                fused_features = torch.cat([quantum_features, classical_features], dim=1)
            elif method_name == 'add':
                q_proj = fusion_components['quantum_proj'](quantum_features)
                c_proj = fusion_components['classical_proj'](classical_features)
                fused_features = q_proj + c_proj
            elif method_name == 'attention':
                fused_features = fusion_components['attention_module'](quantum_features, classical_features)
            elif method_name == 'self_attention':
                concatenated = torch.cat([quantum_features, classical_features], dim=1)
                fused_features = fusion_components['self_attention_module'](concatenated)
            
            # Final classification
            outputs = fusion_components['classifier'](fused_features)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = correct / total
    avg_loss = val_loss / len(data_loader)
    
    # Additional metrics
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # ROC AUC (for binary classification)
    if len(set(all_labels)) == 2:
        try:
            roc_auc = roc_auc_score(all_labels, [prob[1] for prob in all_probs])
        except:
            roc_auc = 0.0
    else:
        roc_auc = 0.0
    
    print(f"  {split_name:5} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
          f"Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs
    }

# Train each fusion method
for fusion_method in fusion_methods:
    print(f"\n{'='*50}")
    print(f"TRAINING FUSION METHOD: {fusion_method.upper()}")
    print(f"{'='*50}")
    
    # Determine fusion output dimension
    if fusion_method == 'concat':
        fusion_output_dim = quantum_output_size + classical_output_size
    elif fusion_method == 'add':
        fusion_output_dim = min(quantum_output_size, classical_output_size)
    elif fusion_method in ['attention', 'self_attention']:
        fusion_output_dim = common_hyperparams["attention_dim"]
    
    print(f"Fusion output dimension for {fusion_method}: {fusion_output_dim}")
    
    # Initialize fusion-specific components
    fusion_components = {
        'classifier': nn.Sequential(
            nn.Linear(fusion_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, common_hyperparams["num_classes"])
        ).to(device),
        'quantum_proj': None,
        'classical_proj': None,
        'attention_module': None,
        'self_attention_module': None
    }
    
    # Create method-specific components
    if fusion_method == 'add':
        min_dim = min(quantum_output_size, classical_output_size)
        fusion_components['quantum_proj'] = nn.Linear(quantum_output_size, min_dim).to(device)
        fusion_components['classical_proj'] = nn.Linear(classical_output_size, min_dim).to(device)
    elif fusion_method == 'attention':
        fusion_components['attention_module'] = DimensionAdaptiveAttention(
            quantum_output_size, 
            classical_output_size, 
            common_hyperparams["attention_dim"]
        ).to(device)
    elif fusion_method == 'self_attention':
        fusion_components['self_attention_module'] = DimensionAdaptiveSelfAttention(
            quantum_output_size + classical_output_size,
            common_hyperparams["attention_dim"]
        ).to(device)
    
    # Feature alignment loss for this method
    feature_alignment_loss = DimensionAdaptiveAlignmentLoss(
        quantum_output_size, 
        classical_output_size
    ).to(device)
    
    # Prepare optimizer parameters
    optimizer_params = [fusion_components['classifier'].parameters()]
    
    if fusion_method == 'add':
        optimizer_params.extend([
            fusion_components['quantum_proj'].parameters(),
            fusion_components['classical_proj'].parameters()
        ])
    elif fusion_method == 'attention':
        optimizer_params.append(fusion_components['attention_module'].parameters())
    elif fusion_method == 'self_attention':
        optimizer_params.append(fusion_components['self_attention_module'].parameters())
    
    # Add feature alignment parameters
    optimizer_params.append(feature_alignment_loss.parameters())
    
    # Flatten parameter list
    all_params = []
    for param_group in optimizer_params:
        all_params.extend(list(param_group))
    
    # Create optimizer
    fusion_optimizer = optim.Adam(all_params, lr=fusion_hyperparams["learning_rate"])
    
    # Training variables
    best_method_accuracy = 0.0
    best_method_epoch = 0
    best_method_metrics = {}
    
    # Train this fusion method
    for epoch in range(common_hyperparams["epochs_phase2"]):
        print(f"\n{fusion_method.upper()} - Epoch {epoch + 1}/{common_hyperparams['epochs_phase2']}")
        
        # Set to training mode
        quantum_extractor.train()
        classical_extractor.train()
        feature_alignment_loss.train()
        
        for component in fusion_components.values():
            if component is not None:
                component.train()
        
        total_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_alignment_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            fusion_optimizer.zero_grad()
            
            # Extract features (frozen models)
            with torch.no_grad():
                quantum_features = quantum_extractor(images)
                classical_features = classical_extractor(images)
            
            # Apply fusion based on method
            if fusion_method == 'concat':
                fused_features = torch.cat([quantum_features, classical_features], dim=1)
            elif fusion_method == 'add':
                q_proj = fusion_components['quantum_proj'](quantum_features)
                c_proj = fusion_components['classical_proj'](classical_features)
                fused_features = q_proj + c_proj
            elif fusion_method == 'attention':
                fused_features = fusion_components['attention_module'](quantum_features, classical_features)
            elif fusion_method == 'self_attention':
                concatenated = torch.cat([quantum_features, classical_features], dim=1)
                fused_features = fusion_components['self_attention_module'](concatenated)
            
            # Final classification
            outputs = fusion_components['classifier'](fused_features)
            
            # Losses
            ce_loss = criterion(outputs, labels)
            alignment_loss = feature_alignment_loss(quantum_features, classical_features)
            
            # Combined loss
            total_loss_batch = (1 - common_hyperparams["beta"]) * ce_loss + common_hyperparams["beta"] * alignment_loss
            
            total_loss_batch.backward()
            fusion_optimizer.step()
            
            total_loss += total_loss_batch.item()
            epoch_ce_loss += ce_loss.item()
            epoch_alignment_loss += alignment_loss.item()
            
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Total: {total_loss_batch.item():.4f}, "
                    f"CE: {ce_loss.item():.4f}, Align: {alignment_loss.item():.4f}")
        
        # Epoch summary
        num_batches = len(train_loader)
        print(f"{fusion_method.upper()} - Epoch {epoch + 1} Summary:")
        print(f"  Total Loss: {total_loss / num_batches:.4f}")
        print(f"  CE Loss: {epoch_ce_loss / num_batches:.4f}")
        print(f"  Alignment Loss: {epoch_alignment_loss / num_batches:.4f}")
        
        # Validation
        val_metrics = evaluate_fusion_model(
            quantum_extractor, classical_extractor, fusion_components, val_loader, fusion_method, "Val"
        )
        
        # Check if this is the best for this method
        if val_metrics['accuracy'] > best_method_accuracy:
            best_method_accuracy = val_metrics['accuracy']
            best_method_epoch = epoch + 1
            print(f"NEW BEST {fusion_method.upper()} MODEL! Accuracy: {val_metrics['accuracy']:.4f} (Epoch {best_method_epoch})")
            
            # Save best model for this method
            method_state = {
                'quantum_extractor': quantum_extractor.state_dict(),
                'classical_extractor': classical_extractor.state_dict(),
                'classifier': fusion_components['classifier'].state_dict(),
                'feature_alignment_loss': feature_alignment_loss.state_dict(),
                'accuracy': val_metrics['accuracy'],
                'epoch': best_method_epoch,
                'method': fusion_method,
                'val_metrics': val_metrics
            }
            
            # Add method-specific components
            if fusion_method == 'add':
                method_state['quantum_proj'] = fusion_components['quantum_proj'].state_dict()
                method_state['classical_proj'] = fusion_components['classical_proj'].state_dict()
            elif fusion_method == 'attention':
                method_state['attention_module'] = fusion_components['attention_module'].state_dict()
            elif fusion_method == 'self_attention':
                method_state['self_attention_module'] = fusion_components['self_attention_module'].state_dict()
            
            torch.save(method_state, os.path.join(output_dir, f"best_{fusion_method}_model.pth"))
            
            # Store best metrics for final evaluation
            best_method_metrics = {
                'components': fusion_components,
                'feature_alignment': feature_alignment_loss
            }
    
    # Final evaluation on all splits with best model
    print(f"\n{fusion_method.upper()} - FINAL EVALUATION (Best Epoch {best_method_epoch}):")
    
    # Evaluate on train, validation, and test sets
    train_metrics = evaluate_fusion_model(
        quantum_extractor, classical_extractor, best_method_metrics['components'], train_loader, fusion_method, "Train"
    )
    val_metrics = evaluate_fusion_model(
        quantum_extractor, classical_extractor, best_method_metrics['components'], val_loader, fusion_method, "Val"
    )
    test_metrics = evaluate_fusion_model(
        quantum_extractor, classical_extractor, best_method_metrics['components'], test_loader, fusion_method, "Test"
    )
    
    # Store best results for this method
    best_results[fusion_method] = {
        'epoch': best_method_epoch,
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
    
    print(f"\n{fusion_method.upper()} COMPLETE!")

# Final Results Summary
print("\n" + "="*80)
print("COMPREHENSIVE RESULTS SUMMARY")
print("="*80)
print(f"Phase 1 - Best Student Accuracy: {best_student_accuracy:.4f} (Epoch {best_epoch})")
print("\nPhase 2 - Fusion Methods Comparison:")
print("="*80)

# Table header
print(f"{'Method':<15} {'Split':<6} {'Accuracy':<8} {'Precision':<9} {'Recall':<8} {'F1':<8} {'AUC':<8}")
print("-" * 80)

best_fusion_method = None
best_fusion_test_acc = 0.0

for method, results in best_results.items():
    for split_name, metrics in results.items():
        if split_name == 'epoch':
            continue
        
        print(f"{method.upper():<15} {split_name.upper():<6} "
              f"{metrics['accuracy']:<8.4f} {metrics['precision']:<9.4f} "
              f"{metrics['recall']:<8.4f} {metrics['f1']:<8.4f} {metrics['roc_auc']:<8.4f}")
    
    # Track best test accuracy
    if results['test']['accuracy'] > best_fusion_test_acc:
        best_fusion_test_acc = results['test']['accuracy']
        best_fusion_method = method
    
    print("-" * 80)

print("\n" + "="*80)
print("FINAL RANKING BY TEST ACCURACY:")
print("="*80)

# Sort methods by test accuracy
sorted_methods = sorted(best_results.items(), key=lambda x: x[1]['test']['accuracy'], reverse=True)

for rank, (method, results) in enumerate(sorted_methods, 1):
    status = "🏆" if rank == 1 else f"{rank}."
    print(f"{status} {method.upper():<15} - Test Accuracy: {results['test']['accuracy']:.4f} "
          f"(Val: {results['val']['accuracy']:.4f}, Train: {results['train']['accuracy']:.4f})")

print("\n" + "="*80)
print(f"🏆 BEST FUSION METHOD: {best_fusion_method.upper()}")
print(f"🏆 BEST TEST ACCURACY: {best_fusion_test_acc:.4f}")
print("="*80)

# Save comprehensive results
results_summary = {
    'phase1_best_student_accuracy': best_student_accuracy,
    'phase1_best_epoch': best_epoch,
    'phase2_results': best_results,
    'best_fusion_method': best_fusion_method,
    'best_fusion_test_accuracy': best_fusion_test_acc
}

print("\nSaved Models:")
print(f"Best Phase 1: {os.path.join(output_dir, 'best_phase1_model.pth')}")
for method in fusion_methods:
    print(f"Best {method}: {os.path.join(output_dir, f'best_{method}_model.pth')}")