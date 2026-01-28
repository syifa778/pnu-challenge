import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import csv
import os
from datetime import datetime
from datasets_classification import build_splits, CarTypeDataset, CLASS_NAMES
from models.cnn_model_classification import CNNCarClassifier
from models.vit_model_classification import ViTCarClassifier
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_model_checkpoint(
    model,
    model_name,
    backbone,
    epoch,
    accuracy,
    macro_f1,
    class_names,
    output_dir="models",
    is_best=False
):
    """Save model checkpoint with metadata"""
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # 1. Save model weights for this epoch
    checkpoint_name = f"epoch_{epoch+1}.pth" if not is_best else "best.pth"
    model_path = os.path.join(model_dir, checkpoint_name)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'macro_f1': macro_f1,
    }
    
    torch.save(checkpoint, model_path)

    # 2. Save metadata
    metadata = {
        "model_name": model_name,
        "backbone": backbone,
        "epoch": epoch,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classes": class_names,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    meta_path = os.path.join(model_dir, f"metrics_epoch_{epoch+1}.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[INFO] Checkpoint saved to {model_path}")
    
    # Also update best.json if this is the best model
    if is_best:
        best_meta_path = os.path.join(model_dir, "best_metrics.json")
        with open(best_meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"[INFO] Best model updated (Accuracy: {accuracy:.4f}, F1: {macro_f1:.4f})")


def save_results_csv(
    model_name,
    backbone,
    accuracy,
    macro_f1,
    per_class_f1,
    class_names,
    epoch=None,
    output_dir="models"
):
    """Save evaluation results to CSV"""
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{model_name}_classification_results.csv")

    row = {
        "model_name": model_name,
        "backbone": backbone,
        "epoch": epoch if epoch is not None else "final",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    for cls, f1 in zip(class_names, per_class_f1):
        row[f"{cls}_f1"] = float(f1)

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[INFO] Results saved to {csv_path}")


def get_transforms(train=True):
    """Get data transforms for training or validation"""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def evaluate_model(model, loader, class_names):
    """Evaluate model and return metrics"""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(DEVICE)).argmax(1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = np.mean(y_true == y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)))
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    return accuracy, macro_f1, per_class_f1


def train_model(model, loaders, class_weights, model_name="resnet50", backbone="cnn", epochs=30):
    """Train the model with checkpoint saving at every epoch"""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-4,
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_val_f1 = 0.0
    
    print(f"\n{'='*60}")
    print(f"Starting training: {model_name} (backbone: {backbone})")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for imgs, labels in loaders["train"]:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches

        # Unfreeze backbone after initial epochs
        if epoch == 5:
            print(f"\n[INFO] Unfreezing backbone at epoch {epoch+1}")
            model.unfreeze_backbone()
            # Reset optimizer to include newly unfrozen parameters
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=1e-5,  # Lower learning rate for fine-tuning
                weight_decay=1e-5
            )

        # Validation phase
        val_accuracy, val_macro_f1, val_per_class_f1 = evaluate_model(
            model, loaders["val"], CLASS_NAMES
        )
        
        # Update learning rate scheduler
        scheduler.step(val_macro_f1)

        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Val Macro F1: {val_macro_f1:.4f}")
        print(f"  Val Per-class F1: {[f'{f1:.3f}' for f1 in val_per_class_f1]}")
        
        # Save checkpoint for this epoch
        is_best = val_macro_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_macro_f1
        
        save_model_checkpoint(
            model=model,
            model_name=model_name,
            backbone=backbone,
            epoch=epoch,
            accuracy=val_accuracy,
            macro_f1=val_macro_f1,
            class_names=CLASS_NAMES,
            output_dir="models",
            is_best=is_best
        )
        
        # Save validation results to CSV
        save_results_csv(
            model_name=model_name,
            backbone=backbone,
            accuracy=val_accuracy,
            macro_f1=val_macro_f1,
            per_class_f1=val_per_class_f1,
            class_names=CLASS_NAMES,
            epoch=epoch+1,
            output_dir="models"
        )

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"{'='*60}\n")
    
    return best_val_f1


def final_evaluation(model, loader, model_name, backbone, class_names):
    """Final evaluation on test set"""
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60 + "\n")
    
    accuracy, macro_f1, per_class_f1 = evaluate_model(model, loader, class_names)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print(f"\nPer-class F1 scores:")
    for cls, f1 in zip(class_names, per_class_f1):
        print(f"  {cls:20s}: {f1:.4f}")
    
    # Get confusion matrix
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(DEVICE)).argmax(1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Save final results
    save_results_csv(
        model_name=model_name,
        backbone=backbone,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        class_names=class_names,
        epoch="test",
        output_dir="models"
    )
    
    return accuracy, macro_f1, per_class_f1


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CAR TYPE CLASSIFICATION TRAINING")
    print("="*60 + "\n")
    
    # Build data splits
    print("[INFO] Loading data splits...")
    train_df, val_df, test_df = build_splits("label_studio_data/clean_car_type.csv")
    
    print(f"[INFO] Train samples: {len(train_df)}")
    print(f"[INFO] Val samples: {len(val_df)}")
    print(f"[INFO] Test samples: {len(test_df)}")
    print(f"[INFO] Classes: {CLASS_NAMES}\n")

    # Create datasets
    train_ds = CarTypeDataset(train_df, "label_studio_data/images", get_transforms(False))
    val_ds   = CarTypeDataset(val_df,   "label_studio_data/images", get_transforms(False))
    test_ds  = CarTypeDataset(test_df,  "label_studio_data/images", get_transforms(False))

    # Create dataloaders
    loaders = {
        "train": DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True),
        "val":   DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True),
        "test":  DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True),
    }

    # Calculate class weights for imbalanced data
    class_counts = train_df["car_type"].value_counts().reindex(CLASS_NAMES).fillna(1)
    class_weights = torch.tensor(1.0 / class_counts.values, dtype=torch.float)
    
    print("[INFO] Class distribution:")
    for cls, count, weight in zip(CLASS_NAMES, class_counts.values, class_weights):
        print(f"  {cls:20s}: {int(count):4d} samples (weight: {weight:.4f})")
    print()

    # Initialize model
    # model_name = "resnet50"
    # backbone_type = "cnn"
    
    model_name = "vit_b_16"
    backbone_type = "vit"
    
    print(f"[INFO] Initializing model: {model_name}")
    # model = CNNCarClassifier(backbone=model_name, num_classes=len(CLASS_NAMES))
    model = ViTCarClassifier(num_classes=len(CLASS_NAMES))
    model.freeze_backbone()
    print(f"[INFO] Backbone frozen for initial training\n")

    # Train model
    best_val_f1 = train_model(
        model=model, 
        loaders=loaders, 
        class_weights=class_weights,
        model_name=model_name,
        backbone=backbone_type,
        epochs=30
    )

    # Load best model for final evaluation
    print("\n[INFO] Loading best model for final evaluation...")
    best_checkpoint_path = os.path.join("models", model_name, "best.pth")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        print(f"[INFO] Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Final evaluation on test set
    test_accuracy, test_macro_f1, test_per_class_f1 = final_evaluation(
        model=model,
        loader=loaders["test"],
        model_name=model_name,
        backbone=backbone_type,
        class_names=CLASS_NAMES
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Final test macro F1: {test_macro_f1:.4f}")
    print("="*60 + "\n")