import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from src.model import get_tb_model


def set_seed(seed: int = 42) -> None:
    """
    Enforces deterministic execution across random, numpy, and torch environments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# Hardware execution target
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active compute device: {device}")

# Dataset locations and training parameters
TRAIN_DIR = os.path.join("data", "processed", "train")
VAL_DIR = os.path.join("data", "processed", "val")
MODEL_DIR = "models"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_model.pth")

BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

# Confirm dataset directory existence before pipeline execution
if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    raise FileNotFoundError(
        f"Data directories missing. Verify '{TRAIN_DIR}' and '{VAL_DIR}' exist."
    )

# Preprocessing transforms (data augmentation on train, deterministic on val)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize image folders
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_transforms)

# DataLoaders: num_workers=0 ensures zero subprocess deadlock on Windows CPU
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print(f"Samples: {len(train_dataset)} training, {len(val_dataset)} validation.")
print(f"Class indices: {train_dataset.class_to_idx}")

# Initialize model, loss function, and optimizer
model = get_tb_model(num_classes=len(train_dataset.classes), pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()

# Pass only trainable parameters to Adam to minimize optimizer memory overhead
trainable_parameters = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.Adam(trainable_parameters, lr=LEARNING_RATE)

os.makedirs(MODEL_DIR, exist_ok=True)
best_val_accuracy = 0.0

# Execute Epoch Loops
for epoch in range(NUM_EPOCHS):
    # Training Stage
    model.train()
    running_train_loss = 0.0
    train_correct = 0
    train_total = 0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Training",
        leave=True,
        unit="batch"
    )

    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_len = images.size(0)
        running_train_loss += loss.item() * batch_len
        _, preds = torch.max(outputs, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += batch_len

        # Live metric telemetry
        batch_acc = (train_correct / train_total) * 100
        train_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{batch_acc:.2f}%"
        )

    epoch_train_loss = running_train_loss / train_total
    epoch_train_acc = train_correct / train_total

    # Validation Stage
    model.eval()
    running_val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        val_bar = tqdm(
            val_loader,
            desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Validate",
            leave=False,
            unit="batch"
        )
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_len = images.size(0)
            running_val_loss += loss.item() * batch_len
            _, preds = torch.max(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += batch_len

    epoch_val_loss = running_val_loss / val_total
    epoch_val_acc = val_correct / val_total

    print(
        f"\nEpoch {epoch + 1} Summary -> "
        f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc * 100:.2f}% | "
        f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc * 100:.2f}%"
    )

    # Save checkpoint when validation accuracy improves
    if epoch_val_acc > best_val_accuracy:
        best_val_accuracy = epoch_val_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"Optimal checkpoint updated: '{CHECKPOINT_PATH}' (Val Acc: {epoch_val_acc * 100:.2f}%)\n")

print(f"\nExecution complete. Peak Validation Accuracy: {best_val_accuracy * 100:.2f}%")