import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

class RockDataset(Dataset):
    def __init__(self, root_dir, image_paths, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = [(str(p), self.class_to_idx[p.parent.name]) for p in image_paths]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            # Fallback to a black image
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
        return image, label

def get_model(num_classes, pretrained=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    # Add dropout for better generalization
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

def train_model(data_dir, num_epochs=5, batch_size=32, learning_rate=0.001, limit=None, max_samples_per_class=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Custom Normalization for dark rocks
    norm_mean = [0.4, 0.4, 0.4]
    norm_std = [0.25, 0.25, 0.25]

    # Data transformation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
    }

    # Split logic to prevent data leakage (Augmented clones stay with originals)
    root_dir = Path(data_dir)
    classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
    
    train_paths = []
    val_paths = []
    
    for cls_name in classes:
        class_dir = root_dir / cls_name
        class_images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        
        # Sort so we reliably identify originals
        class_images.sort()
        
        if max_samples_per_class and len(class_images) > max_samples_per_class:
            class_images = class_images[:max_samples_per_class]
        elif limit:
            class_images = class_images[:limit]
            
        # Group originals with their augmentations
        image_groups = {}
        for img in class_images:
            name = img.name
            if name.startswith("aug_"):
                # Extract the original name
                # format is generally aug_X_original_name.jpg
                parts = name.split('_', 2)
                if len(parts) >= 3:
                    orig_name = parts[2]
                else:
                    orig_name = name # fallback
            else:
                orig_name = name
                
            if orig_name not in image_groups:
                image_groups[orig_name] = []
            image_groups[orig_name].append(img)
            
        # Now 80/20 train/val split based on original image groups
        group_keys = list(image_groups.keys())
        import random
        random.seed(42) # Ensure reproducible splits
        random.shuffle(group_keys)
        
        split_idx = int(0.8 * len(group_keys))
        train_keys = group_keys[:split_idx]
        val_keys = group_keys[split_idx:]
        
        for k in train_keys:
            train_paths.extend(image_groups[k])
        for k in val_keys:
             # Do not include strongly augmented images in validation, only originals if possible
             # Or just include all. But conceptually validation should be original images. 
             for img in image_groups[k]:
                 if not img.name.startswith("aug_"):
                     val_paths.append(img)
             
             # if the original wasn't available (only augmented were selected by the class limit),
             # we grab at least one of its variants for validation so the validation set has data.
             if not any(not img.name.startswith("aug_") for img in image_groups[k]):
                 if image_groups[k]:
                     val_paths.append(image_groups[k][0])


    print(f"Total training images: {len(train_paths)}")
    print(f"Total validation images: {len(val_paths)}")

    if len(train_paths) == 0:
        print("No training images found!")
        return

    # Datasets
    train_dataset = RockDataset(data_dir, train_paths, transform=data_transforms['train'])
    val_dataset = RockDataset(data_dir, val_paths, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = get_model(len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4) # Added weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # Added LR Scheduler

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # distinct phases directly in loop for brevity
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            batch_idx = 0
            total_batches = len(dataloader)
            for inputs, labels in dataloader:
                batch_idx += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if batch_idx % 100 == 0 or batch_idx == total_batches:
                    print(f"  Batch {batch_idx}/{total_batches} ({100*batch_idx/total_batches:.1f}%) | Loss: {loss.item():.4f}")

            if phase == 'train':
                 scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), "rock_classifier.pth")
    print("Model saved to rock_classifier.pth")
    
    # Save classes
    import json
    with open("classes.json", "w") as f:
        json.dump(classes, f)
    print("Classes saved to classes.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--limit", type=int, default=None, help="Limit images per class for fast training")
    parser.add_argument("--max_samples_per_class", type=int, default=None, help="Maximum images per class")
    args = parser.parse_args()
    
    # Ensure data exists before running
    if os.path.exists(args.data_dir):
        train_model(args.data_dir, num_epochs=args.epochs, batch_size=args.batch_size, limit=args.limit, max_samples_per_class=args.max_samples_per_class)
    else:
        print(f"Data directory '{args.data_dir}' not found. Please check your path.")
