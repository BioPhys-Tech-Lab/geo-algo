import os
import argparse
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms
import random

def augment_class(class_dir, target_count):
    class_dir = Path(class_dir)
    images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
    
    current_count = len(images)
    if current_count >= target_count or current_count == 0:
        return 0

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

    needed = target_count - current_count
    generated = 0

    print(f"Augmenting {class_dir.name} from {current_count} to {target_count} ({needed} needed)")

    while generated < needed:
        img_path = random.choice(images)
        try:
            img = Image.open(img_path).convert("RGB")
            aug_img = transform(img)
            
            # Save augmented image
            new_name = class_dir / f"aug_{generated}_{img_path.name}"
            aug_img.save(new_name, "JPEG")
            generated += 1
        except Exception as e:
            print(f"Error augmenting {img_path}: {e}")
            pass

    return generated

def main():
    parser = argparse.ArgumentParser(description="Offline data augmentation for minority classes.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to dataset directory")
    parser.add_argument("--target", type=int, default=500, help="Target number of images per class")
    args = parser.parse_args()

    root_dir = Path(args.data_dir)
    if not root_dir.exists():
        print(f"Data directory {root_dir} not found.")
        return

    classes = [d for d in root_dir.iterdir() if d.is_dir()]
    total_augmented = 0

    for cls_dir in classes:
        augmented = augment_class(cls_dir, args.target)
        total_augmented += augmented

    print(f"Augmentation complete. Generated {total_augmented} new images.")

if __name__ == "__main__":
    main()
