import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path

def calculate_mean_std(data_dir, batch_size=32):
    # Resize and tensor only (no augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    
    # Calculate Mean and Std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = len(dataset)
    
    print(f"Calculating mean and std for {total_images} images...")
    
    # Compute mean
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / total_images
    
    # Compute std (variance approach)
    variance = torch.zeros(3)
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        # Note: broadcasting mean
        variance += ((images - mean.unsqueeze(1))**2).mean(2).sum(0)
    std = torch.sqrt(variance / total_images)
    
    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw")
    args = parser.parse_args()
    
    mean, std = calculate_mean_std(args.data_dir)
    print(f"Dataset Mean: {mean}")
    print(f"Dataset Std: {std}")
