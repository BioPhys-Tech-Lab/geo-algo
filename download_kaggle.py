import kagglehub
import shutil
from pathlib import Path
import os

def download_and_struct():
    print("Downloading dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("salmaneunus/rock-classification")
        print("Path to dataset files:", path)
        
        # Define target directory
        target_dir = Path("data/kaggle_rocks")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        source_path = Path(path)
        
        # The dataset likely has structure: Type/Class/Images
        # We want to flatten it to: target_dir/Class/Images
        
        print(f"Flattening dataset to {target_dir}...")
        
        # Walk through the downloaded directory
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Get the parent folder name, which should be the Rock Class (e.g., Granite)
                    # Use the immediate parent.
                    # Verify if the parent is a Rock Class or a Type.
                    # Structure: .../Igneous/Granite/image.jpg -> parent is Granite
                    # Structure: .../Granite/image.jpg -> parent is Granite
                    
                    parent_dir = Path(root)
                    class_name = parent_dir.name
                    
                    # Ignore "Igneous", "Sedimentary", "Metamorphic" if they contain subfolders
                    # But if we walk, we are in the bottom directory.
                    # So if we are in .../Igneous/Granite, class_name is Granite.
                    
                    # Create class dir in target
                    target_class_dir = target_dir / class_name
                    target_class_dir.mkdir(exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(os.path.join(root, file), target_class_dir / file)
                    
        print("Dataset structure ready.")
        
        # Verify classes
        classes = [d.name for d in target_dir.iterdir() if d.is_dir()]
        print(f"Found classes: {classes}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_and_struct()
