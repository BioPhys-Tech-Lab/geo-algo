import os
import shutil
from pathlib import Path
import json

def merge_datasets():
    source_dir = Path("data/Rocks")
    dest_dir = Path("data/raw")
    
    if not source_dir.exists():
        print(f"Source directory {source_dir} not found!")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    
    classes = set()
    
    # Get all subdirectories (classes) in source
    for class_path in source_dir.iterdir():
        if class_path.is_dir():
            class_name = class_path.name
            classes.add(class_name)
            
            dest_class_dir = dest_dir / class_name
            dest_class_dir.mkdir(exist_ok=True)
            
            print(f"Processing {class_name}...")
            
            # Move files
            for img_file in class_path.glob("*"):
                if img_file.is_file():
                    # Generate a unique name to avoid collisions if merging
                    # We can prepend 'neelgajare_' or just rely on uuid/names
                    dest_file = dest_class_dir / img_file.name
                    if not dest_file.exists():
                         shutil.move(str(img_file), str(dest_file))
                    else:
                         print(f"Skipping {img_file.name}, already exists.")
            
            # Remove empty source dir
            try:
                class_path.rmdir() 
            except:
                pass # might not be empty if hidden files

    # Scan dest_dir to get final class list (merging with existing)
    final_classes = [d.name for d in dest_dir.iterdir() if d.is_dir()]
    final_classes.sort()
    
    print(f"Merged dataset contains {len(final_classes)} classes.")
    
    # Update classes.json
    with open("classes.json", "w") as f:
        json.dump(final_classes, f, indent=4)
    print("Updated classes.json")

if __name__ == "__main__":
    merge_datasets()
