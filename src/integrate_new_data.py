import os
import shutil
import re
from pathlib import Path

def integrate_data(source_root, target_root):
    """
    Reorganizes and merges rock images from source_root into target_root.
    Source structure: source_root/{igneous, metamorphic, sedimentary}/image.png
    Target structure: target_root/class_name/image.png
    """
    source_root = Path(source_root)
    target_root = Path(target_root)
    
    if not source_root.exists():
        print(f"Source directory {source_root} does not exist.")
        return

    # Get existing classes from target_root
    existing_classes = sorted([d.name for d in target_root.iterdir() if d.is_dir()])
    print(f"Found {len(existing_classes)} existing classes in {target_root}")

    # Regex to extract class name from filenames like '100Andesite.png' or 'Amphibolite0.png'
    # It tries to find a known class name within the filename
    class_pattern = "|".join(existing_classes)
    
    count_moved = 0
    count_skipped = 0

    for category in ['igneous', 'metamorphic', 'sedimentary']:
        category_path = source_root / category
        if not category_path.exists():
            continue
            
        print(f"Processing category: {category}")
        for img_path in category_path.glob("*.png"):
            filename = img_path.name
            
            # Match the filename against existing classes (case insensitive)
            match = None
            for cls in existing_classes:
                if cls.lower() in filename.lower():
                    match = cls
                    break
            
            if match:
                dest_dir = target_root / match
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / filename
                
                # Copy the file
                shutil.copy2(img_path, dest_path)
                count_moved += 1
            else:
                print(f"Warning: Could not identify class for {filename}")
                count_skipped += 1

    print(f"Integration complete.")
    print(f"Moved: {count_moved} files")
    print(f"Skipped: {count_skipped} files")

if __name__ == "__main__":
    source = r"d:\Proyecto de Investigación_Cristian Ibadango\Rocks\data\rocks_new\rocks_three"
    target = r"d:\Proyecto de Investigación_Cristian Ibadango\Rocks\data\raw"
    integrate_data(source, target)
