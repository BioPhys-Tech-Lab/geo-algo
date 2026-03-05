import os
import shutil
from pathlib import Path

def integrate_minerals(sources, target_root):
    """
    Merges mineral images from multiple source directories into target_root.
    """
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    
    count_moved = 0
    count_skipped = 0
    
    for source in sources:
        source_path = Path(source)
        if not source_path.exists():
            print(f"Source {source_path} does not exist. Skipping.")
            continue
            
        print(f"Processing source: {source_path}")
        
        # Each subdirectory in source is a mineral class
        for mineral_dir in source_path.iterdir():
            if not mineral_dir.is_dir():
                continue
                
            mineral_name = mineral_dir.name.capitalize()
            dest_dir = target_root / mineral_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all images in the mineral directory
            for img_path in mineral_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    dest_path = dest_dir / f"{source_path.parent.name}_{mineral_name}_{img_path.name}"
                    
                    # Avoid overwriting or unnecessary work if we already copied it (unlikely with unique prefix)
                    if not dest_path.exists():
                        shutil.copy2(img_path, dest_path)
                        count_moved += 1
                    else:
                        count_skipped += 1

    print(f"Mineral integration complete.")
    print(f"Files copied: {count_moved}")
    print(f"Files skipped: {count_skipped}")

if __name__ == "__main__":
    # Define source paths based on exploration
    source_paths = [
        r"d:\Proyecto de Investigación_Cristian Ibadango\Rocks\data\rocks_data_large\data\data",
        r"d:\Proyecto de Investigación_Cristian Ibadango\Rocks\data\rocks_data_large\mineral-trier"
    ]
    target = r"d:\Proyecto de Investigación_Cristian Ibadango\Rocks\data\raw"
    
    integrate_minerals(source_paths, target)
