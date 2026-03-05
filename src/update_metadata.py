import json
from pathlib import Path

def update_metadata():
    classes_path = Path("classes.json")
    metadata_path = Path("src/rock_metadata.json")
    
    if not classes_path.exists():
        print("classes.json not found.")
        return

    with open(classes_path, "r") as f:
        classes = json.load(f)
        
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
        
    updated = False
    for rock_class in classes:
        if rock_class not in metadata:
            print(f"Adding generic metadata for {rock_class}...")
            # Create generic metadata
            # We construct the search URL dynamically
            metadata[rock_class] = {
                "hardness": "N/A (Requires Expert Review)",
                "specific_gravity": "N/A",
                "crystal_system": "N/A",
                "composition": "N/A",
                "fracture": "N/A",
                "mindat_search_url": f"https://www.mindat.org/search.php?search={rock_class}",
                "drp_search_url": f"https://www.digitalrocksportal.org/projects?search={rock_class}"
            }
            updated = True
            
    if updated:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Updated metadata for {len(classes)} classes.")
    else:
        print("Metadata already up to date.")

if __name__ == "__main__":
    update_metadata()
