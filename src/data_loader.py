import os
import requests
from pyinaturalist import get_observations, get_projects
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def search_geology_projects(query="geology"):
    """Searches for projects related to geology to find curated lists of rocks."""
    projects = get_projects(q=query)
    print(f"Found {len(projects['results'])} projects matching '{query}':")
    for p in projects['results'][:5]:
        print(f" - {p['id']}: {p['title']} ({p['description'][:50]}...)")
    return projects

def download_image(url, save_path):
    """Downloads a single image."""
    try:
        if os.path.exists(save_path):
            return # Skip if exists
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def fetch_rock_images(rock_types, output_dir="dataset", photos_per_type=50):
    """
    Searches for observations matching specific rock types and downloads images.
    
    Args:
        rock_types (list): List of strings, e.g., ["Granite", "Basalt"]
        output_dir (str): Root directory for the dataset.
        photos_per_type (int): Target number of images per class.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for rock in rock_types:
        print(f"Searching for {rock}...")
        class_dir = output_dir / rock
        class_dir.mkdir(exist_ok=True, parents=True)
        
        # Search for observations. 
        # Since rocks aren't a Taxon, we search by query string 'q'.
        # We try to filter for 'research' grade if possible, but for rocks 'casual' might be necessary.
        observations = get_observations(
            q=rock,
            per_page=photos_per_type * 2, # Request more to filter null photos
            photos=True
            # quality_grade="any"  # Removed as it caused error, default includes all
        )
        
        count = 0
        download_tasks = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            for obs in observations['results']:
                if count >= photos_per_type:
                    break
                
                if not obs['photos']:
                    continue
                
                # Get the medium size photo URL
                photo_url = obs['photos'][0]['url'].replace("square", "medium")
                obs_id = obs['id']
                file_name = f"{obs_id}.jpg"
                save_path = class_dir / file_name
                
                download_tasks.append(executor.submit(download_image, photo_url, save_path))
                count += 1
                
        print(f"Queued {count} downloads for {rock}")

if __name__ == "__main__":
    # Example usage
    rocks = ["Granite", "Basalt", "Limestone", "Sandstone"]
    fetch_rock_images(rocks, output_dir="data/raw", photos_per_type=200)
