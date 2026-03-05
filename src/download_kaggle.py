import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    api = KaggleApi()
    try:
        api.authenticate()
        print("Authenticated successfully.")
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("Please ensure kaggle.json is in ~/.kaggle/ or KAGGLE_USERNAME/KAGGLE_KEY env vars are set.")
        return

    dataset = "neelgajare/rocks-dataset"
    path = "data/downloaded"
    
    print(f"Downloading {dataset} to {path}...")
    try:
        api.dataset_download_files(dataset, path=path, unzip=True)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    download_dataset()
