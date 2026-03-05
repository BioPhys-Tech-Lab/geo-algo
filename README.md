# Geo-Algo: Advanced Vision & 3D Reconstruction Pipeline for Geological Minerals

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![TripoSR](https://img.shields.io/badge/TripoSR-3D%20Reconstruction-orange)](https://github.com/VAST-AI-Research/TripoSR)

**Geo-Algo** is a sophisticated computer vision platform designed to bridge the gap between 2D geological imagery and 3D architectural/mineralogical analysis. Utilizing an ensemble of specialized deep-learning models, it offers high-accuracy robust classification, fully automated 3D mesh reconstruction, and rich geological context generation all from simple 2D rock images.

---
##📥 Dataset & Pre-trained Models
Due to their massive size, the full dataset and the compiled ensemble weights are hosted externally. You must download the model weights to run inferences locally.

##🪨 Dataset (Zenodo): The complete 70,000 augmented image dataset covering 63 distinct mineralogical classes is available open-access on Zenodo. [Download Dataset Here (Zenodo) -> https://drive.google.com/drive/folders/1CIjmAcxCM40NxCHJ4xHa7kPSw2I-Qqe2?usp=sharing]

##🧠 Model Weights (Google Drive): Download the pre-trained .pth ensemble models required for classification. [Download Weights Here (Google Drive) -> ]. Note: Place all downloaded .pth files directly into the root directory of this repository.
---
## 🌟 Key Features

### 1. 🧠 Dynamic Ensemble Classification Architecture
Rather than relying on a single monolithic model which is prone to class imbalance, the platform utilizes a **Dynamic Ensemble Engine**. Built on top of `ResNet50` backbones, the system seamlessly loads multiple distinct models (e.g., massive 68-class general models combined with highly-tuned, niche 7-10 class models). 
* Images run through all available weights concurrently.
* The system resolves predictions using a "Winner-Takes-All" confidence polling system, ensuring incredibly accurate edge-case detection.
* Custom dataloader normalizations tailored strictly for geological/dark textures (`mean=[0.4, 0.4, 0.4]`, `std=[0.25, 0.25, 0.25]`).

### 2. 🧊 Automated 3D Reconstruction (TripoSR)
Once the mineral is classified, the platform feeds the highest-confidence perspective into **TripoSR** (a state-of-the-art transformer-based 3D generation model) to construct an incredibly detailed 3D OBJ mesh of the specimen in under a few seconds.
* Automatically bakes textures onto the generated geometry.
* Features integrated mesh repair, Taubin smoothing algorithms, and small disconnected-part removal for completely rendering-ready assets.

### 3. 📄 Automated Metadata & PDF Report Generation
Every successful positive classification directly interfaces with the platform's robust `rock_metadata.json` registry. The system pulls relevant geological features, historical data, and chemical compositions, pairing them with calculated 3D volume/sphericity metrics to build and export a fully-fledged professional PDF report.

### 4. 🎛️ Interactive Web Dashboard
A sleek, modern web dashboard interface allowing researchers and users to upload their images via drag-and-drop, track the realtime prediction and 3D reconstruction logs, and interactively view the final 3D `.obj` model directly in their browser using Three.js visualization.

---

## 🚀 Quickstart Guide

### Prerequisites
* Windows/Linux with Python 3.10+
* An NVIDIA GPU with CUDA Toolkit installed (Minimum 8GB VRAM recommended for Ensemble + 3D Generation handling).

### Installation
Clone the repository and install the strict dependencies:
```bash
git clone https://github.com/BioPhys-Tech-Lab/geo-algo.git
cd geo-algo
pip install -r requirements_fixed.txt
```

### Running Inference & Reconstruction
Run `main.py` targeting an image. The script will automatically discover all compiled `.pth` ensemble models in the root directory and attempt a classification and 3D reconstruction.

```bash
python src/main.py my_rock_image.jpg
```
The output, including the `.obj` mesh, `.png` texture maps, and `.pdf` report will be generated into the `/output` folder.

### Running the Web Dashboard
Boot up the integrated Flask/WebSocket server to launch the GUI panel:
```bash
python src/serve_dashboard.py
```
Open `http://localhost:5000` in your browser.

---

## 🛠️ Data Augmentation & Training Pipelines

The pipeline includes scripts to construct, balance, and train your own geological classes on top of the ResNet50 backbone. Note that actual dataset images and `.pth` weights are excluded from this repository due to their size.

* **Augmentation**: `src/augment_data.py` - Runs offline synthetic data augmentation via flips, rotations, and color jittering to mitigate severe natural mineral class imbalances.
* **Leakage Prevention**: Our `classifier.py` script utilizes custom train/validation splitting directly at the file level *before* torch Dataset loading, mathematically preventing augmented variants of the same source image from crossing bounds.
* **Training Execution**: 
```bash
python src/classifier.py --data_dir "data/Rocks" --epochs 25 --batch_size 16
```
