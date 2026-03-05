import sys
import os
import torch
from pathlib import Path

# Add Dust3R core to path
DUST3R_PATH = Path(__file__).parent / "dust3r_core"
if DUST3R_PATH.exists():
    sys.path.append(str(DUST3R_PATH))

try:
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    HAS_DUST3R = True
except ImportError as e:
    print(f"Warning: Could not import Dust3R: {e}")
    HAS_DUST3R = False

class Dust3RPipeline:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = None
        if HAS_DUST3R:
            try:
                # Load a standard model (e.g. ViT-Large based)
                # This requires downloading weights. 
                # For this demo, we might need to handle weight download or just fail if not present.
                model_name = "naver/dust3r_vit_large"
                self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(self.device)
            except Exception as e:
                print(f"Failed to load Dust3R model: {e}")
                self.model = None

    def run(self, image_paths, output_path):
        if not HAS_DUST3R or not self.model:
            print("Dust3R is not available. Please install dependencies.")
            return False

        print(f"Running Dust3R on {len(image_paths)} images...")
        
        # Load images
        images = load_images(image_paths, size=512)
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        
        # Inference
        output = inference(pairs, self.model, self.device, batch_size=1)
        
        # Global Alignment
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.compute_global_alignment(init="mst", niter=300, schedule='linear', lr=0.01)
        
        # Export
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(1.0)))
        mesh = scene.get_mesh()
        mesh.export(output_path)
        print(f"Dust3R mesh saved to {output_path}")
        return True
