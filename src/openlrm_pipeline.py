import os
import sys
import torch
import numpy as np

# Add external/OpenLRM to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
openlrm_path = os.path.join(project_root, "external", "OpenLRM")
if openlrm_path not in sys.path:
    sys.path.append(openlrm_path)

# Set environment variables for OpenLRM configuration
# We use the base model as it's a good balance. 
# "zxhezexin/openlrm-mix-base-1.1" is the one used in the demo.
os.environ['APP_MODEL_NAME'] = "zxhezexin/openlrm-mix-base-1.1"
os.environ['APP_INFER'] = os.path.join(openlrm_path, "configs", "infer-gradio.yaml")
# Set APP_ENABLED to 1 to tell LRMInferrer we are in 'app' mode (interactive), 
# effectively bypassing the requirement for CLI 'image_input' and 'export_video/mesh' arguments at init.
os.environ['APP_ENABLED'] = "1"
# Ensure tokenizers don't parallelize to avoid deadlocks in some envs
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from openlrm.runners.infer.lrm import LRMInferrer
except ImportError:
    print("Warning: Could not import LRMInferrer. Ensure requirements are installed.")
    LRMInferrer = None

class OpenLRMPipeline:
    def __init__(self, device='cpu'):
        print(f"Initializing OpenLRM Pipeline on {device}...")
        
        if LRMInferrer is None:
            raise ImportError("OpenLRM module not found.")

        # LRMInferrer uses accelerate, which auto-detects device.
        # To force CPU, we mask CUDA devices.
        if device == 'cpu':
             os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        try:
            self.inferrer = LRMInferrer()
        except Exception as e:
            print(f"Failed to initialize OpenLRM Inferrer: {e}")
            raise

    def run(self, image_path, output_path):
        print(f"Running OpenLRM on {image_path}...")
        
        try:
            # We explicitly define arguments to match infer_single signature
            self.inferrer.infer_single(
                image_path=image_path,
                source_cam_dist=2.0, # Default from app.py
                export_video=False,
                export_mesh=True,
                dump_video_path="", # Not used
                dump_mesh_path=output_path
            )
            
            # Check if file exists
            if os.path.exists(output_path):
                print(f"OpenLRM mesh saved to {output_path}")
                return True
            else:
                print("OpenLRM finished but output file not found.")
                # Sometimes it might save with a different extension or path if logic dictates
                return False
        except Exception as e:
            print(f"OpenLRM inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        pipeline = OpenLRMPipeline(device="cpu")
        pipeline.run(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python src/openlrm_pipeline.py <input_image> <output.obj>")
