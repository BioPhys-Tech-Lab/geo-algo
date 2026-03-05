import sys
print("Starting script...")
try:
    import torch
    print(f"Torch imported: {torch.__version__}")
except ImportError as e:
    print(f"Failed to import torch: {e}")

try:
    import transformers
    print(f"Transformers imported: {transformers.__version__}")
except ImportError as e:
    print(f"Failed to import transformers: {e}")

from src.reconstruction import Reconstructor

def test_triposr_loading():
    print("Testing TripoSR loading via Reconstructor...")
    try:
        rec = Reconstructor()
        if rec.model is not None:
            print("Successfully loaded TripoSR model.")
            return True
        else:
            print("Failed to load model (rec.model is None)")
            return False
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to load model: {e}")
        return False

if __name__ == "__main__":
    if test_triposr_loading():
        sys.exit(0)
    else:
        sys.exit(1)
