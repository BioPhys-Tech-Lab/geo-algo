import json
from pathlib import Path
import trimesh
from report_generator import ReportGenerator

def main():
    out_dir = Path("output")
    result_path = out_dir / "result.json"
    
    if not result_path.exists():
        print("Error: content/result.json not found.")
        return

    with open(result_path, "r") as f:
        data = json.load(f)

    rock_type = data["rock_type"]
    confidence = data["confidence"]
    images = data["images"]
    best_idx = data["best_image_index"]
    
    # Check for best model
    # main.py naming might differ if --all-views was used or not
    # If all-views used, we have rock_model_view{best_idx}.obj
    # If not, rock_model.obj
    # Let's try to find the specific view model first, then fallback
    
    model_path = out_dir / f"rock_model_view{best_idx}.obj"
    if not model_path.exists():
        model_path = out_dir / "rock_model.obj"
        
    if not model_path.exists():
        print("Error: No 3D model found.")
        return
        
    print(f"Loading model: {model_path}...")
    mesh = trimesh.load(str(model_path))
    
    metrics = {}
    metrics["volume"] = abs(mesh.volume if mesh.is_watertight else mesh.convex_hull.volume)
    metrics["area"] = mesh.area
    if metrics["area"] > 0:
        metrics["sphericity"] = (3.14159**(1/3) * (6 * metrics["volume"])**(2/3)) / metrics["area"]
    else:
        metrics["sphericity"] = 0
        
    # Ensure standard python floats for JSON/Report compatibility
    metrics = {k: float(v) for k, v in metrics.items()}
    
    # Fix image paths
    images = [str(Path(img).resolve()) for img in images]
        
    print(f"Metrics: {metrics}")
    
    print("Generating report...")
    report_gen = ReportGenerator()
    report_path = out_dir / "report.pdf"
    report_gen.generate_report(str(report_path), rock_type, confidence, metrics, images, str(model_path))
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
