import argparse
import torch
import os
from pathlib import Path
from PIL import Image
try:
    from .classifier import get_model, RockDataset
    from .reconstruction import Reconstructor
    from .visualize import visualize_mesh
except ImportError:
    from classifier import get_model, RockDataset
    from reconstruction import Reconstructor

from report_generator import ReportGenerator

from torchvision import transforms
import json

def load_classifier(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes, pretrained=False) # Ensure we initialize ResNet50 without downloading weights for inference
    # Handle case where model might not exist yet
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")
    model = model.to(device)
    model.eval()
    return model

def predict_rock(model, image_paths, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Custom Normalization for dark rocks
    norm_mean = [0.4, 0.4, 0.4]
    norm_std = [0.25, 0.25, 0.25]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    
    predictions = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            predictions.append(probs)
    
    # Average predictions
    avg_probs = torch.mean(torch.stack(predictions), dim=0)
    top_prob, top_idx = torch.max(avg_probs, 1)
    
    # Find which image had the highest confidence for the predicted class
    predicted_class_idx = top_idx.item()
    best_conf = -1.0
    best_img_idx = 0
    
    for i, probs in enumerate(predictions):
        conf = probs[0][predicted_class_idx].item()
        if conf > best_conf:
            best_conf = conf
            best_img_idx = i

    return classes[predicted_class_idx], top_prob.item(), best_img_idx

def main():
    parser = argparse.ArgumentParser(description="Rock ID and 3D Reconstruction")
    parser.add_argument("images", nargs='+', help="Paths to images of the rock (1 or more)")
    parser.add_argument("--model_path", default="rock_classifier.pth", help="Path to trained classifier")
    parser.add_argument("--output_dir", default="output", help="Directory to save results")
    parser.add_argument("--visualize", action="store_true", help="Visualize the 3D model")
    parser.add_argument("--resolution", type=int, default=512, help="Marching cubes resolution (default: 512)")
    parser.add_argument("--bake", action="store_false", dest="no_bake", help="Disable texture baking")
    parser.set_defaults(bake=True)
    parser.add_argument("--all-views", action="store_true", help="Generate 3D models for ALL input images, not just the best one")
    parser.add_argument("--threshold", type=float, default=25.0, help="Marching cubes isosurface threshold (lower=fuller, higher=thinner)")
    parser.add_argument("--dust3r", action="store_true", help="Use experimental Dust3R Transformer pipeline (requires separate install)")
    parser.add_argument("--lrm", action="store_true", help="Use OpenLRM (Large Reconstruction Model) for high-fidelity 3D (Slow, requires ~16GB RAM)")
    parser.add_argument("--fuse", action="store_true", help="Fuse multiple views into a single mesh (requires --all-views)")
    parser.add_argument("--pro", action="store_true", help="Enable professional PyVista mesh cleanup and filtering")
    args = parser.parse_args()

    # Setup output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    
    # Locate Models
    model_dir = Path(".")
    model_pairs = []
    
    # Pair .pth models with their respective classes.json
    for pth_file in model_dir.glob("rock_classifier*.pth"):
        base_name = pth_file.stem
        
        # Determine expected json name
        if base_name == "rock_classifier":
            json_name = "classes.json"
        else:
            suffix = base_name.split("rock_classifier")[-1]
            json_name = f"classes{suffix}.json"
            
        json_file = model_dir / json_name
        
        if json_file.exists():
            model_pairs.append((pth_file, json_file))
            print(f"Found model pair: {pth_file.name} + {json_file.name}")
        else:
            print(f"Warning: Model {pth_file.name} found, but matching {json_file.name} is missing. Skipping.")

    if not model_pairs:
        print("Error: No valid rock_classifier*.pth and classes*.json pairs found in the current directory.")
        return

    # Load Mineralogical Metadata
    metadata_path = Path("data/rock_metadata.json")
    rock_metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            rock_metadata = json.load(f)
        print(f"Loaded mineralogical metadata for {len(rock_metadata)} rock and mineral types.")
    else:
        print("Warning: data/rock_metadata.json not found.")

    # Ensemble Classification
    print("\nRunning Ensemble Inference...")
    best_overall_confidence = -1.0
    best_overall_rock_type = "Unknown"
    best_overall_img_idx = 0
    winning_model_name = ""
    
    for pth_path, json_path in model_pairs:
        # Load classes for this specific model
        with open(json_path, "r") as f:
            classes = json.load(f)
            
        print(f"  -> Evaluating via {pth_path.name} ({len(classes)} classes)...")
        # Load the architecture
        classifier = load_classifier(str(pth_path), len(classes))
        
        # Predict
        rock_type, confidence, best_img_idx = predict_rock(classifier, args.images, classes)
        print(f"     Prediction: {rock_type} ({confidence*100:.2f}%)")
        
        # Determine highest confidence
        if confidence > best_overall_confidence:
            best_overall_confidence = confidence
            best_overall_rock_type = rock_type
            best_overall_img_idx = best_img_idx
            winning_model_name = pth_path.name
            
        # Free memory before loading the next one
        del classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nFinal Ensemble Result: {best_overall_rock_type} ({best_overall_confidence*100:.2f}%) [Winning Model: {winning_model_name}]")
    print(f"Best view selected: {args.images[best_overall_img_idx]}")
    
    # Save Metadata
    metadata = {
        "rock_type": best_overall_rock_type,
        "confidence": best_overall_confidence,
        "images": args.images,
        "best_image_index": best_overall_img_idx,
        "winning_model": winning_model_name
    }
    
    # Enrich with geological info
    if best_overall_rock_type in rock_metadata:
        metadata["geology"] = rock_metadata[best_overall_rock_type]
        print(f"Added geological info for {best_overall_rock_type}")
    else:
        print(f"No geological metadata found for {best_overall_rock_type}")
             
    # Final metadata will be saved at the end
    # with open(out_dir / "result.json", "w") as f:
    #     json.dump(metadata, f, indent=2)
        
    # 3D Reconstruction and Vision Analysis
    print("\nStarting 3D Reconstruction & Vision Analysis...")
    
    if args.dust3r:
        print("Using Dust3R Transformer Pipeline...")
        try:
            from dust3r_pipeline import Dust3RPipeline
            d3r = Dust3RPipeline()
            output_obj = out_dir / "rock_model_dust3r.obj"
            success = d3r.run(args.images, str(output_obj))
            if success:
                print(f"Done! Dust3R result saved to {output_obj}")
                if args.visualize:
                    visualize_mesh(str(output_obj))
            else:
                print("Dust3R failed or not available. Falling back to TripoSR (ViT-based)...")
                # Fallback logic below
                reconstructor = Reconstructor()
                # Continue with TripoSR logic...
        except ImportError:
             print("Could not import Dust3R pipeline. Falling back to TripoSR...")
             reconstructor = Reconstructor()
    elif args.lrm:
        pass # Don't load TripoSR yet, only if fallback needed
    else:
        reconstructor = Reconstructor()
    
    # OpenLRM Logic
    if args.lrm and not args.dust3r:
        print("Using OpenLRM Pipeline (High Fidelity, CPU Fallback)...")
        try:
             from openlrm_pipeline import OpenLRMPipeline
             lrm = OpenLRMPipeline(device="cpu") # We force CPU as per plan/hardware
             
             # OpenLRM typically takes a single image. Use best image.
             best_image = args.images[best_img_idx]
             output_obj = out_dir / "rock_model_lrm.obj"
             
             lrm.run(best_image, str(output_obj))
             print(f"Done! OpenLRM result saved to {output_obj}")
             
             if args.visualize:
                  visualize_mesh(str(output_obj))
                  # pass
             
             # Ensure report uses this model
             # We set a flag or just overwrite what 'best_model_path' looks for later?
             # For simplicity, main.py logic below looks for 'rock_model.obj' or fused.
             # We should probably update the default path or add logic there.
             # Or just set a variable 'final_model_path' to use in report generation.
             final_model_path = output_obj
             
        except ImportError as e:
             print(f"Could not import OpenLRM pipeline: {e}. Falling back to TripoSR...")
             reconstructor = Reconstructor()
    
    # Only run TripoSR if no Advanced Model was selected/successful
    # 'reconstructor' is only set if dust3r or lrm failed fallback, or if neither was selected.
    if 'reconstructor' in locals():
        if args.all_views:
            print("Generating models for ALL input views...")
            output_objs = []
            for i, img_path in enumerate(args.images):
                # Naming convention: rock_model_view{i}.obj
                out_name = f"rock_model_view{i}.obj"
                out_path = out_dir / out_name
                print(f"  Processing View {i+1}/{len(args.images)}: {img_path}")
                try:
                    reconstructor.generate_3d(img_path, str(out_path), resolution=args.resolution, bake=args.bake, threshold=args.threshold, pro_clean=args.pro)
                except Exception as e:
                    print(f"  Error generating 3D for {img_path}: {e}")
                    print("  Retrying with higher threshold (50.0) to reduce memory usage...")
                    try:
                        reconstructor.generate_3d(img_path, str(out_path), resolution=args.resolution, bake=args.bake, threshold=50.0, pro_clean=args.pro)
                    except Exception as e2:
                         print(f"  Failed again: {e2}")
                output_objs.append(out_path)
                
            print(f"Done! All views saved in {out_dir}")
            
            # Fusion Step
            if args.fuse:
                if len(output_objs) < 2:
                    print("Warning: Need at least 2 models for fusion. Skipping fusion step.")
                else:
                    print("Fusing multiple views into a single mesh...")
                try:
                    from mesh_fusion import fuse_meshes
                    import shutil

                    # Convert Path objects to strings if needed
                    mesh_paths = [str(p) for p in output_objs]
                    fused_path = out_dir / "rock_model_fused.obj"
                    
                    fuse_meshes(mesh_paths, str(fused_path))
                    print(f"Fused model saved to {fused_path}")
                    
                    # Update output_obj to point to fused model for visualization/report
                    output_obj = fused_path
                    
                    # Also update best_idx to -1 or similar to indicate fused model use?
                    # For now, let's just ensure report uses this if it exists.
                except Exception as e:
                    print(f"Fusion failed: {e}")

            # Visualize the best one by default, but maybe mention others?
            if args.visualize:
                if args.fuse and (out_dir / "rock_model_fused.obj").exists():
                    print("Visualizing FUSED model...")
                    # visualize_mesh(str(out_dir / "rock_model_fused.obj"))
                    pass # Don't block
                else:
                    print("Visualizing STITCHED model...")
                    # visualize_mesh(str(output_objs[best_img_idx]))
                    pass
                
        else:
            # Use the best image for reconstruction
            best_image = args.images[best_img_idx] 
            output_obj = out_dir / ("rock_model_pro.obj" if args.pro else "rock_model.obj")
            try:
                reconstructor.generate_3d(best_image, str(output_obj), resolution=args.resolution, bake=args.bake, threshold=args.threshold, pro_clean=args.pro)
            except Exception as e:
                 print(f"Error generating 3D: {e}")
                 print("Retrying with higher threshold (50.0)...")
                 reconstructor.generate_3d(best_image, str(output_obj), resolution=args.resolution, bake=args.bake, threshold=50.0, pro_clean=args.pro)

    # Results Visualization
    if args.visualize and 'output_obj' in locals():
        visualize_mesh(str(output_obj))

    # Generate PDF Report
    try:
        print("\nGenerating Doctoral PDF Report...")
        report_gen = ReportGenerator()
        # Use metrics from the last generated model (or best view if single)
        # We need to capture metrics from generate_3d. 
        # But generate_3d returns nothing currently (printed only). Time to fix that in main logic?
        # Actually I updated generate_3d to return metrics.
        # But I didn't capture them in the loop above.
        # Let's re-run metrics calculation or just capture them.
        # Wait, I can't easily capture them from the loop without changing structure significantly.
        # Let's just calculate metrics for the BEST view again? Or make generate_3d return them and capture.
        
        # Simpler: Just load the best view mesh and calculate metrics here?
        # No, better to use the ones from reconstruction.
        
        # Let's assuming we just run it for the best view to get metrics for the report.
        # Or I can just load the obj file and calculate metrics using trimesh here?
        # That avoids re-running generation.
        
        import trimesh
        import trimesh
        # Prioritize OpenLRM / Dust3R / Fused / TripoSR in that order
        if args.pro and (out_dir / "rock_model_pro.obj").exists():
            best_model_path = out_dir / "rock_model_pro.obj"
        elif args.lrm and (out_dir / "rock_model_lrm.obj").exists():
            best_model_path = out_dir / "rock_model_lrm.obj"
        elif args.dust3r and (out_dir / "rock_model_dust3r.obj").exists():
             best_model_path = out_dir / "rock_model_dust3r.obj"
        elif args.fuse and (out_dir / "rock_model_fused.obj").exists():
             best_model_path = out_dir / "rock_model_fused.obj"
        else:
             # Fallback to best view from TripoSR
             best_model_path = out_dir / f"rock_model_view{best_img_idx}.obj" if args.all_views else out_dir / "rock_model.obj"
        
        if best_model_path.exists():
            print(f"Calculating metrics for {best_model_path.name}...")
            mesh = trimesh.load(str(best_model_path))
            vol = abs(mesh.volume if mesh.is_watertight else mesh.convex_hull.volume)
            area = mesh.area
            sphericity = (3.14159**(1/3) * (6 * vol)**(2/3)) / area if area > 0 else 0
            metrics = {"volume": vol, "area": area, "sphericity": sphericity}
            
            # Convert to float for JSON/FPDF
            metrics = {k: float(v) for k, v in metrics.items()}
            
            report_path = out_dir / "report.pdf"
            
            # Construct Geological Info
            geology_info = {
                "type": "Igneous" if rock_type in ["Basalt", "Granite"] else "Sedimentary" if rock_type in ["Sandstone", "Limestone", "Coal"] else "Metamorphic",
                "formation": "Automatically identified",
                "description": f"Identified as {rock_type} with {confidence*100:.1f}% confidence."
            }
            
            # Enrich with expert metadata if available
            if rock_type in rock_metadata:
                expert_data = rock_metadata[rock_type]
                geology_info.update(expert_data)

            # Find 3D screenshot preview
            screenshot_3d = str(best_model_path).replace(".obj", "_preview.png")
            if not os.path.exists(screenshot_3d):
                # Fallback to general screenshot if preview doesn't exist
                screenshot_3d = str(out_dir / "screenshot_rock_model.png")
                
            # Finalize metadata with paths for the dashboard
            metadata.update({
                "obj_path": "output/" + best_model_path.name,
                "mtl_path": "output/" + best_model_path.with_suffix(".mtl").name,
                "texture_path": "output/" + best_model_path.with_suffix(".png").name,
                "preview_path": "output/" + os.path.basename(screenshot_3d) if screenshot_3d else None,
                "metrics": metrics
            })
            
            # Save final result.json
            with open(out_dir / "result.json", "w") as f:
                json.dump(metadata, f, indent=2)
                print(f"Final metadata saved to {out_dir / 'result.json'}")

            report_gen.generate_report(str(report_path), rock_type, confidence, metrics, [str(img) for img in args.images[:3]], str(best_model_path), geology_info, screenshot_3d=screenshot_3d)
            print(f"Report saved to {report_path}")
        else:
            print("Could not find model to generate report.")
            # Save metadata even if report fails
            with open(out_dir / "result.json", "w") as f:
                json.dump(metadata, f, indent=2)

    except Exception as e:
        print(f"Failed to generate report: {e}")

if __name__ == "__main__":
    main()
