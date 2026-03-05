import torch
import numpy as np
import rembg
import trimesh
from PIL import Image
import os
import sys
import open3d as o3d
try:
    from .vision_analysis import VisionAnalyzer
except ImportError:
    from vision_analysis import VisionAnalyzer

# Add TripoSR to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'TripoSR'))

try:
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground
    # Try importing baking dependencies
    try:
        import xatlas
        import moderngl
        from tsr.bake_texture import bake_texture
        CAN_BAKE = True
    except ImportError as e:
        print(f"Texture baking dependencies missing: {e}")
        CAN_BAKE = False
except ImportError as e:
    print(f"Error importing TripoSR (tsr): {e}")
    with open("error.log", "w") as f:
        f.write(f"ImportError: {e}")
    TSR = None

class Reconstructor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading TripoSR on {self.device}...")
        self.vision_analyzer = VisionAnalyzer()
        
        if TSR is None:
            print("TripoSR module not found. Please ensure src/TripoSR exists.")
            self.model = None
            return

        try:
            self.model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            self.model.to(self.device)
            # Optional: set chunk size
            self.model.renderer.set_chunk_size(8192)
            
        except Exception as e:
            print(f"Error loading TripoSR model: {e}")
            with open("error.log", "w") as f:
                f.write(str(e))
                import traceback
                traceback.print_exc(file=f)
            self.model = None

    def preprocess_image(self, image_path):
        """Removes background and resizes image using TripoSR utils."""
        input_image = Image.open(image_path).convert("RGB")
        
        # Enhance image contrast and sharpness to help with structure
        from PIL import ImageEnhance
        input_image = ImageEnhance.Contrast(input_image).enhance(1.2)
        input_image = ImageEnhance.Sharpness(input_image).enhance(1.5)
        
        # Use TripoSR's recommended preprocessing with alpha matting for better edges
        rembg_session = rembg.new_session()
        image = remove_background(input_image, rembg_session, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10)
        image = resize_foreground(image, 0.85)
        
        # Save debug image to check segmentation
        os.makedirs("output", exist_ok=True)
        image.save("output/debug_segmented.png")
        
        # Composite on gray background as per run.py
        image = np.array(image).astype(np.float32) / 255.0
        # Alpha blending with gray (0.5) background
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        
        image.save("output/debug_input_to_model.png")
        return image

    def generate_3d(self, image_path, output_path="output/rock_model.obj", resolution=512, bake=True, threshold=25.0, pro_clean=False):
        if not self.model:
            print("Model not loaded.")
            return

        print(f"Processing {image_path}...")
        print(f"Config: resolution={resolution}, bake_texture={bake and CAN_BAKE}")
        
        # Preprocess
        image = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            scene_codes = self.model([image], device=self.device)
            
            # Extract Mesh
            # Note: extract_mesh second arg is has_vertex_color. 
            # If baking texture, we usually want vertex colors ONLY if not baking? 
            # Actually run.py does: meshes = model.extract_mesh(..., not args.bake_texture, ...)
            # So if baking, pass False? 
            # Let's check run.py again. 
            # run.py: meshes = model.extract_mesh(scene_codes, not args.bake_texture, ...)
            
            should_vertex_color = not (bake and CAN_BAKE)
            meshes = self.model.extract_mesh(scene_codes, should_vertex_color, resolution=resolution, threshold=threshold)
            
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Select the initial mesh
            mesh = meshes[0]
            
            # 2. Polishing and repairing mesh (Cleaning + Repair + Smoothing)
            print(f"Polishing and repairing mesh (Pro={pro_clean})...")
            
            # Cleaning: Remove small disconnected artifacts
            try:
                # More aggressive split for pro_clean
                components = mesh.split(only_watertight=False)
                if len(components) > 1:
                    print(f"  Removing {len(components)-1} small disconnected parts...")
                    components.sort(key=lambda m: m.area, reverse=True)
                    mesh = components[0]
            except Exception as e:
                print(f"  Warning: Mesh cleaning failed: {e}")
            
            # Repair: Fix normals and fill small holes
            try:
                print("  Attempting mesh repair (normals + holes)...")
                mesh.fix_normals()
                trimesh.repair.fill_holes(mesh)
                
                if not mesh.is_winding_consistent:
                    print("  Fixing inconsistent winding order...")
                    trimesh.repair.fix_winding(mesh)
            except Exception as e:
                print(f"  Warning: Mesh repair failed: {e}")
            
            # Smoothing: Taubin is better at preserving volume than Laplacian
            try:
                iters = 100 if pro_clean else 50
                print(f"  Applying Taubin smoothing ({iters} iterations)...")
                trimesh.smoothing.filter_taubin(mesh, iterations=iters)
                
                if pro_clean:
                    # Step A: Point Cloud Refinement (Open3D Outlier Removal)
                    print("  Step A: Refining point cloud (Open3D Statistical Outlier Removal)...")
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
                    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                    
                    # If we removed outliers, keep only the relevant vertices and update faces
                    if len(ind) < len(mesh.vertices):
                        print(f"    Removed {len(mesh.vertices) - len(ind)} outlier points.")
                        mesh.update_vertices(ind)
                    
                    # Step B: Laplacian Surface Smoothing
                    print("  Step B: Applying Laplacian Surface Smoothing...")
                    # Laplacian helps with high-frequency noise that Taubin might miss
                    trimesh.smoothing.filter_laplacian(mesh, iterations=10)
                    
            except Exception as e:
                print(f"  Warning: Advanced smoothing failed: {e}")

            # 3. Handle Texture Baking or Plain Export
            if bake and CAN_BAKE:
                print("Baking texture...")
                texture_resolution = 2048
                bake_output = bake_texture(mesh, self.model, scene_codes[0], texture_resolution)
                
                # Export textured mesh
                print("Exporting textured mesh...")
                xatlas.export(
                    output_path, 
                    mesh.vertices[bake_output["vmapping"]], 
                    bake_output["indices"], 
                    bake_output["uvs"], 
                    mesh.vertex_normals[bake_output["vmapping"]]
                )
                
                # Save texture image
                texture_path = os.path.splitext(output_path)[0] + ".png"
                Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save(texture_path)
                print(f"Texture saved to {texture_path}")
                
                # Step C: Save Normal Map
                normal_path = os.path.splitext(output_path)[0] + "_normal.png"
                Image.fromarray((bake_output["normals"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save(normal_path)
                print(f"Normal map saved to {normal_path}")
                
                # Link MTL
                mtl_path = os.path.splitext(output_path)[0] + ".mtl"
                tex_filename = os.path.basename(texture_path)
                norm_filename = os.path.basename(normal_path)
                with open(mtl_path, "w") as f:
                    f.write(f"newmtl material_0\n")
                    f.write(f"map_Kd {tex_filename}\n")
                    f.write(f"map_Bump {norm_filename}\n") # Standard MTL bump/normal mapping
                
                # Prepend MTL reference to OBJ
                with open(output_path, 'r+') as f:
                    content = f.read()
                    f.seek(0, 0)
                    f.write(f"mtllib {os.path.basename(mtl_path)}\nusemtl material_0\n" + content)
            else:
                if bake and not CAN_BAKE:
                    print("  Warning: Baking requested but dependencies missing. Exporting plain mesh.")
                mesh.export(output_path)

            
        print(f"3D model saved to {output_path}")
        
        # Calculate Object Metrics with robust fallback
        metrics = {}
        try:
            # Re-ensure normals are correct before volume calculation
            trimesh.repair.fix_normals(mesh)
            
            # Use absolute volume as single-view reconstructions can be inverted
            # If not watertight, convex_hull is a decent approximation for solid volume
            if mesh.is_watertight:
                metrics["volume"] = abs(mesh.volume)
            else:
                print("  Mesh not watertight. Using convex hull for volume approximation.")
                metrics["volume"] = abs(mesh.convex_hull.volume)
                
            metrics["area"] = mesh.area
            
            # Sphericity calculation
            if metrics["area"] > 0:
                # Formula: Psi = (pi^(1/3) * (6V)^(2/3)) / A
                metrics["sphericity"] = (np.pi**(1/3) * (6 * metrics["volume"])**(2/3)) / metrics["area"]
            else:
                metrics["sphericity"] = 0
                
            # Cap sphericity at 1.0 (numerical errors can sometimes push it slightly above)
            metrics["sphericity"] = min(1.0, metrics["sphericity"])
                
            print(f"  Final Metrics: Volume={metrics['volume']:.5f}, Area={metrics['area']:.5f}, Sphericity={metrics['sphericity']:.5f}")
        except Exception as e:
            print(f"  Warning: Metrics calculation failed: {e}")
            metrics = {"volume": 0.0, "area": 0.0, "sphericity": 0.0}
            
        # 4. Generate Vision "Sheets" (Depth, Edges, Segments)
        try:
            print("Generating vision 'Sheets' (Depth, Edges, Segments)...")
            # Render a depth map from the front view (elevation 0, 1 view)
            render_out = self.model.render(scene_codes, n_views=1, elevation_deg=0.0)
            depth_map = render_out["depths"][0][0] # First scene, first view
            
            # Save depth map visualization
            self.vision_analyzer.process_depth_map(depth_map)
            
            # Save edge detection and segmentation from the debug input
            debug_input = "output/debug_segmented.png"
            if os.path.exists(debug_input):
                self.vision_analyzer.detect_edges(debug_input)
                self.vision_analyzer.segment_minerals(debug_input)
                
        except Exception as e:
            print(f"  Warning: Vision analysis failed: {e}")

        return metrics

    def run_batch(self, image_paths):
        for img in image_paths:
            out_name = "output/" + os.path.splitext(os.path.basename(img))[0] + ".obj"
            self.generate_3d(img, out_name)

if __name__ == "__main__":
    # Example usage
    rec = Reconstructor()
    # rec.generate_3d("data/raw/Granite/123.jpg")
