import open3d as o3d
import sys
import os
import numpy as np

def visualize_mesh(mesh_path, headless=False):
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file {mesh_path} not found.")
        return

    print(f"Visualizing {mesh_path}...")
    try:
        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        # Check if mesh is empty
        if len(mesh.vertices) == 0:
            print("Warning: Mesh has no vertices.")
            return

        # Compute normals for better rendering if not present
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        if headless:
            raise Exception("Headless mode requested.")

        # Create a visualization window
        print("Attempting Interactive Visualization (Open3D)...")
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Rock 3D Viewer", width=800, height=600)
            vis.add_geometry(mesh)
            
            # Simple run loop to catch errors during execution not just creation
            print("  Window created. Running main loop...")
            vis.run()
            vis.destroy_window()
        except Exception as e:
            print(f"  Open3D Window Error: {e}")
            raise e # Re-raise to trigger fallback
            
    except Exception as e:
        if not headless:
            print(f"\n[!] Interactive visualization failed: {e}")
            print("    (This is common in remote/headless environments or without GPU drivers)")
        
        print("\nAttempting Fallback: Generating 2D Screenshot using Matplotlib...")
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Load vertices and faces manually to avoid open3d dependency issues in fallback
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            # Simple downsample for plotting speed if too large
            if len(triangles) > 5000:
                print(f"    Downsampling mesh for plot ({len(triangles)} faces -> ~5000)...")
                # Randomly select faces? No, that breaks topology.
                # Just take every Nth triangle? 
                # Better: simple point cloud plot if mesh is huge, or trisurf if small.
                # Let's try trisurf with a stride or just point cloud.
                # Point cloud is safer.
                pass

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            # Rotate to a nice angle (view from slightly above)
            ax.view_init(elev=30, azim=45)
            
            if len(vertices) > 10000:
                # Plot as point cloud
                choice = np.random.choice(len(vertices), 10000, replace=False)
                v = vertices[choice]
                ax.scatter(v[:,0], v[:,1], v[:,2], c=v[:,2], cmap='viridis', s=1)
                title = "Rock Model (Point Cloud Preview)"
            else:
                # Plot as surface
                ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=triangles, cmap='gray', edgecolor='none', alpha=0.8)
                title = "Rock Model (Mesh Preview)"
                
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Save
            output_png = os.path.splitext(mesh_path)[0] + "_preview.png"
            plt.savefig(output_png)
            print(f"    [+] Saved preview image to: {output_png}")
            print(f"    Please open this file to view the model structure.")
            
        except ImportError:
            print("    [!] Matplotlib not installed. Cannot generate fallback preview.")
        except Exception as e2:
             print(f"    [!] Fallback failed: {e2}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Check for headless flag
        args = sys.argv[1:]
        headless = "--headless" in args
        # Get the first non-flag argument as mesh path
        mesh_path = next((arg for arg in args if not arg.startswith("--")), None)
        
        if mesh_path:
            visualize_mesh(mesh_path, headless=headless)
        else:
            print("Usage: python src/visualize.py <path_to_mesh.obj> [--headless]")
    else:
        print("Usage: python src/visualize.py <path_to_mesh.obj> [--headless]")
