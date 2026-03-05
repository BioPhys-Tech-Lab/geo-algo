import open3d as o3d
import sys
import os
import time

def capture_mesh(mesh_path, image_path):
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file {mesh_path} not found.")
        return

    print(f"Visualizing {mesh_path}...")
    try:
        # Load mesh
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        # Center the mesh
        center = mesh.get_center()
        mesh.translate(-center)

        vis = o3d.visualization.Visualizer()
        # On some systems visible=False might fail if no headless support, 
        # but we'll try standard window creation.
        vis.create_window(width=800, height=600, visible=True)
        vis.add_geometry(mesh)
        
        # Adjust view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.rotate(10.0, 0.0) # Slight rotation

        # Updates
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1) # Give it a moment to render
        
        vis.capture_screen_image(image_path, do_render=True)
        vis.destroy_window()
        print(f"Saved screenshot to {image_path}")
        
    except Exception as e:
        print(f"Failed to capture mesh: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        capture_mesh(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python src/capture_mesh.py <input_mesh.obj> <output_image.png>")
