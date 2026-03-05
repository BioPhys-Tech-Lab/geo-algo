import open3d as o3d
import trimesh
import numpy as np
import copy

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, mesh_paths):
    pcds = []
    pcd_downs = []
    pcd_fpfhs = []
    for path in mesh_paths:
        mesh = o3d.io.read_triangle_mesh(path)
        pcd = mesh.sample_points_poisson_disk(number_of_points=5000)
        pcds.append(pcd)
        pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size)
        pcd_downs.append(pcd_down)
        pcd_fpfhs.append(pcd_fpfh)
    return pcds, pcd_downs, pcd_fpfhs

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def fuse_meshes(mesh_paths, output_path, voxel_size=0.05):
    print(f"Fusing {len(mesh_paths)} meshes with Color Preservation...")
    
    # Use Open3D for consistent color handling
    meshes = []
    pcds = []
    
    for p in mesh_paths:
        mesh = o3d.io.read_triangle_mesh(p)
        # Ensure normals
        mesh.compute_vertex_normals()
        
        # Center the mesh (simple alignment)
        center = mesh.get_center()
        mesh.translate(-center)
        
        # Sample points WITH colors
        # sample_points_poisson_disk preserves colors if present? yes usually.
        # sample_points_uniformly is faster and also preserves colors.
        try:
             # Try Poisson Disk first for better distribution
             # INCREASED to 100,000 for High Fidelity Color
             pcd = mesh.sample_points_poisson_disk(number_of_points=100000, init_factor=5)
        except:
             # Fallback
             pcd = mesh.sample_points_uniformly(number_of_points=100000)
             
        pcds.append(pcd)
        
    # Combine Point Clouds
    all_points = []
    all_colors = []
    all_normals = []
    
    for pcd in pcds:
        all_points.append(np.asarray(pcd.points))
        if pcd.has_colors():
            all_colors.append(np.asarray(pcd.colors))
        if pcd.has_normals():
            all_normals.append(np.asarray(pcd.normals))
            
    if not all_points:
        print("No points generated.")
        return False
        
    combined_points = np.vstack(all_points)
    
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined.points = o3d.utility.Vector3dVector(combined_points)
    
    if all_colors:
        combined_colors = np.vstack(all_colors)
        pcd_combined.colors = o3d.utility.Vector3dVector(combined_colors)
    else:
        print("Warning: Input meshes have no colors.")
        
    if all_normals:
         combined_normals = np.vstack(all_normals)
         pcd_combined.normals = o3d.utility.Vector3dVector(combined_normals)
    else:
         pcd_combined.estimate_normals()

    # --- ADVANCED REFINEMENT (Spatial Coherence & Overlapping Layers) ---
    print("Refining Point Cloud (Merging Overlaps & Removing Outliers)...")
    
    # 1. Voxel Downsampling to merge overlapping points from different views
    # This ensures "spatial coherence" by treating close points as one.
    # Use a small voxel size to maintain detail but merge exact duplicates/near-duplicates.
    pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.005) 
    
    # 2. Statistical Outlier Removal
    # This removes floating noise that causes "holes" or artifact islands.
    cl, ind = pcd_combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_combined = pcd_combined.select_by_index(ind)
    
    # Re-estimate normals after cleaning for better Poisson reconstruction
    pcd_combined.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # --------------------------------------------------------------------

    # Poisson Surface Reconstruction
    print("Running Poisson Surface Reconstruction (High Fidelity)...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        # Depth 10 = Higher resolution mesh (more vertices = sharper colors)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_combined, depth=10)
        
    # COLOR TRANSFER: Project colors from point cloud to new mesh vertices
    if pcd_combined.has_colors():
        print("Transferring colors to fused mesh...")
        # Build KDTree for nearest neighbor search
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_combined)
        
        vertices = np.asarray(mesh.vertices)
        vertex_colors = []
        
        for i in range(len(vertices)):
            # Find nearest point in the original colored cloud
            [k, idx, _] = pcd_tree.search_knn_vector_3d(vertices[i], 1)
            # Assign that color
            color = np.asarray(pcd_combined.colors)[idx[0]]
            vertex_colors.append(color)
            
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertex_colors))

    # Clean artifacts
    # Removing low density vertices can help remove "bubbles" and create cleaner edges.
    # We use a low quantile to remove only the very low density areas (outliers/noise).
    print("Trimming low-density mesh artifacts...")
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Fused mesh saved to {output_path}")
    return True
