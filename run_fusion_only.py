from src.mesh_fusion import fuse_meshes
from pathlib import Path

out_dir = Path("output")
mesh_paths = [
    str(out_dir / "rock_model_view0.obj"),
    str(out_dir / "rock_model_view1.obj"),
    str(out_dir / "rock_model_view2.obj")
]

print("Running fusion only...")
fuse_meshes(mesh_paths, str(out_dir / "rock_model_fused.obj"))
print("Done.")
