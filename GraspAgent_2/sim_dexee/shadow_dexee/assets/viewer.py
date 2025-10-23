import trimesh
import glob
import os

# Get all .stl files in current directory
stl_files = glob.glob("*.stl")

if not stl_files:
    print("No STL files found in current directory.")
else:
    for stl_path in stl_files:
        print(f"Viewing: {os.path.basename(stl_path)}")
        mesh = trimesh.load(stl_path)
        mesh.show()  # Opens an interactive 3D viewer