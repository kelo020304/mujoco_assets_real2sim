import os
import sys
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

sys.path.append("utils")
sys.path.append("scene")

from scene.colmap_loader import read_points3D_binary
from utils.graphics_utils import *
from scene.gaussian_model import BasicPointCloud

def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element], text=True)  # ASCII 输出
    ply_data.write(path)

def bin_to_ply(bin_path, ply_path):
    print(f"Reading: {bin_path}")
    xyz, rgb, _ = read_points3D_binary(bin_path)
    print(f"Found {xyz.shape[0]} points.")
    print(f"Saving to: {ply_path}")
    storePly(ply_path, xyz, rgb)
    print("Done.")

if __name__ == "__main__":
    base = Path("data/0422_5/sparse/0")  # 你可以改成任何路径
    bin_file = base / "points3D.bin"
    ply_file = base / "points3D.ply"

    if not bin_file.exists():
        print("❌ points3D.bin 不存在")
        sys.exit(1)
    
    bin_to_ply(bin_file, ply_file)
