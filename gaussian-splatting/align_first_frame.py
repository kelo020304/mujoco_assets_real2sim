#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import open3d as o3d
import argparse
from scipy.spatial.transform import Rotation as R

def parse_first_image_pose(images_txt_path):
    with open(images_txt_path, 'r') as f:
        for line in f:
            if line.strip() == "" or line.startswith('#'):
                continue
            parts = line.split()
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz    = map(float, parts[5:8])
            return qw, qx, qy, qz, tx, ty, tz
    raise RuntimeError(f"No pose found in {images_txt_path}")

def quaternion_from_rotation_matrix(R_mat):
    rot = R.from_matrix(R_mat)
    q = rot.as_quat()  # returns [x, y, z, w]
    return q[3], q[0], q[1], q[2]

def align_to_first_frame(sparse_dir, colmap_cmd="colmap"):
    # 1) BIN -> TXT
    subprocess.check_call(
        f"{colmap_cmd} model_converter "
        f"--output_type TXT "
        f"--input_path {sparse_dir} --output_path {sparse_dir}",
        shell=True
    )

    # 2) 读第一帧外参
    txt_images = os.path.join(sparse_dir, "images.txt")
    qw, qx, qy, qz, tx, ty, tz = parse_first_image_pose(txt_images)

    # 3) BIN -> PLY
    # subprocess.check_call(
    #     f"{colmap_cmd} model_converter "
    #     f"--input_type BIN --output_type PLY "
    #     f"--input_path {sparse_dir} --output_path {sparse_dir}",
    #     shell=True
    # )

    # 4) 读取并对齐点云
    ply_in  = os.path.join(sparse_dir, "points3D.ply")
    ply_out = os.path.join(sparse_dir, "points3D_aligned.ply")
    pcd = o3d.io.read_point_cloud(ply_in)

    # 构造 T_wc 和它的逆 T_cw
    R0   = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
    T_wc = np.eye(4);  T_wc[:3,:3] = R0;  T_wc[:3,3] = [tx, ty, tz]
    T_cw = np.linalg.inv(T_wc)

    pcd.transform(T_cw)
    o3d.io.write_point_cloud(ply_out, pcd)
    print(f"Aligned point cloud saved to {ply_out}")

    # 5) 对齐并重写 images.txt
    txt_out = os.path.join(sparse_dir, "images_aligned.txt")
    with open(txt_images, 'r') as fin, open(txt_out, 'w') as fout:
        for line in fin:
            if line.strip() == "" or line.startswith('#'):
                fout.write(line)
                continue
            parts = line.split()
            qw0, qx0, qy0, qz0 = map(float, parts[1:5])
            tx0, ty0, tz0    = map(float, parts[5:8])

            # 原 T_wc0
            Rm = o3d.geometry.get_rotation_matrix_from_quaternion([qw0, qx0, qy0, qz0])
            T0 = np.eye(4); T0[:3,:3] = Rm; T0[:3,3] = [tx0, ty0, tz0]
            T2 = T_cw @ T0

            R2 = T2[:3,:3]; t2 = T2[:3,3]
            qw2, qx2, qy2, qz2 = quaternion_from_rotation_matrix(R2)
            fout.write(
                f"{parts[0]} {qw2:.6f} {qx2:.6f} {qy2:.6f} {qz2:.6f} "
                f"{t2[0]:.6f} {t2[1]:.6f} {t2[2]:.6f} {' '.join(parts[8:])}\n"
            )

    # 6) TXT -> BIN（覆盖原有 bin 文件）
    subprocess.check_call(
        f"{colmap_cmd} model_converter "
        f"--output_type BIN "
        f"--input_path {sparse_dir} --output_path {sparse_dir}",
        shell=True
    )
    print("Camera poses aligned and written back to BIN format.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sparse_dir", required=True, help="Path to COLMAP sparse/0 folder")
    parser.add_argument("--colmap_cmd", default="colmap", help="COLMAP executable")
    args = parser.parse_args()
    align_to_first_frame(args.sparse_dir, args.colmap_cmd)
