import numpy as np
import os

def write_points3D_txt(filename, xyz, rgb, error=None, track_data=None):
    """
    写入符合 COLMAP points3D.txt 格式的文件，包含 XYZRGB 及默认 ERROR 和 TRACK 数据。
    """
    assert xyz.shape[0] == rgb.shape[0]
    N = xyz.shape[0]

    if error is None:
        error = -np.ones(N, dtype=np.float64)

    if track_data is None:
        # Dummy track (image_id=1, point2D_idx=0)，用于兼容 COLMAP 格式
        track_data = [[(-1, -1)] for _ in range(N)]

    with open(filename, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i in range(N):
            x, y, z = xyz[i]
            r, g, b = map(int, rgb[i])
            err = error[i]
            track_str = " ".join(f"{img} {pt2d}" for img, pt2d in track_data[i])
            f.write(f"{i} {x:.10f} {y:.10f} {z:.10f} {r} {g} {b} {err:.6f} {track_str}\n")

# ==== 主程序 ====

ply_path = "/home/jiziheng/Music/robot/SuGaR/test/bottle/vis_results/filtered_points.ply"
output_txt_path = "/home/jiziheng/Music/robot/SuGaR/test/bottle/vis_results/points3D.txt"

xyz_list = []
rgb_list = []

# 读取 PLY 文件
with open(ply_path, "r") as f:
    header_ended = False
    for line in f:
        if header_ended:
            parts = line.strip().split()
            if len(parts) >= 6:
                x, y, z = map(float, parts[0:3])
                r, g, b = map(int, parts[3:6])
                xyz_list.append([x, y, z])
                rgb_list.append([r, g, b])
        elif line.strip() == "end_header":
            header_ended = True

xyz_arr = np.array(xyz_list, dtype=np.float64)
rgb_arr = np.array(rgb_list, dtype=np.uint8)

# 写入 TXT 文件
write_points3D_txt(output_txt_path, xyz_arr, rgb_arr)

print(f"✅ 写入成功：{output_txt_path}，共 {len(xyz_arr)} 个点。")
