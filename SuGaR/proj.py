import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 文件路径（请根据你的路径调整）
image_path = "/home/jiziheng/Music/robot/SuGaR/test/bottle/images/image_0001.png"
mask_path = "/home/jiziheng/Music/robot/SuGaR/test/bottle/input_ready/masks/image_0001.mask.png"
cameras_txt = "/home/jiziheng/Music/robot/SuGaR/test/bottle/sparse/0/cameras.txt"
images_txt = "/home/jiziheng/Music/robot/SuGaR/test/bottle/sparse/0/images.txt"
points3D_txt = "/home/jiziheng/Music/robot/SuGaR/test/bottle/sparse/0/points3D.txt"

# 解析 cameras.txt（获取相机内参）
def parse_cameras_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()
    intrinsics = {}
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            continue
        elems = line.strip().split()
        camera_id = int(elems[0])
        model = elems[1]
        width = int(elems[2])
        height = int(elems[3])
        params = list(map(float, elems[4:]))
        intrinsics[camera_id] = {
            "model": model,
            "width": width,
            "height": height,
            "params": params
        }
    return intrinsics

# 解析 images.txt（获取外参和相机id）
def parse_images_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()
    images = {}
    i = 0
    while i < len(lines):
        if lines[i].startswith("#") or lines[i].strip() == "":
            i += 1
            continue
        elems = lines[i].strip().split()
        image_id = int(elems[0])
        qw, qx, qy, qz = map(float, elems[1:5])
        tx, ty, tz = map(float, elems[5:8])
        camera_id = int(elems[8])
        image_name = elems[9]
        images[image_name] = {
            "qvec": np.array([qw, qx, qy, qz]),
            "tvec": np.array([tx, ty, tz]),
            "camera_id": camera_id
        }
        i += 2  # skip 2 lines per image entry
    return images

# 四元数转旋转矩阵
def qvec2rotmat(qvec):
    w, x, y, z = qvec
    R = np.array([
        [1 - 2 * y**2 - 2 * z**2,     2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w,       1 - 2 * x**2 - 2 * z**2,   2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w,       2 * y * z + 2 * x * w,     1 - 2 * x**2 - 2 * y**2]
    ])
    return R

# 解析点云 points3D.txt
def parse_points3D_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()
    points = []
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            continue
        elems = line.strip().split()
        xyz = np.array(list(map(float, elems[1:4])))
        rgb = np.array(list(map(int, elems[4:7])))
        points.append((xyz, rgb))
    return points

# 主执行流程
intrinsics = parse_cameras_txt(cameras_txt)
images = parse_images_txt(images_txt)
points = parse_points3D_txt(points3D_txt)

# 读取第一张图像名称
first_image_name = sorted(images.keys())[0]
cam_id = images[first_image_name]["camera_id"]
K_params = intrinsics[cam_id]["params"]

if intrinsics[cam_id]["model"] == "SIMPLE_PINHOLE":
    fx = fy = K_params[0]
    cx = K_params[1]
    cy = K_params[2]
elif intrinsics[cam_id]["model"] == "PINHOLE":
    fx, fy, cx, cy = K_params[:4]
else:
    raise ValueError("Unsupported camera model")

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# 外参
R = qvec2rotmat(images[first_image_name]["qvec"])
t = images[first_image_name]["tvec"].reshape(3, 1)
P = K @ np.hstack([R, t])

# 投影所有点
xyz_world = np.array([p[0] for p in points])
rgb = np.array([p[1] for p in points]) / 255.0
xyz_cam = (R @ xyz_world.T + t).T
xyz_proj = (K @ xyz_cam.T).T
uv = xyz_proj[:, :2] / xyz_proj[:, 2:3]

# 可视化投影点
image = np.array(Image.open(image_path))
mask = np.array(Image.open(mask_path).convert("L")) / 255.0

# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.scatter(uv[:, 0], uv[:, 1], s=1, c=rgb)
# plt.title("Projection of points3D onto first image")
# plt.axis("off")
# plt.show()

# plt.figure(figsize=(10, 10))
# plt.imshow(mask, cmap='gray')
# plt.title("Mask overlay for comparison")
# plt.axis("off")
# plt.show()

# 可视化投影点（原图 + mask 各一张）
# 计算整数投影坐标，并判断哪些点在图像范围内
uv_int = np.round(uv).astype(int)
H, W = mask.shape
valid = (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)

# 筛选在图像范围内的点
uv_valid = uv_int[valid]
rgb_valid = rgb[valid]
xyz_valid = xyz_world[valid]

# 根据 mask 值进一步筛选 mask 中白色区域（=1）内的点
mask_values = mask[uv_valid[:, 1], uv_valid[:, 0]] > 0.5  # 白色区域
xyz_masked = xyz_valid[mask_values]
rgb_masked = rgb_valid[mask_values]

# 可视化：3列图像（原图、mask、mask筛选后点云）
fig, axs = plt.subplots(1, 3, figsize=(21, 7))

# 原图投影
axs[0].imshow(image)
axs[0].scatter(uv[:, 0], uv[:, 1], s=1, c=rgb)
axs[0].set_title("Projected 3D Points on Image")
axs[0].axis("off")

# Mask投影
axs[1].imshow(mask, cmap='gray')
axs[1].scatter(uv[:, 0], uv[:, 1], s=1, c=rgb)
axs[1].set_title("Projected Points on Mask")
axs[1].axis("off")

# Mask过滤后有效点云（不投影，仅显示xyz）
axs[2].scatter(xyz_masked[:, 0], xyz_masked[:, 1], s=1, c=rgb_masked)
axs[2].set_title("3D Points in Mask Region (xy view)")
axs[2].axis("equal")

plt.tight_layout()
plt.show()


plt.tight_layout()
plt.show()

