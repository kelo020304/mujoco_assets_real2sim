import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tqdm
import cv2
# ========== 解析工具函数 ==========

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
        i += 2
    return images

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

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    R = np.array([
        [1 - 2 * y**2 - 2 * z**2,     2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w,       1 - 2 * x**2 - 2 * z**2,   2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w,       2 * y * z + 2 * x * w,     1 - 2 * x**2 - 2 * y**2]
    ])
    return R

# ========== 主流程封装 ==========

def project_and_filter(
    image_dir, mask_dir, intrinsics, images, points, save_dir=None
):
    os.makedirs(save_dir, exist_ok=True)

    # 初始化原始点云
    xyz_all = np.array([p[0] for p in points])
    rgb_all = np.array([p[1] for p in points]) / 255.0

    # 初始化过滤后的点云（逐步过滤）
    filtered_xyz = xyz_all.copy()
    filtered_rgb = rgb_all.copy()

    # 遍历每张图像进行投影和mask筛选
    for idx, image_name in enumerate(tqdm.tqdm(sorted(images.keys()),desc="Processing images")):
        # if (max_views is not None) and (idx >= max_views):
        #     print("[warning]: exit")
        #     break

        # ==== 加载图像和mask ====
        image_path = os.path.join(image_dir, image_name)
        # mask_path = os.path.join(mask_dir, image_name.replace(".png", ".mask.png"))
        mask_path = os.path.join(mask_dir, image_name)
        if not os.path.exists(mask_path):
            print(f"[WARN] Mask not found for {image_name}")
            continue
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0

        # ==== 加载图像（RGB） ====
        # image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # if image_bgr is None:
        #     print(f"[WARN] Cannot read image {image_path}")
        #     continue
        # image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # for matplotlib

        # # ==== 加载 mask（灰度） ====
        # mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # if mask_gray is None:
        #     print(f"[WARN] Cannot read mask {mask_path}")
        #     continue
        # mask = mask_gray.astype(np.float32) / 255.0



        H, W = mask.shape

        # ==== 获取相机参数 ====
        cam_info = images[image_name]
        cam_id = cam_info["camera_id"]
        K_params = intrinsics[cam_id]["params"]

        # 内参矩阵
        if intrinsics[cam_id]["model"] == "SIMPLE_PINHOLE":
            fx = fy = K_params[0]
            cx = K_params[1]
            cy = K_params[2]
        elif intrinsics[cam_id]["model"] == "PINHOLE":
            fx, fy, cx, cy = K_params[:4]
        else:
            continue

        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])

        # ==== 获取外参 ====
        R = qvec2rotmat(cam_info["qvec"])
        t = cam_info["tvec"].reshape(3, 1)

        # ==== 投影所有点到图像平面 ====
        xyz_cam_all = (R @ xyz_all.T + t).T
        uv_all = (K @ xyz_cam_all.T).T
        uv_all = uv_all[:, :2] / uv_all[:, 2:3]

        # ==== 投影当前的 filtered 点 ====
        xyz_cam = (R @ filtered_xyz.T + t).T
        uv = (K @ xyz_cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        uv_int = np.round(uv).astype(int)

        # ==== 检查mask内哪些点是有效的 ====
        valid = (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
        uv_valid = uv_int[valid]
        mask_flags = mask[uv_valid[:, 1], uv_valid[:, 0]] > 0.5

        final_valid = np.zeros(len(filtered_xyz), dtype=bool)
        final_valid[np.where(valid)[0][mask_flags]] = True

        # ==== 更新过滤后的点云 ====
        filtered_xyz = filtered_xyz[final_valid]
        filtered_rgb = filtered_rgb[final_valid]

        # ==== 可视化 ====
        fig, axs = plt.subplots(1, 3, figsize=(21, 7))

        axs[0].imshow(image)
        axs[0].scatter(uv_all[:, 0], uv_all[:, 1], s=1, c=rgb_all)
        axs[0].set_title(f"All Projected Points: {image_name}")
        axs[0].axis("off")

        axs[1].imshow(mask, cmap="gray")
        axs[1].scatter(uv_all[:, 0], uv_all[:, 1], s=1, c=rgb_all)
        axs[1].set_title("Mask + Projection")
        axs[1].axis("off")

        axs[2].scatter(filtered_xyz[:, 0], -filtered_xyz[:, 1], s=1, c=filtered_rgb)
        axs[2].set_title("Filtered 3D Points (xy view)")
        axs[2].set_aspect("equal")
        axs[2].set_xlabel("X")
        axs[2].set_ylabel("-Y (flipped)")
        axs[2].grid(True)
        axs[2].axis("off")

        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{idx:03d}_{image_name}.png"))
        plt.close()


        # ==== 保存为 .ply ====
    if save_dir:
        ply_path = os.path.join(save_dir, "filtered_points.ply")
        with open(ply_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(filtered_xyz)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(filtered_xyz, filtered_rgb):
                r, g, b = (c * 255).astype(np.uint8)
                f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b}\n")

    # ==== 保存为 .bin ====
    # 格式为 float32，每个点 [x, y, z, r, g, b]
    if save_dir:
        bin_path = os.path.join(save_dir, "filtered_points.bin")
        bin_data = np.hstack([
            filtered_xyz.astype(np.float32),
            (filtered_rgb * 255).astype(np.uint8).astype(np.float32)
        ])
        bin_data.tofile(bin_path)

        # ==== 保存为 COLMAP 格式的 points3D.txt ====
        txt_path = os.path.join(save_dir, "points3D.txt")

        with open(txt_path, "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            for i, (p, c) in enumerate(zip(filtered_xyz, filtered_rgb)):
                r, g, b = (c * 255).astype(np.uint8)
                f.write(f"{i} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f} {r} {g} {b} -1 -1 -1\n")
    return filtered_xyz, filtered_rgb

def get_scene_paths(scene_root):
    return {
        "image_dir": os.path.join(scene_root, "images"),
        "mask_dir": os.path.join(scene_root, "images_seg", "masks"),
        "camera_txt": os.path.join(scene_root, "sparse", "0", "cameras.txt"),
        "images_txt": os.path.join(scene_root, "sparse", "0", "images.txt"),
        "points3D_txt": os.path.join(scene_root, "sparse", "0", "points3D.txt"),
        "save_dir": os.path.join(scene_root, "vis_results"),
    }



def main():
    scene_root = "/home/jiziheng/Music/robot/gs_scene/gs_hs/object_recon/raw_video/dianzuan"
    paths = get_scene_paths(scene_root)
    filtered_xyz, filtered_rgb = project_and_filter(
        image_dir=paths["image_dir"],
        mask_dir=paths["mask_dir"],
        intrinsics=parse_cameras_txt(paths["camera_txt"]),
        images=parse_images_txt(paths["images_txt"]),
        points=parse_points3D_txt(paths["points3D_txt"]),
        save_dir=paths["save_dir"],
    )
    
if __name__ =="__main__":
    main()


