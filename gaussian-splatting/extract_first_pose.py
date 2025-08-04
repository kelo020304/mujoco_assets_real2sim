import numpy as np

def read_first_camera_pose_from_images_txt(images_txt_path):
    with open(images_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or len(line) == 0:
                continue
            parts = line.split()
            if len(parts) >= 10:
                image_name = parts[9]
                q = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])  # qw, qx, qy, qz
                t = np.array([float(parts[5]), float(parts[6]), float(parts[7])])  # tx, ty, tz
                return image_name, q, t
    raise RuntimeError("No valid image pose found in images.txt")

def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def pose_to_matrix(q, t):
    R = quaternion_to_rotation_matrix(q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

if __name__ == "__main__":
    images_txt_path = "/home/jiziheng/Music/robot/gs_scene/gaussian-splatting/data/0422_2/distorted/sparse/0_txt/images.txt"  # <- 修改为你自己的路径
    image_name, q, t = read_first_camera_pose_from_images_txt(images_txt_path)
    T_camera_colmap = pose_to_matrix(q, t)

    print(f"✅ First image: {image_name}")
    print("✅ Transformation ^camera T_colmap = ")
    print(T_camera_colmap)
