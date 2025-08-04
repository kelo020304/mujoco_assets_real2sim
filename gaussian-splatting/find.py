def find_nearest_to_identity(images_txt_path):
    import math
    closest_id = None
    closest_name = None
    min_error = float('inf')

    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) >= 10:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                pose_error = math.sqrt((qw - 1)**2 + qx**2 + qy**2 + qz**2 + tx**2 + ty**2 + tz**2)
                if pose_error < min_error:
                    min_error = pose_error
                    closest_id = parts[0]
                    closest_name = parts[9]

    print(f"✅ Closest to identity pose: ID={closest_id}, name={closest_name}, error={min_error:.6e}")
    return closest_id, closest_name

# 调用
find_nearest_to_identity("/home/jiziheng/Music/robot/gs_scene/gaussian-splatting/data/0422_2/distorted/sparse/0_txt/images.txt")
