#!/usr/bin/env python3
import numpy as np
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
    raise RuntimeError("no valid pose in images.txt")

if __name__ == "__main__":
    images_txt = "data/crane/sparse/0/images.txt"  # 改成你的路径
    qw, qx, qy, qz, tx, ty, tz = parse_first_image_pose(images_txt)

    # 构造 T_wc
    rot = R.from_quat([qx, qy, qz, qw])  # scipy 格式是 [x,y,z,w]
    R0  = rot.as_matrix()
    T_wc = np.eye(4)
    T_wc[:3,:3] = R0
    T_wc[:3, 3] = [tx, ty, tz]

    # 逆变换
    T_cw = np.linalg.inv(T_wc)

    print("T_cw = inv(T_wc) =\n", T_cw)
    print("\nDeviation from identity (T_cw - I) =\n", T_cw - np.eye(4))
