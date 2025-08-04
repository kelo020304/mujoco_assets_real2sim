import numpy as np

# 示例：你需要替换成你实际的矩阵
# camera 在 colmap 下的变换矩阵（4x4）
T_camera_colmap = np.array([
    [0.99505101, -0.0894869 ,  0.04319234 ,-1.99674742],
    [0.07004411,  0.94000949 , 0.3338802,  -0.70101264],
    [-0.07047912, -0.32920246,  0.94162542 , 2.42273688],
    [0, 0, 0, 1]
])

# camera 在 board/world 下的变换矩阵（4x4）
T_camera_board = np.array([
    [0.0588151,  0.996553, 0.0585041 ,-0.110309],
    [ 0.582897, 0.0132935 ,-0.812437 ,-0.156028],
    [-0.810414, 0.0818855, -0.580106,   1.36021],
    [0,  0,  0, 1]
])

# 计算 T_world_colmap = inv(T_camera_board) @ T_camera_colmap
T_world_colmap = np.linalg.inv(T_camera_board) @ T_camera_colmap

print("===== ^world T_colmap（COLMAP 坐标系在棋盘/world坐标系下的变换）=====")
print(T_world_colmap)
