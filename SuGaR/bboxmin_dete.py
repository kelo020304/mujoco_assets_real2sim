import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/chassis/sugar/point_cloud/iteration_7000/point_cloud.ply")
points = np.asarray(pcd.points)
print("min:", points.min(axis=0))
print("max:", points.max(axis=0))
