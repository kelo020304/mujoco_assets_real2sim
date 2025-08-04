#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# 1. 在这里修改为你的图片文件夹路径
image_dir = "/home/jiziheng/Music/robot/gs_scene/gaussian-splatting/data/floor_1_0421/input"
# 输出的 imagelist 文件
output_path = os.path.join(image_dir, "imagelist.txt")

# 2. 扫描所有文件，挑选 .jpg 或 .img 结尾的
images = []
for fn in os.listdir(image_dir):
    if not os.path.isfile(os.path.join(image_dir, fn)):
        continue
    ext = fn.lower().split('.')[-1]
    if ext in ("jpg", "png"):
        images.append(fn)

# 3. 排序（如果你的文件名都是 0001.jpg 这种零头，就用普通排序即可）
images.sort()

# 4. 写入 imagelist.txt
with open(output_path, "w") as f:
    for fn in images:
        f.write(fn + "\n")

print(f"Found {len(images)} images in '{image_dir}'.")
print(f"Wrote imagelist to '{output_path}'.")
