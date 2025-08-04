import os
from PIL import Image
from tqdm import tqdm

# === 路径设置 ===
image_dir = "/home/jiziheng/Music/robot/SuGaR/test/bottle/images"      # 原始图像文件夹
mask_dir = "/home/jiziheng/Music/robot/SuGaR/test/bottle/input_ready/masks"  # mask 文件夹
output_dir = "/home/jiziheng/Music/robot/SuGaR/test/bottle/output_rgba"  # 输出 RGBA 图像

os.makedirs(output_dir, exist_ok=True)

# === 遍历图像文件夹 ===
for filename in tqdm(sorted(os.listdir(image_dir))):
    if not filename.endswith(".png"):
        continue

    # 构造文件路径
    image_path = os.path.join(image_dir, filename)
    base_name = filename.replace(".png", "")
    mask_name = f"{base_name}.mask.png"
    mask_path = os.path.join(mask_dir, mask_name)
    output_path = os.path.join(output_dir, filename)

    # 检查 mask 是否存在
    if not os.path.exists(mask_path):
        print(f"⚠️ 找不到 mask: {mask_path}")
        continue

    # 加载图像和 mask
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  # 灰度图 → alpha 通道

    # 构造 RGBA 图像
    rgba_image = image.copy()
    rgba_image.putalpha(mask)

    # 保存结果
    rgba_image.save(output_path, "PNG")

print(f"✅ 所有 RGBA 图像已保存到: {output_dir}")
