import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm


def show_segmentation_and_select_masks(image, masks):
    """
    显示分割结果图，允许用户点击 mask 区域进行选择，回车结束，返回合并后的 mask
    """
    combined_mask = np.zeros_like(masks[0]['segmentation'], dtype=bool)
    colored = image.copy()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 用颜色渲染所有 mask
    for idx, ann in enumerate(masks):
        mask = ann['segmentation']
        color = np.random.randint(0, 255, size=3)
        for c in range(3):
            colored[:, :, c] = np.where(mask, color[c], colored[:, :, c])
        # 标注编号
        y, x = np.where(mask)
        if len(x) > 0 and len(y) > 0:
            ax.text(x[0], y[0], str(idx), color='white', fontsize=10, weight='bold')

    ax.imshow(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    plt.title("Click on segments to select (Enter to finish)")
    selected_ids = set()

    def on_click(event):
        if event.inaxes != ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        for idx, ann in enumerate(masks):
            mask = ann["segmentation"]
            if mask[y, x]:
                nonlocal combined_mask
                combined_mask |= mask
                selected_ids.add(idx)
                ax.add_patch(Rectangle((x-5, y-5), 10, 10, edgecolor='lime', facecolor='none', lw=2))
                ax.text(x, y, "✓", color='lime', fontsize=12, fontweight='bold')
                print(f"[INFO] Selected mask {idx} at ({x},{y})")
                fig.canvas.draw()
                break

    def on_key(event):
        if event.key == 'enter':
            plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    print(f"[INFO] Selected {len(selected_ids)} masks.")
    return combined_mask


def apply_mask_to_all_images(image_dir, output_dir, mask_generator, reference_mask):
    rgba_dir = os.path.join(output_dir, "rgba")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(rgba_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    image_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    for fname in tqdm(image_list, desc="Processing images"):
        fpath = os.path.join(image_dir, fname)
        image = cv2.imread(fpath)
        masks = mask_generator.generate(image)

        best_iou = 0
        best_mask = None

        for ann in masks:
            mask = ann["segmentation"]
            iou = np.logical_and(reference_mask, mask).sum() / np.logical_or(reference_mask, mask).sum()
            if iou > best_iou:
                best_iou = iou
                best_mask = mask

        if best_mask is None:
            print(f"[WARN] No mask found for {fname}")
            continue

        # 保存 mask（灰度图 0-255）
        mask_path = os.path.join(masks_dir, fname)  # 与原图同名
        cv2.imwrite(mask_path, best_mask.astype(np.uint8) * 255)

        # 保存透明背景图像（RGBA）
        # rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        # rgba[:, :, 3] = best_mask.astype(np.uint8) * 255  # alpha 通道
        # rgba_path = os.path.join(rgba_dir, fname)  # 与原图同名
        # cv2.imwrite(rgba_path, rgba)
        # 保存白背景 RGB 图像
        white_bg_image = image.copy()
        white_bg_image[best_mask == 0] = [255, 255, 255]  # 将非物体区域设为白色

        white_path = os.path.join(rgba_dir, fname)  # 输出路径依然复用原来的
        cv2.imwrite(white_path, white_bg_image)



def run_full_pipeline():
    image_dir = "/home/jiziheng/Music/robot/SuGaR/test/bowl_green/images"
    output_dir = "/home/jiziheng/Music/robot/SuGaR/test/bowl_green/images_seg"
    sam_checkpoint = "./ckpts/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    image_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if len(image_list) == 0:
        print("No images found.")
        return

    first_image_path = os.path.join(image_dir, image_list[0])
    image = cv2.imread(first_image_path)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    print(f"Generating masks for first image: {image_list[0]}")
    masks = mask_generator.generate(image)

    reference_mask = show_segmentation_and_select_masks(image, masks)

    print("Now applying selected mask across all images...")
    apply_mask_to_all_images(image_dir, output_dir, mask_generator, reference_mask)
    print("✅ All done and ready for COLMAP.")


run_full_pipeline()
