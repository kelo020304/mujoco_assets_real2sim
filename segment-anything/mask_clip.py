#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def show_segmentation_and_select_masks(image, masks):
    """
    交互式在第一张图上选择 SAM 的分割区域，返回合并后的二值 mask (bool ndarray)
    """
    H, W = masks[0]['segmentation'].shape
    combined = np.zeros((H, W), dtype=bool)

    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for idx, ann in enumerate(masks):
        seg = ann['segmentation']
        color = np.random.rand(3)
        ax.contour(seg, colors=[color], linewidths=1)
        ys, xs = np.where(seg)
        if xs.size:
            ax.text(xs[0], ys[0], str(idx), color='white', fontsize=8)

    selected = set()
    def on_click(evt):
        if evt.inaxes!=ax: return
        xi, yi = int(evt.xdata), int(evt.ydata)
        for i, ann in enumerate(masks):
            if ann['segmentation'][yi, xi]:
                combined[:] |= ann['segmentation']
                selected.add(i)
                ax.scatter(xi, yi, s=50, facecolors='none', edgecolors='yellow')
                fig.canvas.draw()
                break

    def on_key(evt):
        if evt.key=='enter':
            plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.title("点击若干区域，按回车结束选择")
    plt.show()
    print(f"[INFO] 你选了这些 mask 索引：{sorted(selected)}")
    return combined

def extract_reference_feature(image, mask, clip_model, clip_proc, device):
    """
    用第一张图的合并 mask，生成 CLIP 参考特征向量
    """
    # 把 background 置为白
    img = image.copy()
    img[~mask] = 255
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inputs = clip_proc(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = clip_model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat  # [1, D]

def select_mask_by_clip(image, masks, ref_feat, clip_model, clip_proc, device):
    """
    对一张图的所有 SAM mask，用 CLIP 相似度挑选最贴近 first-mask 的那个
    """
    best_i, best_sim = -1, -1.0
    for i, ann in enumerate(masks):
        seg = ann['segmentation']
        img = image.copy()
        img[~seg] = 255
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        inputs = clip_proc(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = clip_model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sim = (ref_feat @ feat.T).item()
        if sim > best_sim:
            best_sim, best_i = sim, i
    return best_i

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image_dir',     required=True,  help="原始图像目录")
    p.add_argument('--output_dir',    required=True,  help="输出目录")
    p.add_argument('--sam_checkpoint',required=True,  help="SAM checkpoint 路径")
    p.add_argument('--model_type',    default='vit_h', help="SAM 模型类型")
    p.add_argument('--device',        default='cuda', help="torch device")
    args = p.parse_args()

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    masks_out = os.path.join(args.output_dir, 'masks')
    rgb_out   = os.path.join(args.output_dir, 'rgb_whitebg')
    os.makedirs(masks_out, exist_ok=True)
    os.makedirs(rgb_out,   exist_ok=True)

    # 1) 初始化 SAM
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # 2) 载入 CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 3) 列出所有图片，先对第一张做交互
    imgs = sorted([f for f in os.listdir(args.image_dir)
                   if f.lower().endswith(('.jpg','.png','.jpeg'))])
    first = imgs[0]
    im0 = cv2.imread(os.path.join(args.image_dir, first))
    print(f"[INFO] 先对第一张图交互选择：{first}")
    anns0 = mask_generator.generate(im0)
    ref_mask = show_segmentation_and_select_masks(im0, anns0)
    # 保存第一张 mask
    cv2.imwrite(os.path.join(masks_out, first), (ref_mask.astype(np.uint8)*255))
    white0 = im0.copy(); white0[~ref_mask] = 255
    cv2.imwrite(os.path.join(rgb_out, first), white0)

    # 4) 用第一张的 mask 生成参考特征
    ref_feat = extract_reference_feature(im0, ref_mask, clip_model, clip_proc, device)

    # 5) 迭代后续每张
    for fname in tqdm(imgs[1:], desc="Processing"):
        img = cv2.imread(os.path.join(args.image_dir, fname))
        anns = mask_generator.generate(img)
        if not anns:
            # 无分割候选，留空
            cv2.imwrite(os.path.join(masks_out, fname), np.zeros(img.shape[:2],dtype=np.uint8))
            cv2.imwrite(os.path.join(rgb_out, fname), np.ones_like(img)*255)
            continue
        sel = select_mask_by_clip(img, anns, ref_feat, clip_model, clip_proc, device)
        mask = anns[sel]['segmentation']
        # 保存
        cv2.imwrite(os.path.join(masks_out, fname), (mask.astype(np.uint8)*255))
        bg = img.copy(); bg[~mask] = 255
        cv2.imwrite(os.path.join(rgb_out, fname), bg)

    print("✅ 全部完成，结果保存在", args.output_dir)


if __name__ == '__main__':
    main()
