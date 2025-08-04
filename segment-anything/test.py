import numpy as np
import torch

import matplotlib.pyplot as plt
import cv2
import os
import sys
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def process_image(image_path, output_path, predictor, input_box):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    # masked_image = np.zeros_like(image)
    masked_image = np.full_like(image, 255)
    predictor.set_image(image)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )


    mask = masks[0]  # shape: (H, W)
    

    # 创建 box 区域 mask
    box_mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = input_box
    box_mask[y1:y2, x1:x2] = 1

    preserve_region = (box_mask == 1) & (mask == 0)
    masked_image[preserve_region] = image[preserve_region]

    cv2.imwrite(output_path, masked_image)


    # generate mask
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(os.path.dirname(output_path), f"{base_name}.mask.png")
    mask_img = (mask * 255).astype(np.uint8)
    cv2.imwrite(mask_path, mask_img)




def main():
    input_dir = '/home/jiziheng/Music/robot/gs_scene/car_assets/arm_images/0408_3'
    output_dir = '/home/jiziheng/Music/robot/gs_scene/car_assets/arm_0408_3/images'
    os.makedirs(output_dir, exist_ok=True)
    sam_checkpoint = './ckpts/sam_vit_h_4b8939.pth' # 预训练模型地址
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # mask_generator = SamAutomaticMaskGenerator(sam)
    # image = cv2.imread('/home/jiziheng/Music/robot/gs_scene/video/whole_car_test/images/frame_0000.jpg')
    # predictor.set_image(image)
    input_box = np.array([0, 500, 700, 1120])
    image_list = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])
    
    print(f"found {len(image_list)} images")


    for fname in tqdm(image_list):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)
        process_image(input_path, output_path, predictor, input_box)

    print("all finish transfer")




if __name__ == "__main__":
    main()