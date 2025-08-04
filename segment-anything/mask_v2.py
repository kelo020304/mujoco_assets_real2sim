#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def parse_images_txt(path):
    """
    解析 COLMAP sparse/0/images.txt
    返回 dict: image_id -> {name, qvec, tvec, camera_id, xys, point3D_ids}
    """
    raw = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            raw.append(s)
    images = {}
    for i in range(0, len(raw), 2):
        meta = raw[i].split()
        pts2d = raw[i+1].split()
        image_id = int(meta[0])
        qvec = np.array(list(map(float, meta[1:5])))
        tvec = np.array(list(map(float, meta[5:8])))
        camera_id = int(meta[8])
        name = meta[9]
        xys, pids = [], []
        for j in range(0, len(pts2d), 3):
            x = float(pts2d[j]); y = float(pts2d[j+1]); pid = int(float(pts2d[j+2]))
            xys.append([x,y]); pids.append(pid)
        images[image_id] = {
            'name': name,
            'qvec': qvec,
            'tvec': tvec,
            'camera_id': camera_id,
            'xys': np.array(xys, dtype=np.float32),
            'point3D_ids': np.array(pids, dtype=int)
        }
    return images


def parse_points3D_txt(path):
    pts = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip()=='': continue
            elems = line.split()
            pid = int(elems[0])
            xyz = np.array(list(map(float, elems[1:4])))
            rgb = np.array(list(map(int, elems[4:7])), dtype=np.uint8)
            pts[pid] = {'xyz': xyz, 'rgb': rgb}
    return pts


def show_segmentation_and_select_masks(image, masks):
    """
    弹出窗口，让用户点击 SAM 分割结果选择目标 mask，回车确认
    返回合并后的二值 mask (bool ndarray)
    """
    combined = np.zeros_like(masks[0]['segmentation'], dtype=bool)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    colors = []
    for idx, ann in enumerate(masks):
        seg = ann['segmentation']
        color = np.random.rand(3)
        colors.append(color)
        ax.contour(seg, linewidths=1, colors=[color])
        y,x = np.where(seg)
        if x.size:
            ax.text(x[0], y[0], str(idx), color='white', fontsize=8)
    selected = set()
    def on_click(event):
        if event.inaxes!=ax: return
        xi, yi = int(event.xdata), int(event.ydata)
        for i, ann in enumerate(masks):
            if ann['segmentation'][yi, xi]:
                combined[:] |= ann['segmentation']
                selected.add(i)
                ax.scatter(xi, yi, s=50, facecolors='none', edgecolors='yellow')
                fig.canvas.draw(); break
    def on_key(event):
        if event.key=='enter': plt.close()
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.title('Click segments then press Enter')
    plt.show()
    print(f"[INFO] Selected masks: {sorted(selected)}")
    return combined


def collect_object_point_ids(images, ref_img_name, ref_mask_path):
    # 校验参考 mask 路径
    if not os.path.exists(ref_mask_path):
        alt = os.path.join(os.path.dirname(ref_mask_path), ref_img_name.replace('.png','.mask.png'))
        if os.path.exists(alt): ref_mask_path=alt
        else: raise FileNotFoundError(f"参考 mask 未找到: {ref_mask_path}")
    mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: raise IOError(f"无法读取参考 mask: {ref_mask_path}")
    h,w = mask.shape
    ref_id = next((iid for iid,d in images.items() if d['name']==ref_img_name), None)
    if ref_id is None: raise ValueError(f"{ref_img_name} 未在 images.txt 中找到")
    obj_pids=set()
    data=images[ref_id]
    for (x,y), pid in zip(data['xys'], data['point3D_ids']):
        if pid>=0 and 0<=int(round(x))<w and 0<=int(round(y))<h and mask[int(round(y)),int(round(x))]>127:
            obj_pids.add(pid)
    print(f"[INFO] Collected {len(obj_pids)} 3D points on the object.")
    return obj_pids


def project2D_and_score(masks, pts2d):
    best_i,best_s=-1,0.0
    for i,ann in enumerate(masks):
        seg=ann['segmentation']; H,W=seg.shape; cnt=0
        for x,y in pts2d:
            xi,yi=int(round(x)),int(round(y))
            if 0<=xi<W and 0<=yi<H and seg[yi,xi]: cnt+=1
        s=cnt/ (len(pts2d)+1e-8)
        if s>best_s: best_s, best_i = s,i
    return best_i,best_s


def propagate_masks(colmap_dir, image_dir, ref_img_name, ref_mask_path, output_dir,
                    sam_checkpoint, model_type='vit_h', device='cuda'):
    masks_out=os.path.join(output_dir,'masks'); rgb_out=os.path.join(output_dir,'rgb_whitebg')
    os.makedirs(masks_out,exist_ok=True); os.makedirs(rgb_out,exist_ok=True)
    images=parse_images_txt(os.path.join(colmap_dir,'images.txt'))
    points3D=parse_points3D_txt(os.path.join(colmap_dir,'points3D.txt'))
    obj_pids=collect_object_point_ids(images, ref_img_name, ref_mask_path)
    sam=sam_model_registry[model_type](checkpoint=sam_checkpoint); sam.to(device=device)
    maskgen=SamAutomaticMaskGenerator(sam)
    for fname in tqdm(sorted(os.listdir(image_dir))):
        if not fname.lower().endswith(('.png','.jpg','jpeg')): continue
        img=cv2.imread(os.path.join(image_dir,fname)); H,W=img.shape[:2]
        this=next((d for d in images.values() if d['name']==fname),None)
        if this is None: print(f"[WARN] {fname} not in images.txt, skip"); continue
        pts2d=[(x,y) for (x,y),pid in zip(this['xys'],this['point3D_ids']) if pid in obj_pids]
        if not pts2d:
            cv2.imwrite(os.path.join(masks_out,fname), np.zeros((H,W),dtype=np.uint8));
            cv2.imwrite(os.path.join(rgb_out,fname), np.ones_like(img)*255); continue
        anns=maskgen.generate(img)
        best_i,_=project2D_and_score(anns,pts2d)
        seg = anns[best_i]['segmentation'] if best_i>=0 else np.zeros((H,W),bool)
        mask8=(seg.astype(np.uint8)*255)
        cv2.imwrite(os.path.join(masks_out,fname), mask8)
        white=img.copy(); white[~seg]=(255,255,255)
        cv2.imwrite(os.path.join(rgb_out,fname),white)
    print("✅ Completed. Results in", output_dir)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--colmap_dir', required=True)
    p.add_argument('--image_dir', required=True)
    p.add_argument('--ref_image', required=True)
    p.add_argument('--ref_mask', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--sam_checkpoint', required=True)
    p.add_argument('--model_type', default='vit_h')
    p.add_argument('--device', default='cuda')
    args=p.parse_args()

    # 如果参考 mask 不存在，进行交互式选择
    if not os.path.exists(args.ref_mask):
        sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
        sam.to(device=args.device)
        maskgen = SamAutomaticMaskGenerator(sam)
        ref_img = cv2.imread(os.path.join(args.image_dir, args.ref_image))
        anns = maskgen.generate(ref_img)
        combined = show_segmentation_and_select_masks(ref_img, anns)
        os.makedirs(os.path.dirname(args.ref_mask), exist_ok=True)
        cv2.imwrite(args.ref_mask, (combined.astype(np.uint8)*255))
        print(f"[INFO] Saved interactive mask to {args.ref_mask}")

    propagate_masks(
        colmap_dir     = args.colmap_dir,
        image_dir      = args.image_dir,
        ref_img_name   = args.ref_image,
        ref_mask_path  = args.ref_mask,
        output_dir     = args.output_dir,
        sam_checkpoint = args.sam_checkpoint,
        model_type     = args.model_type,
        device         = args.device,
    )
