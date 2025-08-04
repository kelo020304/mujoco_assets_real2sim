 python mask_v2.py   --colmap_dir   /home/jiziheng/Music/robot/SuGaR/test/bowl_green/sparse/0   --image_dir    /home/jiziheng/Music/robot/SuGaR/test/bowl_green/images   --ref_image    image_0001.png   --ref_mask     /home/jiziheng/Music/robot/SuGaR/test/bowl_green/images_seg/masks/image_0001.mask.png   --output_dir   /home/jiziheng/Music/robot/SuGaR/test/bowl_green/propagated   --sam_checkpoint ./ckpts/sam_vit_h_4b8939.pth   --model_type   vit_h   --device       cuda


 python mask_clip.py \
  --image_dir  /home/jiziheng/Music/robot/gs_scene/gs_hs/object_recon/raw_video/dianzuan/images \
  --output_dir  /home/jiziheng/Music/robot/gs_scene/gs_hs/object_recon/raw_video/dianzuan/images_seg \
  --sam_checkpoint ./ckpts/sam_vit_h_4b8939.pth \
  --model_type  vit_h \
  --device      cuda
