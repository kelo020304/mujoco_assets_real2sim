python train.py   -s /home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/chassis/sugar   -c /home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/chassis/sugar/   -i 7000   -r density  --project_mesh_on_surface_points True -f 15000  --export_ply True   --low_poly False   --high_poly True   --eval False   --gpu 0   --white_background True -bboxmin=-0.19146565,-0.18411618,-0.07752886   --bboxmax=0.17916323,0.17476723,0.20157814



```
python extract_mesh.py   -s /home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/chassis/sugar   -c /home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/chassis/sugar/   -i 7000  -o /home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/chassis/sugar --eval False   --gpu 0  


--bboxmin=-0.19146565,-0.18411618,-0.07752886   --bboxmax=0.17916323,0.17476723,0.20157814


obj2mjcf --obj-dir /home/jiziheng/Music/robot/SuGaR/output/refined_mesh/if_sugar --save-mjcf  --add-free-joint --decompose
obj2mjcf --obj-dir /home/jiziheng/Music/robot/SuGaR/test/yellow_box --save-mjcf  --add-free-joint --decompose
obj2mjcf --obj-dir /home/jiziheng/Music/robot/SuGaR/output/refined_mesh/if_4--save-mjcf  --add-free-joint --decompose
obj2mjcf --obj-dir /home/jiziheng/Music/robot/gs_scene/gs_hs/object_recon/gripper/obj --save-mjcf  --add-free-joint

ffmpeg -i 1.mp4 -q:v 1 -vf "fps=2" input/image_%04d.png


python train_full_pipeline.py -s /home/jiziheng/Music/robot/gs_scene/gs_hs/object_recon/dianzuan -r "density"  --high_poly True --export_obj True --postprocess_mesh True -l 0.1

python train_full_pipeline.py \
  -s /home/jiziheng/Music/robot/gs_scene/gs_hs/object_recon/gripper \
  -r "density" \
  -l 0.1 \
  --postprocess_mesh True \
  --export_ply True \
  --export_obj True


python recolmap.py \
  --source_dir /home/jiziheng/Music/robot/gs_scene/gs_hs/object_recon/raw_video/gripper \
  --dest_root  /home/jiziheng/Music/robot/gs_scene/gs_hs/object_recon \
  --object_name gripper

