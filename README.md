# 🌍 Real-World to MuJoCo: Physical Asset Reconstruction Pipeline

> Transform real-world objects into high-quality MuJoCo physics assets through advanced 3D reconstruction techniques

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![COLMAP](https://img.shields.io/badge/COLMAP-3.6+-orange.svg)](https://colmap.github.io/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

<summary> Contents to be done </summary>

- [ ] [🎯Pipeline Overview](#-Pipeline-Overview)
- [x] [🚀 Prerequisites](#-Prerequisites)
- [x] [🎬 Step 1: Video Capture](#-step-1-video-capture)
- [x] [🏗️ Step 2: COLMAP Initialization](#️-step-2-colmap-initialization)
- [x] [🎨 Step 3: SAM+CLIP Segmentation](#-step-3-samclip-segmentation)
- [x] [🔄 Step 4: COLMAP Data Reconstruction](#-step-4-colmap-data-reconstruction)
- [x] [🍰 Step 5: SuGaR Mesh Reconstruction](#-step-5-sugar-mesh-reconstruction)
- [x] [🎨 Step 6: Post-processing](#-step-6-post-processing)
- [ ] [📚 Blender Fix Process](#-examples)
- [ ] [🛠️ Troubleshooting](#️-troubleshooting)



## 📋 Table of Contents

- [🌍 Real-World to MuJoCo: Physical Asset Reconstruction Pipeline](#-real-world-to-mujoco-physical-asset-reconstruction-pipeline)
  - [📋 Table of Contents](#-table-of-contents)
  - [📊 Pipeline Overview](#-pipeline-overview)
  - [🚀 Prerequisites](#-prerequisites)
      - [🏗️ COLMAP](#️-colmap)
      - [🎯 Segment Anything Model (SAM)](#-segment-anything-model-sam)
      - [🍰SuGaR (Surface-Aligned Gaussian Splatting)](#sugar-surface-aligned-gaussian-splatting)
  - [🎬 Step 1: Video Capture](#-step-1-video-capture)
  - [🏗️ Step 2: COLMAP Initialization](#️-step-2-colmap-initialization)
  - [🎨 Step 3: SAM+CLIP Segmentation](#-step-3-samclip-segmentation)
  - [🔄 Step 4: COLMAP Data Reconstruction](#-step-4-colmap-data-reconstruction)
  - [🍰 Step 5: SuGaR Mesh Reconstruction](#-step-5-sugar-mesh-reconstruction)
  - [🎨 Step 6: Post-processing](#-step-6-post-processing)
  - [📚 Blender Fix Process](#-blender-fix-process)
  - [🛠️ Troubleshooting](#️-troubleshooting)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)



## 📊 Pipeline Overview

## 🚀 Prerequisites
This repo mainly rely on followin repos, please follow the installation request of repose listed below

---

####  🏗️ COLMAP

**Official Repository**: [colmap/colmap](https://github.com/colmap/colmap)

<summary><strong>🐧 Linux (Ubuntu/Debian)</strong></summary>

> [!WARNING]
> I am not sure installation below can work, please check colmap repo for precise installation


```bash
# Pre-built binaries
sudo apt update
sudo apt install colmap

# Or build from source
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build && cd build
cmake .. -GNinja
ninja
sudo ninja install
```


#### 🎯 Segment Anything Model (SAM)

**Official Repository**: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

<summary><strong>🐧 Linux (Ubuntu/Debian)</strong></summary>

> [!NOTE]
> You can create a conda env to install and The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`.

Install Segment Anything:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```bash
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```bash
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

> [!TIP]
> Please make sure to download a [model checkpoint](#model-checkpoints). If you want other checkpoints, please check SAM repo.

#### 🍰SuGaR (Surface-Aligned Gaussian Splatting)
**Official Repository**:[Anttwo/SuGaR](https://github.com/Anttwo/SuGaR)

<summary><strong>🐧 Linux (Ubuntu/Debian)</strong></summary>

> [!NOTE]
> You can use the same conda env with SAM while you need to make sure all dependencies are all compatiable, or you can just create a new conda env

Intstall SuGaR:

- Easy Installation :

    ```bash
    # HTTPS
    git clone https://github.com/Anttwo/SuGaR.git --recursive
    # or
    # SSH
    git clone git@github.com:Anttwo/SuGaR.git --recursive
    ```

    Then run the script
    ``` python
    python install.py
    ```
- Manual Installation
    ```bash
    # if you want to use a new conda env, you can just run:
    conda env create -f environment.yml
    conda activate sugar
    # If you want to use the sam env, you need to install packages in environment.yml manually
    ```
    ```bash
    # if you can not create env with environment.yml
    conda create --name sugar -y python=3.9
    conda activate sugar
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    conda install pytorch3d==0.7.4 -c pytorch3d
    conda install -c plotly plotly
    conda install -c conda-forge rich
    conda install -c conda-forge plyfile==0.8.1
    conda install -c conda-forge jupyterlab
    conda install -c conda-forge nodejs
    conda install -c conda-forge ipywidgets
    pip install open3d
    pip install --upgrade PyMCubes
    ```
    ```bash
    cd gaussian_splatting/submodules/diff-gaussian-rasterization/
    pip install -e .
    cd ../simple-knn/
    pip install -e .
    cd ../../../
    ```




---

## 🎬 Step 1: Video Capture

> **Please use your smartphone** 

to record a video of the object that needs **physical reconstruction**.  
For best results, ensure good lighting and capture the object from multiple angles.
> [!NOTE]
> - Do not change the focal length while recording
> - Try to capture multiple viewpoints of the object, but ensure the entire object is always visible in the video.

After recording the video, you need to first make a folder of your objects and make a 'input' folder in it and then run command below to form initial pictures

```bash
ffmpeg -i 1.mp4 -q:v 1 -vf "fps=2" input/image_%04d.png
```
The structure of the folder is like:
```
Objects  
└── input
```
---

## 🏗️ Step 2: COLMAP Initialization

> **This step uses the Python scripts from the original 3DGS repository.**  
> Please make sure you have **COLMAP** installed.

```bash
python convert.py -s Objects # make sure there is a input folder in your Objects folder
```
After running this command, you will get the initial COLMAP data.
Rearrange the folder structure as follows:
```
Objects   
├── input  
├── images   
├── sparse  
│      └── 0  
│        ├──cameras.bin  
│        ├──images.bin  
│        └──Points3D.bin  
└── distorted  
```
> [!NOTE]
> - images and sparse is the most important contents get from colmap
> - Please make sure that there are enough images in 'images'folder

Next, convert all .bin files to .txt format for easier reading and editing:

```bash
python read_write_model.py --input_model Objects/sparse/0 --output_model Objects/sparse/0 --input_format .bin --output_format .txt
```
After conversion, the folder structure will look like this:
```
Objects   
├── input  
├── images   
├── sparse  
│    └── 0   
│        ├──cameras.bin  
│        ├──images.bin  
│        ├──Points3D.bin  
│        ├──cameras.txt  
│        ├──images.txt  
│        └──Points3D.txt  
└── distorted  
```
And if you want, you can delete all .bin format files after conversion

---

## 🎨 Step 3: SAM+CLIP Segmentation
> **In this step, we use Segment Anything and CLIP to generate masks and corresponding images with a white background.**

```bash
python mask_clip.py \
  --image_dir  Objects/images \
  --output_dir  Objects/images_seg \
  --sam_checkpoint ./ckpts/sam_vit_h_4b8939.pth \
  --model_type  vit_h \
  --device      cuda
```
After running this command, you will get the following updated folder structure:

```bash
Objects   
├── input  
├── images   
├── sparse  
│    └── 0   
│        ├──cameras.bin  
│        ├──images.bin  
│        ├──Points3D.bin  
│        ├──cameras.txt  
│        ├──images.txt  
│        └──Points3D.txt 
├──images_seg
│  ├──masks  
│  └──rgb_whitebg  
├──vis_result
│  ├──validation_imgs...\
│  ├──Points3D.txt \
│  ├──filtered_points.ply \    
│  └──filtered_points.bin 
└── distorted  
```   n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
Some validation images will be generated for visualizing point cloud filtering.    
The **Points3D.txt** file is the most important output.
The **filtered_points.ply** file can be used for visualization in CloudCompare.  
```

---

## 🔄 Step 4: COLMAP Data Reconstruction
> **IN this step, rearrange the new colmap data structure**
```bash
python recolmap.py \
  --source_dir Object/ \
  --dest_root  /your/new/path/to/colmap/data/ \
  --object_name Your_asset_name
```
After running this command, you will obtain a new COLMAP dataset containing the filtered point clouds and white-background images, organized as follows:

```bash
Your_asset_name
├── sparse  
│    └── 0 
│        ├──cameras.txt  
│        ├──images.txt  
│        ├──Points3D.ply
│        └──Points3D.txt 
└── images  
```

---

## 🍰 Step 5: SuGaR Mesh Reconstruction
> **In this step, using SuGaR to reconstruct the mesh of asset**

```bash
python train_full_pipeline.py \
  -s Path/to/your/Your_asset_name/ \
  -r "density" \
  -l 0.1 \
  --postprocess_mesh True \
  --export_ply True \
  --export_obj True
  ```
> [!WARNING]
> - '-l' param only can be 0.1 0.3 0.5
> - '-r' param only can be 'density', 'dn_consistency', 'sdf' , But density achieve better results during my experiments
> '--postprocess_mesh' param is weather to postprocess mesh which will get smoother mesh, and you can change hyper param in the file **sugar_extractors/coarse_mesh.py**: `poisson_depth = 7`  , `vertices_density_quantile = 0`  
> - where **poisson_depth** is default to 10 but you can modify it to 6 or 7 if mesh has too mush holes and  **vertices_density_quantile** defalut to 0.1 but you can change it to 0 
> - More detailed params, you can check in file `train.py` and `train_full_pipeline.py`

Results will be in folder `SuGaR/output`
```bash
SuGaR/output 
├── coarse
├── coarse_mesh
├── refined
├── refined_mesh
│      ├──object1
│      └──object2...
├── refine_ply
└── vanilla_gs
```
> [!NOTE]
> The mesh of assets are in the folder `refined_mesh` with .mtl extension

Then move all files in `refined_mesh/your_object` to `Your_asset_name/obj`:
```bash
Your_asset_name
├── sparse  
│    └── 0 
│        ├──cameras.txt  
│        ├──images.txt  
│        ├──Points3D.ply
│        └──Points3D.txt 
├── obj
│   ├──Your_asset_name.mtl 
│   ├──Your_asset_name.obj  
│   └──Your_asset_name.png 
└── images
```

---

## 🎨 Step 6: Post-processing
> **In this step, process the reconstructed mesh into `.xml` extension for mujoco usage**

> [!WARNING]
> Please make sure `obj2mjcf` installed

```bash
python flip_obj/generate_mjcf.py --obj-dir Your_asset_name/obj/ --model-name Your_asset_name
```
Then you will get `Your_asset_name.xml` in `Your_asset_name/obj/Your_asset_name`

---

## 📚 Blender Fix Process

## 🛠️ Troubleshooting

## 🤝 Contributing

## 📄 License