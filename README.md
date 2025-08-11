# 🌍 Real-World to MuJoCo: Physical Asset Reconstruction Pipeline

> Transform real-world objects into high-quality MuJoCo physics assets through advanced 3D reconstruction techniques

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![COLMAP](https://img.shields.io/badge/COLMAP-3.6+-orange.svg)](https://colmap.github.io/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

<summary> Contents to be done </summary>

- [ ] [🎯Pipeline Overview](#-Pipeline-Overview)
- [ ] [✨ Features](#-features)
- [x] [🚀 Prerequisites](#-Prerequisites)
- [ ] [🎬 Step 1: Video Capture](#-step-1-video-capture)
- [ ] [🏗️ Step 2: COLMAP Initialization](#️-step-2-colmap-initialization)
- [ ] [🎨 Step 3: SAM+CLIP Segmentation](#-step-3-samclip-segmentation)
- [ ] [🔄 Step 4: COLMAP Data Reconstruction](#-step-4-colmap-data-reconstruction)
- [ ] [🍰 Step 5: SuGaR Mesh Reconstruction](#-step-5-sugar-mesh-reconstruction)
- [ ] [🎨 Step 6: Post-processing](#-step-6-post-processing)
- [ ] [📚 Examples](#-examples)
- [ ] [🛠️ Troubleshooting](#️-troubleshooting)
- [ ] [🤝 Contributing](#-contributing)
- [ ] [📄 License](#-license)



## 📋 Table of Contents

- [🎯Pipeline Overview](#-Pipeline-Overview)
- [✨ Features](#-features)
- [🚀 Prerequisites](#-Prerequisites)
- [🎬 Step 1: Video Capture](#-step-1-video-capture)
- [🏗️ Step 2: COLMAP Initialization](#️-step-2-colmap-initialization)
- [🎨 Step 3: SAM+CLIP Segmentation](#-step-3-samclip-segmentation)
- [🔄 Step 4: COLMAP Data Reconstruction](#-step-4-colmap-data-reconstruction)
- [🍰 Step 5: SuGaR Mesh Reconstruction](#-step-5-sugar-mesh-reconstruction)
- [🎨 Step 6: Post-processing](#-step-6-post-processing)
- [📚 Examples](#-examples)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 📊 Pipeline Overview



## ✨ Features

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
> to record a video of the object that needs **physical reconstruction**.  
> For best results, ensure good lighting and capture the object from multiple angles.

---

## 🏗️ Step 2: COLMAP Initialization

### 📝 Overview

### 📋 Prerequisites

#### System Requirements

#### Dependencies

### 🔧 Installation & Setup

#### COLMAP Installation

#### Environment Configuration

### 📷 Feature Extraction

#### Image Preprocessing

#### Feature Detection Parameters

#### Extraction Commands

### 🔗 Feature Matching

#### Matching Strategy

#### Parameter Tuning

#### Matching Commands

### 🏗️ Sparse Reconstruction

#### Bundle Adjustment

#### Camera Calibration

#### Point Cloud Generation

### 📊 Quality Validation

#### Reconstruction Metrics

#### Visual Inspection

#### Error Analysis

### 🔧 Processing Commands

### 🐛 Common Issues

### 📁 Output Structure

---

## 🎨 Step 3: SAM+CLIP Segmentation

### 📝 Overview

### 📋 Prerequisites

#### Model Requirements

#### Hardware Specifications

### 🔧 Installation & Setup

#### SAM Installation

#### CLIP Installation

#### Model Downloads

### 🤖 SAM Segmentation

#### Model Configuration

#### Mask Generation

#### Parameter Tuning

### 🎯 CLIP Classification

#### Object Detection

#### Category Filtering

#### Confidence Thresholds

### 🖼️ Mask Processing

#### Mask Refinement

#### Multi-frame Consistency

#### Quality Control

### 🔄 Batch Processing

#### Automated Pipeline

#### Parallel Processing

#### Memory Management

### 🔧 Processing Commands

### 🐛 Common Issues

### 📁 Output Structure

---

## 🔄 Step 4: COLMAP Data Reconstruction

### 📝 Overview

### 📋 Prerequisites

#### Input Requirements

#### Processing Environment

### 🔧 Data Preparation

#### Mask Integration

#### Image Filtering

#### Quality Assessment

### 🏗️ Dense Reconstruction

#### Stereo Reconstruction

#### Depth Map Generation

#### Point Cloud Densification

### 🎯 Point Cloud Processing

#### Filtering Techniques

#### Noise Removal

#### Density Optimization

### 📊 Quality Enhancement

#### Outlier Detection

#### Surface Smoothing

#### Completeness Check

### 🔧 Processing Commands

### 🐛 Common Issues

### 📁 Output Structure

---

## 🍰 Step 5: SuGaR Mesh Reconstruction

### 📝 Overview

### 📋 Prerequisites

#### System Requirements

#### GPU Specifications

### 🔧 Installation & Setup

#### SuGaR Installation

#### Environment Configuration

#### Model Preparation

### 🍰 Mesh Generation

#### Training Configuration

#### Parameter Optimization

#### Iteration Control

### 🎨 Mesh Refinement

#### Surface Smoothing

#### Topology Optimization

#### Detail Enhancement

### 📊 Quality Assessment

#### Mesh Metrics

#### Visual Inspection

#### Geometric Validation

### ⚡ Performance Optimization

#### Memory Management

#### GPU Utilization

#### Training Acceleration

### 🔧 Processing Commands

### 🐛 Common Issues

### 📁 Output Structure

---

## 🎨 Step 6: Post-processing

### 📝 Overview

### 📋 Prerequisites

#### Software Requirements

#### Input Validation

### 🔧 Mesh Processing

#### Cleaning Operations

#### Topology Repair

#### Decimation Strategies

### 📏 Scale Calibration

#### Reference Measurements

#### Unit Conversion

#### Validation Methods

### 🎨 Material Assignment

#### Physical Properties

#### Texture Mapping

#### Material Libraries

### ⚙️ MuJoCo Integration

#### XML Generation

#### Collision Geometry

#### Inertial Properties

### ✅ Final Validation

#### Physics Testing

#### Simulation Verification

#### Quality Metrics

### 🔧 Processing Commands

### 🐛 Common Issues

### 📁 Output Structure

---

## 📚 Examples

## 🛠️ Troubleshooting

## 🤝 Contributing

## 📄 License