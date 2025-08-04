import os
import shutil
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description="Repackage COLMAP outputs into a new project directory")
    parser.add_argument(
        "--source_dir", "-s",
        required=True,
        help="Path to the raw_video/<object> directory (contains images_seg, sparse, vis_results)"
    )
    parser.add_argument(
        "--dest_root", "-d",
        required=True,
        help="Root directory under which to create the new COLMAP project"
    )
    parser.add_argument(
        "--object_name", "-o",
        required=True,
        help="Name of the new COLMAP project directory (e.g. 'gripper')"
    )
    args = parser.parse_args()

    src = os.path.abspath(args.source_dir)
    dest = os.path.join(os.path.abspath(args.dest_root), args.object_name)

    images_src = os.path.join(src, "images_seg", "rgb_whitebg")
    cameras_src = os.path.join(src, "sparse", "0", "cameras.txt")
    images_txt_src = os.path.join(src, "sparse", "0", "images.txt")
    points3D_src = os.path.join(src, "vis_results", "points3D.txt")

    images_dst = os.path.join(dest, "images")
    sparse0_dst = os.path.join(dest, "sparse", "0")

    # Create destination directories
    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(sparse0_dst, exist_ok=True)

    # Copy images
    for fname in os.listdir(images_src):
        src_path = os.path.join(images_src, fname)
        dst_path = os.path.join(images_dst, fname)
        shutil.copy2(src_path, dst_path)
    print(f"Copied images from {images_src} to {images_dst}")

    # Copy COLMAP files
    shutil.copy2(cameras_src, os.path.join(sparse0_dst, "cameras.txt"))
    shutil.copy2(images_txt_src, os.path.join(sparse0_dst, "images.txt"))
    shutil.copy2(points3D_src, os.path.join(sparse0_dst, "points3D.txt"))
    print(f"Copied cameras.txt, images.txt, points3D.txt to {sparse0_dst}")

if __name__ == "__main__":
    main()

