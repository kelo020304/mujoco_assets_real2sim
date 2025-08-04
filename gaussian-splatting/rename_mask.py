import os

mask_dir = "/home/jiziheng/Music/robot/SuGaR/test/bottle/masks"
files = os.listdir(mask_dir)

for fname in files:
    if fname.endswith(".mask.png"):
        old_path = os.path.join(mask_dir, fname)
        new_name = fname.replace(".mask", ".png")
        new_path = os.path.join(mask_dir, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")
