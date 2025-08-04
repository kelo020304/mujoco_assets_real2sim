import cv2
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def show_image_with_axis(image_path):
    # 读取并转换为 RGB
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 可视化，带坐标轴
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title("Image with Axis")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.grid(True)
    plt.axis('on')  # 显示坐标轴
    plt.savefig("image_with_axis.jpg")
    # plt.show()

# 调用
if __name__ == "__main__":
    image_path = "/home/jiziheng/Music/robot/gs_scene/car_assets/arm_images/0408_3/image_0000.jpg"
    show_image_with_axis(image_path)
