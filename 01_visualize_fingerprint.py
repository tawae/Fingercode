import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Đường dẫn tới dataset FVC2002 (DB1_B)
# Dùng thư mục của file script làm gốc để tránh lỗi không tìm thấy khi gọi từ ngoài thư mục
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "FVC2002", "DB1_B")
SAMPLE_IMAGE = "101_1.tif"

def visualize_fingerprint():
    img_path = os.path.join(DATASET_PATH, SAMPLE_IMAGE)
    
    if not os.path.exists(img_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {img_path}")
        print("Hãy đảm bảo bạn đang chạy script này từ thư mục Python_Implementation")
        return

    # 1. Đọc ảnh dưới dạng Grayscale (Đen trắng)
    # Cờ cv2.IMREAD_GRAYSCALE cực kỳ quan trọng
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Lỗi: Không đọc được ảnh. File có thể bị hỏng.")
        return

    print(f"Đã đọc ảnh thành công: {SAMPLE_IMAGE}")
    print(f"Kích thước ảnh: {img.shape} (Cao x Rộng)")
    print(f"Kiểu dữ liệu: {img.dtype} (uint8 = 0-255)")

    # 2. Vẽ biểu đồ Histogram (Phân bố độ sáng)
    # Giúp chúng ta thấy rõ sự phân biệt giữa Nền (Trắng) và Vân (Đen)
    plt.figure(figsize=(10, 5))

    # Hình 1: Ảnh gốc
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Ảnh Gốc (Grayscale)")
    plt.axis('off')

    # Hình 2: Histogram
    plt.subplot(1, 2, 2)
    plt.hist(img.ravel(), 256, [0, 256], color='black')
    plt.title("Histogram (Phân bố điểm ảnh)")
    plt.xlabel("Giá trị Pixel (0=Đen, 255=Trắng)")
    plt.ylabel("Số lượng Pixel")
    
    # Giải thích trên biểu đồ
    plt.text(20, 1000, "Ridge (Vân)\n(Vùng tối)", color='blue')
    plt.text(200, 1000, "Valley (Nền)\n(Vùng sáng)", color='red')

    plt.tight_layout()
    print("Đang lưu biểu đồ thành file histogram.png...")
    plt.savefig("histogram.png", dpi=300) # Thêm dòng này để lưu thành file ảnh
    plt.show()

if __name__ == "__main__":
    visualize_fingerprint()
