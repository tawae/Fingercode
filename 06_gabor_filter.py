"""
Bước 5: Gabor kernel cho Fingercode
===================================
Pipeline hiện tại không tạo ảnh Gabor demo riêng; Fingercode chỉ cần kernel
Gabor ở nhiều hướng để lọc ảnh và lấy thống kê theo sector.
"""

import numpy as np


def create_gabor_filter(angle, frequency, kx=0.5, ky=0.5):
    """
    Mục đích:
      Tạo kernel Gabor 2D cho một hướng và một tần số ridge cụ thể.

    Tham số:
      angle: Hướng filter, đơn vị radian.
      frequency: Tần số đường vân, đơn vị 1/pixel.
      kx: Hệ số scale sigma theo trục x của kernel.
      ky: Hệ số scale sigma theo trục y của kernel.

    Vì sao chọn tham số này:
      Gabor cần đúng hướng và frequency để tăng ridge tương ứng. `kx=ky=0.5`
      bám theo cấu hình ridgefilter phổ biến, tạo kernel đủ gọn để chạy nhanh
      nhưng vẫn chọn lọc tần số/hướng.

    Đầu ra:
      Ma trận kernel Gabor 2D, hoặc `None` nếu frequency không hợp lệ.

    Vì sao đầu ra như vậy mà không trả ảnh đã lọc:
      Hàm này chỉ chịu trách nhiệm tạo kernel. Ảnh sau lọc phụ thuộc từng ảnh
      truy vấn/enrollment và được tạo trong `extract_fingercode`.
    """
    if frequency <= 0:
        return None

    sigma_x = 1.0 / frequency * kx
    sigma_y = 1.0 / frequency * ky

    sze = int(np.round(3 * max(sigma_x, sigma_y)))
    if sze < 1:
        sze = 1

    x, y = np.meshgrid(np.arange(-sze, sze + 1), np.arange(-sze, sze + 1))
    x_theta = x * np.cos(angle) + y * np.sin(angle)
    y_theta = -x * np.sin(angle) + y * np.cos(angle)

    gaussian = np.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))
    cosine = np.cos(2 * np.pi * frequency * x_theta)
    return gaussian * cosine
