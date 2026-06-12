"""
Bước 3: Orientation Field Estimation (Ước lượng trường hướng vân)
=================================================================
Module này chỉ giữ phần API lõi dùng trong pipeline:
  - `compute_gradient`
  - `estimate_orientation`

Các hàm demo/vẽ hình đã được loại bỏ vì hệ thống thật chỉ cần dữ liệu số để
trích Fingercode, build DB, tìm kiếm top-5 và đánh giá.
"""

import cv2
import numpy as np


def compute_gradient(img, ksize=3):
    """
    Mục đích:
      Tính gradient ảnh theo trục X và Y bằng Sobel để phục vụ ước lượng hướng
      đường vân.

    Tham số:
      img: Ảnh grayscale, thường là ảnh đã enhancement.
      ksize: Kích thước kernel Sobel.

    Vì sao chọn tham số này:
      `img` là dữ liệu tối thiểu cần để tính đạo hàm. `ksize=3` giữ chi tiết
      ridge tốt và là lựa chọn gọn cho ảnh vân tay; kernel lớn hơn làm mượt hơn
      nhưng dễ xóa biên mảnh.

    Đầu ra:
      Tuple `(Gx, Gy)` là hai ma trận float64 cùng kích thước ảnh.

    Vì sao đầu ra như vậy mà không trả magnitude/góc:
      Orientation cần riêng Gx và Gy để tính covariance (`Gxx`, `Gxy`, `Gyy`).
      Magnitude/góc chỉ là dữ liệu phụ nếu cần debug, không cần trong pipeline.
    """
    img_float = img.astype(np.float64)
    Gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=ksize)
    Gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=ksize)
    return Gx, Gy


def estimate_orientation(img, gradient_sigma=1.0, block_sigma=3.0,
                         orient_smooth_sigma=3.0, sobel_ksize=3):
    """
    Mục đích:
      Ước lượng orientation field, tức góc hướng đường vân tại từng pixel.

    Tham số:
      img: Ảnh grayscale đã enhancement.
      gradient_sigma: Giữ để phản ánh tham số ridgeorient gốc; hiện Sobel đang
        đảm nhiệm gradient nên giá trị này không trực tiếp dùng trong code.
      block_sigma: Sigma Gaussian để làm mịn covariance gradient.
      orient_smooth_sigma: Sigma Gaussian để làm mịn sin/cos của góc kép.
      sobel_ksize: Kích thước kernel Sobel truyền xuống `compute_gradient`.

    Vì sao chọn tham số này:
      Bộ tham số bám theo `ridgeorient(normim, 1, 3, 3)` trong MATLAB: mịn vừa
      đủ để giảm nhiễu nhưng không làm mất thay đổi hướng quanh core/delta.
      `sobel_ksize=3` giữ chi tiết tốt cho ảnh SOCOFing.

    Đầu ra:
      Tuple `(orient_img, reliability)`:
      - `orient_img`: góc radian trong miền hướng không phân biệt 0 và pi.
      - `reliability`: độ tin cậy 0-1 của hướng tại từng pixel.

    Vì sao đầu ra như vậy mà không trả ảnh đã vẽ:
      Các bước frequency, core point và Fingercode cần dữ liệu số. Ảnh minh họa
      không tham gia DB/GUI/evaluation nên không được tạo trong module lõi này.
    """
    Gx, Gy = compute_gradient(img, ksize=sobel_ksize)

    Gxx = Gx ** 2
    Gxy = Gx * Gy
    Gyy = Gy ** 2

    sze = int(np.fix(6 * block_sigma))
    if sze % 2 == 0:
        sze += 1

    Gxx = cv2.GaussianBlur(Gxx, (sze, sze), block_sigma)
    Gxy = 2 * cv2.GaussianBlur(Gxy, (sze, sze), block_sigma)
    Gyy = cv2.GaussianBlur(Gyy, (sze, sze), block_sigma)

    denom = np.sqrt(Gxy ** 2 + (Gxx - Gyy) ** 2) + np.finfo(float).eps
    sin2theta = Gxy / denom
    cos2theta = (Gxx - Gyy) / denom

    sze = int(np.fix(6 * orient_smooth_sigma))
    if sze % 2 == 0:
        sze += 1

    sin2theta = cv2.GaussianBlur(sin2theta, (sze, sze), orient_smooth_sigma)
    cos2theta = cv2.GaussianBlur(cos2theta, (sze, sze), orient_smooth_sigma)

    orient_img = np.pi / 2 + np.arctan2(sin2theta, cos2theta) / 2

    Imin = (Gyy + Gxx) / 2 - (Gxx - Gyy) * cos2theta / 2 - Gxy * sin2theta / 2
    Imax = Gyy + Gxx - Imin
    reliability = 1 - Imin / (Imax + 0.001)
    reliability = reliability * (denom > 0.001)

    return orient_img, reliability
