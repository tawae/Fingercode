"""
Bước 4: Frequency Estimation (Ước lượng tần số đường vân)
=========================================================
Module này cung cấp hai hàm lõi để đo khoảng cách ridge theo block và trả về
tần số dùng cho Fingercode/Gabor.
"""

import numpy as np
from scipy.ndimage import maximum_filter1d, rotate


def freqest(block, block_orient, wind_size=5, min_wave_length=5, max_wave_length=15):
    """
    Mục đích:
      Ước lượng tần số đường vân cho một block ảnh bằng cách xoay block theo
      hướng vân, chiếu thành tín hiệu 1D và đo khoảng cách giữa các đỉnh.

    Tham số:
      block: Block ảnh grayscale đã enhancement.
      block_orient: Block orientation cùng kích thước với `block`.
      wind_size: Kích thước cửa sổ tìm local maximum trên projection.
      min_wave_length: Bước sóng nhỏ nhất hợp lệ, đơn vị pixel.
      max_wave_length: Bước sóng lớn nhất hợp lệ, đơn vị pixel.

    Vì sao chọn tham số này:
      `wind_size=5` đủ rộng để gom một đỉnh thật thay vì nhiều đỉnh nhiễu.
      Dải `5-15` pixel là khoảng ridge spacing thường gặp ở ảnh vân tay 500dpi,
      nên loại được block nền hoặc block bị xoay/nhòe bất thường.

    Đầu ra:
      Một số float là frequency `1 / wavelength`, hoặc `0.0` nếu block không
      đủ đỉnh hay bước sóng nằm ngoài dải hợp lệ.

    Vì sao đầu ra như vậy mà không trả wavelength:
      Gabor filter dùng frequency trực tiếp. Trả `0.0` giúp các bước sau bỏ qua
      block không đáng tin thay vì phải xử lý exception hoặc `None`.
    """
    rows, _ = block.shape

    orient_doubled = 2 * block_orient.ravel()
    cos_orient = np.mean(np.cos(orient_doubled))
    sin_orient = np.mean(np.sin(orient_doubled))
    mean_orient = np.arctan2(sin_orient, cos_orient) / 2

    rotate_angle = np.degrees(mean_orient) + 90
    rotated = rotate(block, rotate_angle, reshape=False, order=1, mode="nearest")

    crop_size = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - crop_size) / 2))
    if crop_size < 3 or offset < 0 or offset + crop_size > rows:
        return 0.0

    cropped = rotated[offset:offset + crop_size, offset:offset + crop_size]
    if cropped.size == 0:
        return 0.0

    projection = np.sum(cropped, axis=0)
    dilation = maximum_filter1d(projection, size=wind_size)
    mean_proj = np.mean(projection)
    max_points = (dilation == projection) & (projection > mean_proj)
    max_indices = np.where(max_points)[0]

    if len(max_indices) < 2:
        return 0.0

    num_peaks = len(max_indices)
    wave_length = (max_indices[-1] - max_indices[0]) / (num_peaks - 1)

    if min_wave_length < wave_length < max_wave_length:
        return 1.0 / wave_length
    return 0.0


def ridge_frequency(img, mask, orient_img, block_size=32, wind_size=5,
                    min_wave_length=5, max_wave_length=15):
    """
    Mục đích:
      Tạo bản đồ tần số cho toàn ảnh bằng cách chia ảnh thành block và gọi
      `freqest` trên từng block.

    Tham số:
      img: Ảnh grayscale đã enhancement.
      mask: Mask vùng vân tay để bỏ nền.
      orient_img: Bản đồ hướng từ `estimate_orientation`.
      block_size: Kích thước block để ước lượng tần số.
      wind_size: Cửa sổ tìm peak truyền xuống `freqest`.
      min_wave_length: Bước sóng nhỏ nhất hợp lệ.
      max_wave_length: Bước sóng lớn nhất hợp lệ.

    Vì sao chọn tham số này:
      `block_size=32` cân bằng giữa đủ chu kỳ vân để đo bước sóng và đủ nhỏ để
      theo biến thiên cục bộ. Các tham số còn lại giữ nhất quán với `freqest`
      và tài liệu ridge frequency kinh điển.

    Đầu ra:
      Tuple `(freq_img, median_freq)`.

    Vì sao đầu ra như vậy mà không chỉ trả `freq_img`:
      `freq_img` giữ thông tin cục bộ, còn `median_freq` là tần số ổn định để
      dùng khi Gabor cần một giá trị đại diện, đặc biệt khi một số block bị lỗi.
    """
    rows, cols = img.shape
    freq_img = np.zeros_like(img, dtype=np.float64)

    for r in range(0, rows - block_size, block_size):
        for c in range(0, cols - block_size, block_size):
            blk_img = img[r:r + block_size, c:c + block_size].astype(np.float64)
            blk_orient = orient_img[r:r + block_size, c:c + block_size]
            freq = freqest(blk_img, blk_orient, wind_size,
                           min_wave_length, max_wave_length)
            freq_img[r:r + block_size, c:c + block_size] = freq

    freq_img = freq_img * mask
    valid_freqs = freq_img[freq_img > 0]
    if len(valid_freqs) > 0:
        median_freq = np.median(valid_freqs)
    else:
        median_freq = 1.0 / 9.0

    return freq_img, median_freq
