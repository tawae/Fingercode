"""
Bước 2: Enhancement (Tăng cường ảnh vân tay)
=============================================
Mục tiêu: Làm rõ các đường vân (ridge) và giảm nhiễu trước khi đưa vào
           các bước xử lý tiếp theo (Orientation, Gabor, Thinning...).

Tác giả gốc (MATLAB) dùng FFT Enhancement (fft_enhance_cubs.m) - rất phức tạp.
Ở đây ta dùng các kỹ thuật tương đương trong OpenCV:
  1. CLAHE (Cân bằng histogram thích ứng cục bộ) ← Phương pháp chính
  2. Kết hợp: Chuẩn hóa + Tách nền + CLAHE → Ảnh tăng cường hoàn chỉnh

So sánh với tác giả gốc:
  - fft_enhance_cubs.m: Chia ảnh thành block → FFT → bandpass filter → IFFT
  - CLAHE: Chia ảnh thành block → cân bằng histogram cục bộ → nội suy ghép nối
  → Cả hai đều xử lý CỤC BỘ (theo block), mục đích giống nhau: tăng độ tương 
    phản cục bộ để đường vân rõ hơn ở mọi vùng trên ảnh.
"""

import cv2
import numpy as np

# ============================================================================
# HÀM TỪ BƯỚC TRƯỚC (Preprocessing)
# ============================================================================
def normalize_image(img):
    """
    Mục đích:
      Chuẩn hóa ảnh grayscale về mean=0 và std=1 để giảm ảnh hưởng của ảnh
      quá sáng, quá tối hoặc lệch tương phản trước khi tăng cường.

    Tham số:
      img: Ma trận ảnh grayscale 2D, thường là uint8 từ cv2.imread.

    Vì sao chọn tham số này:
      Pipeline luôn làm việc trên một ảnh truy vấn hoặc ảnh enrollment tại một
      thời điểm, nên truyền trực tiếp ma trận ảnh là đủ; không truyền path để
      hàm này chỉ tập trung vào xử lý số học, còn việc đọc file do tầng gọi lo.

    Đầu ra:
      Ma trận float32 đã chuẩn hóa theo công thức (pixel - mean) / std.

    Vì sao đầu ra như vậy mà không trả uint8:
      Kết quả chuẩn hóa có thể âm và có phần thập phân; giữ float giúp các bước
      thống kê phía sau chính xác hơn. Khi cần hiển thị hoặc dùng CLAHE, pipeline
      sẽ scale lại về 0-255 ở `full_enhancement_pipeline`.
    """
    img = img.astype(np.float32)
    mean = np.mean(img)
    std = np.std(img)
    if std == 0:
        std = 1
    return (img - mean) / std


def segment_fingerprint(img, block_size=16, threshold=0.1):
    """
    Mục đích:
      Tách vùng có vân tay khỏi nền bằng phương sai cục bộ theo block.
      Vùng vân tay có ridge/valley xen kẽ nên phương sai cao, nền trơn có
      phương sai thấp.

    Tham số:
      img: Ảnh grayscale 2D dùng để đo phương sai.
      block_size: Kích thước block vuông để gom thống kê cục bộ.
      threshold: Ngưỡng phương sai sau khi chuẩn hóa pixel về 0-1.

    Vì sao chọn tham số này:
      `block_size=16` đủ nhỏ để bám theo biên vùng vân tay nhưng không quá nhỏ
      đến mức nhiễu từng pixel làm mask vỡ vụn. `threshold` được để mở vì ảnh
      dataset khác nhau có nền và độ tương phản khác nhau; pipeline chính dùng
      giá trị thấp hơn (`0.005`) cho SOCOFing.

    Đầu ra:
      Mask uint8 cùng kích thước ảnh, giá trị 1 là vùng vân tay và 0 là nền.

    Vì sao đầu ra như vậy mà không trả ảnh đã cắt:
      Mask giữ nguyên hệ tọa độ của ảnh gốc, giúp orientation, frequency và
      Gabor áp dụng cùng kích thước mà không phải căn chỉnh lại tọa độ.
    """
    rows, cols = img.shape
    mask = np.zeros_like(img, dtype=np.uint8)

    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            block = img[r:min(r + block_size, rows), c:min(c + block_size, cols)]
            block_var = np.var(block / 255.0)
            if block_var > threshold:
                mask[r:min(r + block_size, rows), c:min(c + block_size, cols)] = 1

    return mask


def clahe_enhancement(img, clip_limit=2.0, grid_size=(8, 8)):
    """
    Mục đích:
      Tăng tương phản cục bộ bằng CLAHE để ridge/valley rõ hơn ở từng vùng ảnh.

    Tham số:
      img: Ảnh grayscale uint8 đã scale về 0-255.
      clip_limit: Giới hạn khuếch đại tương phản của mỗi tile.
      grid_size: Số tile theo chiều ngang/dọc mà CLAHE dùng để chia ảnh.

    Vì sao chọn tham số này:
      `clip_limit=2.0` là mức vừa phải, tránh khuếch đại nhiễu quá mạnh.
      `grid_size=(8, 8)` tạo các vùng đủ nhỏ để xử lý ánh sáng không đều nhưng
      vẫn đủ lớn để không sinh biên giả giữa các tile.

    Đầu ra:
      Ảnh uint8 sau CLAHE.

    Vì sao đầu ra như vậy mà không trả thêm histogram:
      Các bước sau chỉ cần ảnh đã tăng cường. Histogram toàn cục không được trả
      vì hệ thống dùng CLAHE cục bộ thay cho bước so sánh/demo.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)


def full_enhancement_pipeline(img, clip_limit=2.0, grid_size=(8, 8),
                               block_size=16, var_threshold=0.1):
    """
    Mục đích:
      Gom toàn bộ tiền xử lý đang dùng trong hệ thống: chuẩn hóa, scale lại,
      tách nền, CLAHE và áp mask nền trắng.

    Tham số:
      img: Ảnh grayscale 2D đầu vào.
      clip_limit: Mức giới hạn tương phản cho CLAHE.
      grid_size: Lưới tile cho CLAHE.
      block_size: Kích thước block dùng để tách nền bằng phương sai.
      var_threshold: Ngưỡng phương sai để quyết định block thuộc vùng vân tay.

    Vì sao chọn tham số này:
      Đây là các điểm tinh chỉnh ảnh hưởng trực tiếp đến chất lượng vector
      Fingercode. Cho phép truyền vào giúp các script demo/evaluation thử giá
      trị khác nhau, còn pipeline chính dùng cấu hình ổn định cho SOCOFing:
      `clip_limit=2.5`, `grid_size=(8,8)`, `block_size=16`, `var_threshold=0.005`.

    Đầu ra:
      Tuple `(enhanced_masked, mask, enhanced)`.

    Vì sao đầu ra như vậy mà không chỉ trả một ảnh:
      `enhanced_masked` đi vào orientation/frequency/Fingercode, `mask` giúp các
      bước sau bỏ nền, còn `enhanced` chưa áp mask hữu ích cho minh họa hoặc kiểm
      tra chất lượng CLAHE.
    """
    # Bước 1: Chuẩn hóa 
    norm_img = normalize_image(img)
    # Scale lại về 0-255 cho các bước tiếp
    norm_display = cv2.normalize(norm_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Bước 2: Tách nền (Segmentation)
    mask = segment_fingerprint(img, block_size=block_size, threshold=var_threshold)

    # Bước 3: CLAHE Enhancement
    enhanced = clahe_enhancement(norm_display, clip_limit=clip_limit, grid_size=grid_size)

    # Bước 4: Áp mask - chỉ giữ vùng vân tay, nền chuyển thành trắng (255)
    # Nền = trắng vì ở bước Nhị phân hóa sau này, nền trắng = Valley (thung lũng)
    enhanced_masked = np.where(mask == 1, enhanced, 255).astype(np.uint8)

    return enhanced_masked, mask, enhanced
