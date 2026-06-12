"""
Bước 11: Desktop UI bằng PyQt5
===================================================
Tính năng:
  - Giao diện kéo thả (Drag & Drop) ảnh đầu vào.
  - Xử lý trích xuất Fingercode + Query Faiss IVF dưới background thread (Loading state).
  - Trình bày 1 ảnh Query (trái) và 5 ảnh kết quả (phải).
  - Khả năng Click để bung lớn (Zoom lightbox) ảnh bất kỳ.
  - Nút quay lại màn hình chọn ảnh (Back).
"""

import sys
import os
import traceback
import cv2

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QStackedWidget, QGridLayout, QFrame, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont, QCursor

# ============================================================================
# IMPORT HỆ THỐNG BIOMETRIC (Từ Bước 08 và 10)
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from importlib.util import spec_from_file_location, module_from_spec

def _import_module(name, filepath):
    """
    Mục đích:
      Import module pipeline từ file path.

    Tham số:
      name: Tên module tạm.
      filepath: Đường dẫn tới file `.py`.

    Vì sao chọn tham số này:
      Các module đánh số (`08_`, `10_`) không import trực tiếp bằng cú pháp
      Python chuẩn, nên GUI dùng import động.

    Đầu ra:
      Module object đã nạp.

    Vì sao đầu ra như vậy mà không trả hàm/class riêng:
      GUI cần cả `extract_features` từ `08` và class DB từ `10`; trả module giúp
      truy cập rõ ràng theo namespace.
    """
    spec = spec_from_file_location(name, filepath)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

step08 = _import_module("s08", os.path.join(BASE_DIR, "08_fingercode_extraction.py"))
step10 = _import_module("s10", os.path.join(BASE_DIR, "10_database_system.py"))

import config
DB_PATH = config.DB_PATH

# Khởi tạo DB biến toàn cục để tránh load Faiss Index nhiều lần
global_db = None

# ============================================================================
# COMPONENT: Clickable QLabel hỗ trợ Zoom Overlay
# ============================================================================
class ClickableImageLabel(QLabel):
    """
    QLabel hiển thị ảnh có thể click để mở dialog zoom.
    """
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        """
        Mục đích:
          Khởi tạo label hiển thị ảnh có cursor click và trạng thái pixmap gốc.

        Tham số:
          parent: Widget cha theo chuẩn PyQt.

        Vì sao chọn tham số này:
          PyQt widget cần `parent` để quản lý vòng đời và layout; mặc định `None`
          giúp component dùng độc lập khi cần.

        Đầu ra:
          Không return; thiết lập thuộc tính và style cho label.

        Vì sao đầu ra như vậy mà không trả widget mới:
          Constructor của class chính là nơi tạo widget; PyQt kỳ vọng object được
          cấu hình trên `self`.
        """
        super().__init__(parent)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background-color: #f0f0f0; border: 1px solid #cccccc; border-radius: 8px;"
        )
        self.setMinimumSize(180, 220)

        self.image_path = None
        self.image_cv2 = None
        self.base_pixmap = None

    def mousePressEvent(self, event):
        """
        Mục đích:
          Bắt sự kiện click chuột trái để phát signal và mở dialog zoom nếu có ảnh.

        Tham số:
          event: Sự kiện chuột do Qt truyền vào.

        Vì sao chọn tham số này:
          Đây là signature bắt buộc khi override `QLabel.mousePressEvent`.

        Đầu ra:
          Không return; phát signal và có thể mở dialog.

        Vì sao đầu ra như vậy mà không trả trạng thái click:
          Event handler trong Qt điều khiển UI bằng side effect/signal, không dùng
          giá trị trả về.
        """
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            if self.base_pixmap is not None:
                self.show_zoom_dialog()

    def set_image(self, file_path=None, cv_img=None, size=(300, 400)):
        """
        Mục đích:
          Thiết lập ảnh hiển thị từ đường dẫn file hoặc ma trận ảnh OpenCV.

        Tham số:
          file_path: Đường dẫn ảnh trên đĩa, ưu tiên cho kết quả DB.
          cv_img: Ảnh grayscale dạng numpy array, dùng cho ảnh query vừa đọc.
          size: Kích thước tối đa để scale pixmap khi hiển thị.

        Vì sao chọn tham số này:
          Kết quả top-5 có sẵn đường dẫn ảnh, còn query image đã được trả về từ
          pipeline extraction. Hỗ trợ cả hai tránh đọc lại ảnh không cần thiết.

        Đầu ra:
          Không return; cập nhật pixmap trên label.

        Vì sao đầu ra như vậy mà không trả QPixmap:
          Component chịu trách nhiệm hiển thị; caller chỉ cần gọi method và để
          widget tự cập nhật trạng thái.
        """
        self.image_path = file_path
        self.image_cv2 = cv_img

        if file_path:
            # Dùng cv2 để load TIF rồi chuyển sang QPixmap cho tương thích mọi format
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                q_img = QImage(img.data.tobytes(), w, h, w, QImage.Format_Grayscale8)
                self.base_pixmap = QPixmap.fromImage(q_img)
            else:
                self.base_pixmap = QPixmap(file_path)

        elif cv_img is not None:
            # Ảnh Grayscale cv2 -> QImage -> QPixmap
            img_copy = cv_img.copy()  # copy để tránh dangling pointer
            h, w = img_copy.shape
            q_img = QImage(img_copy.data.tobytes(), w, h, w, QImage.Format_Grayscale8)
            self.base_pixmap = QPixmap.fromImage(q_img)

        if self.base_pixmap is not None and not self.base_pixmap.isNull():
            self.setPixmap(
                self.base_pixmap.scaled(
                    size[0], size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

    def show_zoom_dialog(self):
        """
        Mục đích:
          Mở dialog phóng to ảnh đang hiển thị.

        Tham số:
          Không có tham số; dùng `self.base_pixmap`.

        Vì sao chọn không truyền tham số:
          Ảnh zoom luôn là ảnh hiện tại của label, nên truyền thêm pixmap sẽ dễ
          lệch trạng thái.

        Đầu ra:
          Không return; mở modal dialog.

        Vì sao đầu ra như vậy mà không trả dialog:
          Dialog được chạy bằng `exec_()` và đóng ngay trong tương tác UI; caller
          không cần giữ tham chiếu.
        """
        dialog = QDialog(self.window())
        dialog.setWindowTitle("Hình Ảnh Phóng To (Zoom)")
        dialog.resize(600, 700)
        dialog.setStyleSheet("background-color: #ffffff;")

        layout = QVBoxLayout(dialog)
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)

        img_label.setPixmap(
            self.base_pixmap.scaled(
                560, 660, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        layout.addWidget(img_label)

        btn_close = QPushButton("Đóng")
        btn_close.setStyleSheet(
            "QPushButton { background-color: #1976D2; color: white; font-size: 14px;"
            " padding: 8px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1565C0; }"
        )
        btn_close.clicked.connect(dialog.close)
        layout.addWidget(btn_close)

        dialog.exec_()


# ============================================================================
# COMPONENT: Drag & Drop Zone Label
# ============================================================================
class DropAreaLabel(QLabel):
    """
    QLabel đóng vai trò vùng kéo-thả file ảnh.
    """
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        """
        Mục đích:
          Khởi tạo vùng kéo-thả và text hướng dẫn chọn ảnh.

        Tham số:
          parent: Widget cha theo chuẩn PyQt.

        Vì sao chọn tham số này:
          Giữ đúng contract của Qt widget và cho phép gắn component vào layout
          bất kỳ.

        Đầu ra:
          Không return; cấu hình style, text và acceptDrops trên `self`.

        Vì sao đầu ra như vậy mà không trả file path:
          File path chỉ có sau sự kiện drop/click, còn constructor chỉ chuẩn bị
          giao diện.
        """
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText(
            "\n\nKéo & Thả ảnh vân tay (TIF, PNG, JPG) vào đây\n"
            "hoặc Click vào nút bên dưới (Select File)\n\n"
        )
        self._set_default_style()
        self.setAcceptDrops(True)

    def _set_default_style(self):
        """
        Mục đích:
          Đặt lại style mặc định của vùng kéo-thả.

        Tham số:
          Không có tham số; style áp dụng trực tiếp lên label hiện tại.

        Vì sao chọn không truyền tham số:
          Style mặc định là một phần cố định của component, không cần caller cấu
          hình ngoài.

        Đầu ra:
          Không return; cập nhật stylesheet.

        Vì sao đầu ra như vậy mà không trả chuỗi CSS:
          PyQt cần side effect trên widget; trả CSS sẽ buộc caller lặp lại thao
          tác setStyleSheet.
        """
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaaaaa;
                border-radius: 12px;
                background-color: #fafafa;
                color: #555555;
                font-size: 16px;
                padding: 30px;
            }
        """)

    def dragEnterEvent(self, event):
        """
        Mục đích:
          Chấp nhận thao tác kéo file vào vùng drop và đổi style báo hiệu.

        Tham số:
          event: Drag event do Qt truyền vào.

        Vì sao chọn tham số này:
          Đây là signature bắt buộc của Qt để kiểm tra mime data và accept/ignore.

        Đầu ra:
          Không return; gọi `event.accept()` hoặc `event.ignore()`.

        Vì sao đầu ra như vậy mà không trả bool:
          Qt đọc trạng thái accept/ignore từ event object, không từ return value.
        """
        if event.mimeData().hasUrls():
            self.setStyleSheet("""
                QLabel {
                    border: 3px dashed #4CAF50;
                    border-radius: 12px;
                    background-color: #e8f5e9;
                    color: #2e7d32;
                    font-size: 16px;
                    padding: 30px;
                }
            """)
            event.accept()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """
        Mục đích:
          Khôi phục style mặc định khi con trỏ kéo file rời khỏi vùng drop.

        Tham số:
          event: Drag leave event do Qt truyền vào.

        Vì sao chọn tham số này:
          Signature của Qt yêu cầu nhận event, dù logic hiện tại không cần đọc
          nội dung event.

        Đầu ra:
          Không return; cập nhật style.

        Vì sao đầu ra như vậy mà không trả trạng thái:
          Đây là phản hồi UI tức thời, side effect trên widget là kết quả cần
          thiết.
        """
        self._set_default_style()

    def dropEvent(self, event):
        """
        Mục đích:
          Nhận file đầu tiên được thả vào vùng drop và phát signal đường dẫn.

        Tham số:
          event: Drop event chứa danh sách URL/file.

        Vì sao chọn tham số này:
          Qt cung cấp file qua `event.mimeData().urls()`, nên handler phải nhận
          event để bóc tách đường dẫn local.

        Đầu ra:
          Không return; emit `file_dropped` với file path.

        Vì sao đầu ra như vậy mà không gọi xử lý trực tiếp:
          Emit signal tách component UI khỏi logic xử lý ảnh, giúp page chính tự
          quyết định handler.
        """
        self._set_default_style()
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.file_dropped.emit(file_path)


# ============================================================================
# THREAD: Xử lý Biometrics Background
# ============================================================================
class ProcessThread(QThread):
    """
    Worker thread chạy extraction và FAISS search để UI không bị treo.
    """
    finished = pyqtSignal(object, object)  # list_top5, query_cv_img
    error = pyqtSignal(str)

    def __init__(self, img_path):
        """
        Mục đích:
          Khởi tạo worker xử lý một ảnh truy vấn.

        Tham số:
          img_path: Đường dẫn ảnh do người dùng chọn/kéo thả.

        Vì sao chọn tham số này:
          Worker cần tự đọc ảnh trong thread nền để không chặn UI thread.

        Đầu ra:
          Không return; lưu `img_path` vào instance.

        Vì sao đầu ra như vậy mà không xử lý ngay:
          Qt yêu cầu logic chạy nền nằm trong `run`, được kích hoạt bằng
          `thread.start()`.
        """
        super().__init__()
        self.img_path = img_path

    def run(self):
        """
        Mục đích:
          Trích Fingercode từ ảnh query và tìm top-5 trong FAISS.

        Tham số:
          Không có tham số; dùng `self.img_path` đã lưu ở constructor.

        Vì sao chọn không truyền tham số:
          `run` là method chuẩn của `QThread`, không nên đổi signature.

        Đầu ra:
          Không return; emit `finished(top_k_results, q_img)` hoặc `error(str)`.

        Vì sao đầu ra như vậy mà không trả list kết quả:
          Thread nền giao tiếp với UI qua signal để cập nhật an toàn trên main
          thread.
        """
        try:
            # 1. Rút Vector
            q_vec, q_img = step08.extract_features(self.img_path)
            if q_vec is None:
                self.error.emit("Không thể rút trích đặc trưng từ file ảnh.\nKiểm tra file ảnh có đúng định dạng vân tay không.")
                return

            # 2. Tìm kiếm FAISS
            top_k_results = global_db.search_top_k(q_vec, k=5)
            self.finished.emit(top_k_results, q_img)

        except Exception as e:
            # In đầy đủ traceback ra terminal để debug
            tb = traceback.format_exc()
            print("[ERROR] ProcessThread exception:")
            print(tb)
            self.error.emit(f"{type(e).__name__}: {str(e)}")


# ============================================================================
# MAIN WINDOW APP
# ============================================================================
class FingerprintApp(QMainWindow):
    """
    Cửa sổ chính của ứng dụng tìm kiếm vân tay top-5.
    """

    def __init__(self):
        """
        Mục đích:
          Khởi tạo main window, mở DB/FAISS và dựng ba page import/loading/result.

        Tham số:
          Không có tham số; cấu hình giao diện dùng hằng số trong class/module.

        Vì sao chọn không truyền tham số:
          GUI là entry point cuối, chỉ cần cấu hình DB từ `config.py`; người dùng
          tương tác qua UI thay vì tham số hàm.

        Đầu ra:
          Không return; tạo toàn bộ widget trên `self`.

        Vì sao đầu ra như vậy mà không trả QApplication/window khác:
          PyQt xây dựng UI bằng side effect trên instance widget; object window
          được tạo bởi constructor.
        """
        super().__init__()
        self.setWindowTitle("Fingerprint Recognition System")
        self.resize(1100, 750)

        # Light-mode toàn cục
        self.setStyleSheet("background-color: #ffffff; color: #333333;")

        self.initDB()

        # Central widget config
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # QStackedWidget để chuyển Page
        self.main_layout = QVBoxLayout(self.central_widget)
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # Xây dựng các Pages
        self.build_import_page()
        self.build_loading_page()
        self.build_result_page()

        self.stacked_widget.setCurrentIndex(0)

    def initDB(self):
        """
        Mục đích:
          Khởi tạo kết nối global tới SQLite và FAISS index cho phiên GUI.

        Tham số:
          Không có tham số; dùng `DB_PATH` từ config.

        Vì sao chọn không truyền tham số:
          GUI luôn tìm trong DB đã build bởi `10_database_system.py`, nên path
          cấu hình tập trung là đủ.

        Đầu ra:
          Không return; gán `global_db`.

        Vì sao đầu ra như vậy mà không trả DB object:
          Worker thread cần truy cập cùng DB object; biến global hiện là cách đơn
          giản để tránh load FAISS index lại nhiều lần.
        """
        global global_db
        global_db = step10.FingerprintVectorDB(DB_PATH)
        global_db.connect()

    # --- PAGE 1: IMPORT & DRAG-DROP ---
    def build_import_page(self):
        """
        Mục đích:
          Dựng page đầu vào cho phép kéo-thả hoặc chọn file ảnh.

        Tham số:
          Không có tham số; dùng `self.stacked_widget` và tạo widget con.

        Vì sao chọn không truyền tham số:
          Page là một phần cố định của main window, không cần cấu hình ngoài.

        Đầu ra:
          Không return; thêm page vào stacked widget.

        Vì sao đầu ra như vậy mà không trả page:
          Method trực tiếp gắn page vào UI hiện tại, tránh caller quên addWidget.
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 30, 40, 30)

        title = QLabel("TÌM KIẾM VÂN TAY GẦN NHẤT")
        title.setFont(QFont("Arial", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("margin-bottom: 15px; color: #1976D2;")
        layout.addWidget(title)

        subtitle = QLabel("Import ảnh vân tay để tìm 5 ảnh giống nhất trong cơ sở dữ liệu")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 14px; color: #777777; margin-bottom: 10px;")
        layout.addWidget(subtitle)

        # Drop Area
        self.drop_area = DropAreaLabel()
        self.drop_area.file_dropped.connect(self.start_processing)
        layout.addWidget(self.drop_area, stretch=1)

        # Select File button
        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("📂  Chọn File Ảnh")
        self.btn_select.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                font-size: 16px;
                padding: 12px 30px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
        """)
        self.btn_select.clicked.connect(self.browse_file)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_select)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)
        self.stacked_widget.addWidget(page)  # Index 0

    def browse_file(self):
        """
        Mục đích:
          Mở file dialog để người dùng chọn ảnh truy vấn.

        Tham số:
          Không có tham số; dialog dùng main window làm parent.

        Vì sao chọn không truyền tham số:
          Người dùng chọn file tương tác qua UI; filter ảnh và thư mục mặc định
          nằm ngay trong method.

        Đầu ra:
          Không return; nếu chọn file thì gọi `start_processing(file_path)`.

        Vì sao đầu ra như vậy mà không trả path:
          Trong GUI, hành động chọn file nên kích hoạt xử lý ngay để giữ workflow
          liền mạch.
        """
        options = QFileDialog.Options()
        # Mở từ thư mục Home để duyệt được tất cả file trên máy
        home_dir = os.path.expanduser("~")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn một ảnh vân tay",
            home_dir,
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)",
            options=options,
        )
        if file_path:
            self.start_processing(file_path)

    # --- PAGE 2: LOADING / ERROR ---
    def build_loading_page(self):
        """
        Mục đích:
          Dựng page loading/error trong lúc thread xử lý ảnh.

        Tham số:
          Không có tham số; tạo widget con trên main window.

        Vì sao chọn không truyền tham số:
          Nội dung loading là trạng thái cố định, chỉ text màu thay đổi khi có
          lỗi hoặc lần chạy mới.

        Đầu ra:
          Không return; thêm page vào stacked widget.

        Vì sao đầu ra như vậy mà không trả page:
          Method này là một bước dựng UI nội bộ, không phải factory dùng lại.
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 40)

        layout.addStretch()

        self.loading_label = QLabel("⏳ Đang trích xuất đặc trưng và truy vấn FAISS IVF...\nVui lòng chờ.")
        self.loading_label.setFont(QFont("Arial", 16))
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setWordWrap(True)
        self.loading_label.setStyleSheet("color: #E65100; font-weight: bold;")
        layout.addWidget(self.loading_label)

        layout.addStretch()

        # Nút Back luôn hiển thị trên loading page (đặc biệt hữu ích khi có lỗi)
        btn_row = QHBoxLayout()
        self.btn_back_loading = QPushButton("← Quay Lại Import")
        self.btn_back_loading.setStyleSheet("""
            QPushButton {
                background-color: #757575; color: white;
                font-size: 14px; padding: 9px 22px; border-radius: 6px;
            }
            QPushButton:hover { background-color: #616161; }
        """)
        self.btn_back_loading.clicked.connect(self._back_from_loading)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_back_loading)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.stacked_widget.addWidget(page)  # Index 1

    def _back_from_loading(self):
        """
        Mục đích:
          Quay về page import và dừng worker nếu người dùng rời loading page.

        Tham số:
          Không có tham số; kiểm tra `self.thread` nếu tồn tại.

        Vì sao chọn không truyền tham số:
          Method được gắn trực tiếp vào nút Back, trạng thái thread nằm trong
          main window.

        Đầu ra:
          Không return; terminate/wait thread nếu cần và chuyển page.

        Vì sao đầu ra như vậy mà không trả trạng thái hủy:
          UI chỉ cần phản hồi ngay. Nếu cần xử lý hủy tinh vi hơn, có thể thay
          terminate bằng cơ chế cancellation riêng.
        """
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.terminate()
            self.thread.wait()
        self.stacked_widget.setCurrentIndex(0)

    # --- PAGE 3: KẾT QUẢ ---
    def build_result_page(self):
        """
        Mục đích:
          Dựng page hiển thị ảnh query và 5 kết quả gần nhất.

        Tham số:
          Không có tham số; dùng layout/widget thuộc main window.

        Vì sao chọn không truyền tham số:
          Số lượng kết quả của UI cố định là 5 theo yêu cầu hệ thống, nên page có
          thể dựng sẵn 5 slot kết quả.

        Đầu ra:
          Không return; tạo `self.img_query`, `self.res_widgets` và add page.

        Vì sao đầu ra như vậy mà không tạo động sau khi search:
          Dựng sẵn slot giúp kết quả đổ vào nhanh, layout ổn định và tránh nhảy
          giao diện khi thread kết thúc.
        """
        self.result_page = QWidget()
        layout = QVBoxLayout(self.result_page)
        layout.setContentsMargins(20, 15, 20, 15)

        # Title
        header = QLabel("KẾT QUẢ TÌM KIẾM — Click vào ảnh để phóng to")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #1976D2; margin-bottom: 10px;")
        layout.addWidget(header)

        # Main Content Grid (Left: Query | Right: Top-5)
        content_layout = QHBoxLayout()

        # ── Left Panel (Query) ──
        query_frame = QFrame()
        query_frame.setStyleSheet(
            "QFrame { background-color: #f5f5f5; border: 1px solid #dddddd;"
            " border-radius: 8px; padding: 10px; }"
        )
        query_panel = QVBoxLayout(query_frame)

        label_q = QLabel("ẢNH GỐC (INPUT)")
        label_q.setAlignment(Qt.AlignCenter)
        label_q.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #1976D2; border: none;"
            " background: transparent;"
        )
        query_panel.addWidget(label_q)

        self.img_query = ClickableImageLabel()
        query_panel.addWidget(self.img_query)
        query_panel.addStretch()

        content_layout.addWidget(query_frame, stretch=1)

        # ── Right Panel (Top 5 results) ──
        result_frame = QFrame()
        result_frame.setStyleSheet(
            "QFrame { background-color: #f5f5f5; border: 1px solid #dddddd;"
            " border-radius: 8px; padding: 10px; }"
        )
        right_panel = QVBoxLayout(result_frame)

        label_r = QLabel("TOP 5 TƯƠNG ĐỒNG NHẤT")
        label_r.setAlignment(Qt.AlignCenter)
        label_r.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #2e7d32; border: none;"
            " background: transparent;"
        )
        right_panel.addWidget(label_r)

        self.res_grid = QGridLayout()
        self.res_grid.setSpacing(10)
        self.res_widgets = []

        for i in range(5):
            img_lbl = ClickableImageLabel()
            text_lbl = QLabel("")
            text_lbl.setAlignment(Qt.AlignCenter)
            text_lbl.setStyleSheet(
                "font-size: 12px; margin-top: 4px; color: #333333;"
                " border: none; background: transparent;"
            )
            text_lbl.setWordWrap(True)

            box = QVBoxLayout()
            box.addWidget(img_lbl)
            box.addWidget(text_lbl)

            row = i // 3
            col = i % 3
            self.res_grid.addLayout(box, row, col)
            self.res_widgets.append((img_lbl, text_lbl))

        right_panel.addLayout(self.res_grid)
        right_panel.addStretch()

        content_layout.addWidget(result_frame, stretch=3)

        layout.addLayout(content_layout)

        # ── Bottom: Back button ──
        btn_row = QHBoxLayout()
        self.btn_back = QPushButton("← Trở Lại Trang Import")
        self.btn_back.setStyleSheet("""
            QPushButton {
                background-color: #757575; color: white;
                font-size: 15px; padding: 10px 25px; border-radius: 6px;
            }
            QPushButton:hover { background-color: #616161; }
        """)
        self.btn_back.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        btn_row.addStretch()
        btn_row.addWidget(self.btn_back)
        btn_row.addStretch()

        layout.addLayout(btn_row)
        self.stacked_widget.addWidget(self.result_page)  # Index 2

    # ========================================================================
    # XỬ LÝ BACKGROUND LOGIC
    # ========================================================================
    def start_processing(self, file_path):
        """
        Mục đích:
          Bắt đầu xử lý ảnh truy vấn trong worker thread và chuyển sang loading.

        Tham số:
          file_path: Đường dẫn ảnh do người dùng chọn hoặc kéo-thả.

        Vì sao chọn tham số này:
          Đây là dữ liệu duy nhất cần để pipeline `extract_features` đọc ảnh và
          tạo vector query.

        Đầu ra:
          Không return; tạo thread, nối signal và start.

        Vì sao đầu ra như vậy mà không xử lý đồng bộ:
          Extraction + FAISS có thể mất thời gian, chạy đồng bộ sẽ làm UI treo.
        """
        # Reset loading text mỗi lần
        self.loading_label.setText(
            "⏳ Đang trích xuất đặc trưng và truy vấn FAISS IVF...\nVui lòng chờ."
        )
        self.loading_label.setStyleSheet("color: #E65100; font-weight: bold;")

        self.stacked_widget.setCurrentIndex(1)  # Loading

        self.thread = ProcessThread(file_path)
        self.thread.finished.connect(self.display_results)
        self.thread.error.connect(self.show_error)
        self.thread.start()

    def display_results(self, top_k_results, query_cv_img):
        """
        Mục đích:
          Đổ ảnh query và danh sách top-5 vào page kết quả.

        Tham số:
          top_k_results: List dict từ `FingerprintVectorDB.search_top_k`.
          query_cv_img: Ảnh query grayscale để hiển thị bên trái.

        Vì sao chọn tham số này:
          Worker emit đúng hai thứ UI cần: ảnh đầu vào và metadata/đường dẫn của
          kết quả tìm kiếm.

        Đầu ra:
          Không return; cập nhật label ảnh/text và chuyển sang result page.

        Vì sao đầu ra như vậy mà không trả widget/text:
          UI phải thay đổi trực tiếp trên các widget đã dựng sẵn, không cần caller
          xử lý thêm.
        """
        # Set panel Query
        self.img_query.set_image(cv_img=query_cv_img, size=(320, 420))

        # Trải array kết quả vào res_widgets list
        for i in range(5):
            img_lbl, text_lbl = self.res_widgets[i]

            if i < len(top_k_results):
                res = top_k_results[i]
                img_path = res["source_image"]
                sim = res["similarity"]
                clus = res["cluster_id"]
                file_base = os.path.basename(img_path)

                text_lbl.setText(
                    f"Rank {i+1}: {file_base}\n"
                    f"UserID: {res.get('user_id','?')} | {res.get('sex','?')} | {res.get('finger_index','?')}\n"
                    f"Sim: {sim:.3f} | Cluster: {clus}"
                )
                if sim > 0.65:
                    text_lbl.setStyleSheet(
                        "font-size: 12px; margin-top: 4px; color: #2e7d32;"
                        " font-weight: bold; border: none; background: transparent;"
                    )
                else:
                    text_lbl.setStyleSheet(
                        "font-size: 12px; margin-top: 4px; color: #333333;"
                        " border: none; background: transparent;"
                    )

                if os.path.exists(img_path):
                    img_lbl.set_image(file_path=img_path, size=(200, 250))
                else:
                    text_lbl.setText(f"Rank {i+1}: File not found")
                    img_lbl.clear()
            else:
                text_lbl.setText("")
                img_lbl.clear()

        # Hiện màn kết quả
        self.stacked_widget.setCurrentIndex(2)

    def show_error(self, err_msg):
        """
        Mục đích:
          Hiển thị lỗi xử lý ảnh/search trên loading page.

        Tham số:
          err_msg: Nội dung lỗi đã được worker rút gọn thành chuỗi.

        Vì sao chọn tham số này:
          UI chỉ cần chuỗi thân thiện; traceback đầy đủ đã in ra terminal trong
          worker để debug.

        Đầu ra:
          Không return; cập nhật text/style của loading label và nút back.

        Vì sao đầu ra như vậy mà không raise lỗi:
          Lỗi trong GUI nên được trình bày cho người dùng và cho phép thử lại,
          không làm crash ứng dụng.
        """
        self.loading_label.setText(
            f"⚠️ Đã xảy ra lỗi:\n\n{err_msg}\n\nNhấn ← Quay Lại để thử ảnh khác."
        )
        self.loading_label.setStyleSheet(
            "color: #c62828; font-weight: bold; font-size: 14px;"
        )
        self.btn_back_loading.setText("← Quay Lại Import")


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    """
    Mục đích:
      Entry point chạy ứng dụng GUI tìm top-5 vân tay.

    Tham số:
      Không có tham số; dùng `sys.argv` cho QApplication và `DB_PATH` từ config.

    Vì sao chọn không truyền tham số:
      Người dùng mở GUI sau khi DB đã build, nên cấu hình runtime lấy từ file
      project thay vì dòng lệnh.

    Đầu ra:
      Không return trong luồng bình thường; gọi `sys.exit(app.exec_())`.

    Vì sao đầu ra như vậy mà không trả mã lỗi thủ công:
      Đây là chuẩn vận hành của PyQt, để event loop quyết định exit code.
    """
    # Kiểm tra DB có chưa
    if not os.path.exists(DB_PATH):
        print(
            "❌ Lỗi: CSDL 'fingerprint.db' chưa tồn tại."
            " Vui lòng chạy 10_database_system.py trước!"
        )
        return

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = FingerprintApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
