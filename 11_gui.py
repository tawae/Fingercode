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
import time
import traceback
import numpy as np
import cv2

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QStackedWidget, QGridLayout, QFrame, QDialog,
                             QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont, QCursor, QPalette, QColor

# ============================================================================
# IMPORT HỆ THỐNG BIOMETRIC (Từ Bước 08 và 10)
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from importlib.util import spec_from_file_location, module_from_spec

def _import_module(name, filepath):
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
    """QLabel tĩnh có sự kiện click để bật dialog Zoom."""
    clicked = pyqtSignal()

    def __init__(self, parent=None):
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
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            if self.base_pixmap is not None:
                self.show_zoom_dialog()

    def set_image(self, file_path=None, cv_img=None, size=(300, 400)):
        """Thiết lập ảnh từ file path hoặc từ numpy tensor cv2"""
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
        """Mở cửa sổ Dialog phóng to giữ nguyên tỷ lệ mượt."""
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
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText(
            "\n\nKéo & Thả ảnh vân tay (TIF, PNG, JPG) vào đây\n"
            "hoặc Click vào nút bên dưới (Select File)\n\n"
        )
        self._set_default_style()
        self.setAcceptDrops(True)

    def _set_default_style(self):
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
        self._set_default_style()

    def dropEvent(self, event):
        self._set_default_style()
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.file_dropped.emit(file_path)


# ============================================================================
# THREAD: Xử lý Biometrics Background
# ============================================================================
class ProcessThread(QThread):
    finished = pyqtSignal(object, object)  # list_top5, query_cv_img
    error = pyqtSignal(str)

    def __init__(self, img_path):
        super().__init__()
        self.img_path = img_path

    def run(self):
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
    def __init__(self):
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
        global global_db
        global_db = step10.FingerprintVectorDB(DB_PATH)
        global_db.connect()

    # --- PAGE 1: IMPORT & DRAG-DROP ---
    def build_import_page(self):
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
        """Quay về Import page, dừng thread nếu đang chạy."""
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.terminate()
            self.thread.wait()
        self.stacked_widget.setCurrentIndex(0)

    # --- PAGE 3: KẾT QUẢ ---
    def build_result_page(self):
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
        """Khởi chạy Worker Thread cho logic biometrics"""
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
        """Cập nhật UI màn kết quả khi Thread chạy xong"""
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
