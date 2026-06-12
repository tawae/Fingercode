# Fingerprint Recognition System in Python

He thong nhan dang van tay nay dung pipeline Fingercode + FAISS IVF:

1. Nap san 3000 anh SOCOFing Real dau tien vao SQLite va FAISS index.
2. GUI PyQt5 cho phep chon/keo tha mot anh truy van va tra ve 5 anh gan nhat.
3. Hai script danh gia do chinh xac va nguong nhan dang.

## Luong he thong

Luong xu ly anh dung chung cho build DB, GUI va evaluation:

```text
03_enhancement.py
  -> 04_orientation_field.py
  -> 05_frequency_estimation.py
  -> 06_gabor_filter.py
  -> 08_fingercode_extraction.py
  -> 10_database_system.py / 11_gui.py / 12_evaluate_FAR_FRR.py / 13_eval_acc_recall_preci.py
```

Vai tro tung file Python con lai:

| File | Vai tro |
| --- | --- |
| `config.py` | Cau hinh duong dan dataset, DB, FAISS index, output va so ket qua top-k. |
| `03_enhancement.py` | Chuan hoa anh, tach vung van tay va tang cuong tuong phan bang CLAHE. |
| `04_orientation_field.py` | Uoc luong huong duong van bang Sobel, covariance va doubled-angle smoothing. |
| `05_frequency_estimation.py` | Uoc luong tan so/buoc song duong van theo tung block. |
| `06_gabor_filter.py` | Tao kernel Gabor theo huong va tan so de `08` trich Fingercode. |
| `08_fingercode_extraction.py` | Tao vector Fingercode 320 chieu tu anh van tay. |
| `10_database_system.py` | Build lai `fingerprint.db` va `faiss_ivf.index` tu 300 nguoi dau, moi nguoi 10 ngon = 3000 anh. |
| `11_gui.py` | GUI tim kiem anh truy van va hien thi top-5 anh gan nhat. |
| `12_evaluate_FAR_FRR.py` | Danh gia Genuine/Impostor, FAR, FRR, ROC, EER. |
| `13_eval_acc_recall_preci.py` | Danh gia Top-1, Top-5 accuracy, macro precision va macro recall. |

## File da loai bo

Cac file sau la script hoc thu/demo doc lap, khong nam trong luong san pham hien tai:

| File | Ly do loai bo |
| --- | --- |
| `01_visualize_fingerprint.py` | Chi ve anh va histogram, khong duoc import boi DB, GUI hay evaluation. |
| `02_preprocessing.py` | Trung chuc nang voi `normalize_image` va `segment_fingerprint` trong `03_enhancement.py`. |
| `09_matching.py` | Demo so khop 1-vs-1 rieng le, khong phai luong top-5 qua SQLite + FAISS. |
| `Day1_Concepts.md`, `Day2_Preprocessing.md` | Ghi chu hoc tap cho `01/02`, khong con dung sau khi loai cac demo do. |

## Cai dat

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dataset duoc cau hinh mac dinh o:

```text
SOCOFing/Real
SOCOFing/Altered/Altered-Easy
```

Neu can doi dataset, sua `DATASET_PATH` trong `config.py`.

## Cach chay

Build lai database va FAISS index:

```bash
python 10_database_system.py
```

Chay GUI tim top-5:

```bash
python 11_gui.py
```

Chay danh gia FAR/FRR/EER:

```bash
python 12_evaluate_FAR_FRR.py
```

Chay danh gia Top-1/Top-5, precision, recall:

```bash
python 13_eval_acc_recall_preci.py
```

## Du lieu sinh ra

| File/thu muc | Noi dung |
| --- | --- |
| `fingerprint.db` | SQLite DB luu metadata va vector Fingercode. |
| `faiss_ivf.index` | FAISS IVF index dung de search top-k. |
| `output/` | Anh va bao cao danh gia sinh ra tu cac script. |
| `evaluation_results.*` | Ket qua danh gia tu `13_eval_acc_recall_preci.py`. |
