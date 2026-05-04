import os
import glob
import importlib
import json
import csv
import time
from sklearn.metrics import precision_score, recall_score, accuracy_score
import config

# Sử dụng importlib để import các module có tên bắt đầu bằng số
fingercode_extraction = importlib.import_module("08_fingercode_extraction")
extract_features = fingercode_extraction.extract_features

database_system = importlib.import_module("10_database_system")
FingerprintVectorDB = database_system.FingerprintVectorDB

def evaluate_system():
    t_start = time.time()
    # Đường dẫn tới tập ảnh Altered-Easy (tuyệt đối)
    _base = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(_base, "SOCOFing", "Altered", "Altered-Easy")
    all_test_files = sorted(glob.glob(os.path.join(test_dir, "*.BMP")))

    # Lọc chỉ giữ user_id 1-50 (50 người đầu tiên)
    MAX_USER_ID_EVAL = 50
    test_files = []
    for fp in all_test_files:
        parts = os.path.basename(fp).split('__')
        if len(parts) >= 2:
            try:
                if 1 <= int(parts[0]) <= MAX_USER_ID_EVAL:
                    test_files.append(fp)
            except ValueError:
                pass
    
    y_true = []
    y_pred_top1 = []
    
    # Biến đếm cho top-5 accuracy
    correct_top5_count = 0
    total_queries = 0

    results_details = []

    print(f"Bắt đầu đánh giá trên {len(test_files)} ảnh (user_id 1-{MAX_USER_ID_EVAL}) từ Altered-Easy...")

    # Khởi tạo kết nối tới Database FAISS
    print(f"Kết nối tới DB FAISS tại: {config.DB_PATH}")
    db = FingerprintVectorDB(config.DB_PATH)
    db.connect()

    for i, file_path in enumerate(test_files):
        filename = os.path.basename(file_path)
        # Bóc tách ID thực tế từ tên file (ví dụ "1__M_Left..." lấy số "1")
        true_id = filename.split('__')[0]
        
        print(f"[{i+1}/{len(test_files)}] Đang xử lý ảnh: {filename} (ID: {true_id})")
        
        # 1. Trích xuất đặc trưng
        vector, _img = extract_features(file_path)
        if vector is None:
            print("  -> Lỗi: Không thể trích xuất đặc trưng Fingercode. Bỏ qua.")
            # Nếu không tìm thấy tâm vân tay, coi như hệ thống không nhận diện được
            y_true.append(true_id)
            y_pred_top1.append("Unknown")
            total_queries += 1
            
            results_details.append({
                "file_name": filename,
                "true_id": true_id,
                "pred_top1_id": "Unknown",
                "top_5_ids": [],
                "is_top1_correct": False,
                "is_top5_correct": False
            })
            continue
            
        # 2. Truy vấn top 5 kết quả từ FAISS
        # Hàm này trả về list các dictionary thông tin ảnh
        top_k_dicts = db.search_top_k(vector, k=5) 
        
        top_k_results = []
        for item in top_k_dicts:
            res_filename = os.path.basename(item["source_image"])
            res_id = res_filename.split('__')[0]
            top_k_results.append(res_id)
            
        print(f"  -> Top 5 dự đoán ID: {top_k_results}")
        
        if not top_k_results:
            print("  -> Lỗi: Không tìm thấy kết quả từ FAISS Index.")
            y_true.append(true_id)
            y_pred_top1.append("Unknown")
            total_queries += 1
            results_details.append({
                "file_name": filename,
                "true_id": true_id,
                "pred_top1_id": "Unknown",
                "top_5_ids": [],
                "is_top1_correct": False,
                "is_top5_correct": False
            })
            continue

        # Lấy kết quả Top 1 để tính các chỉ số phân loại tiêu chuẩn
        pred_top1_id = top_k_results[0]
        y_true.append(true_id)
        y_pred_top1.append(pred_top1_id)
        
        # Ép kiểu về string để so sánh an toàn
        is_top1_correct = (str(pred_top1_id) == str(true_id))
        is_top5_correct = (str(true_id) in [str(res) for res in top_k_results])

        if is_top1_correct:
            print("  -> Đánh giá: ĐÚNG Top-1")
        elif is_top5_correct:
            print("  -> Đánh giá: ĐÚNG Top-5")
        else:
            print("  -> Đánh giá: SAI hoàn toàn")

        # Kiểm tra Top 5
        if is_top5_correct:
            correct_top5_count += 1
            
        total_queries += 1
        
        results_details.append({
            "file_name": filename,
            "true_id": true_id,
            "pred_top1_id": pred_top1_id,
            "top_5_ids": top_k_results,
            "is_top1_correct": is_top1_correct,
            "is_top5_correct": is_top5_correct
        })

    # Đóng kết nối database
    db.close()

    # 3. Tính toán các chỉ số
    elapsed = time.time() - t_start
    # QUAN TRỌNG: labels=sorted(set(y_true)) để chỉ tính macro trên các user_id
    # thực sự có trong tập test. Nếu không, các predicted user_id ngoài tập test
    # (vd: user 51-300 trong DB) tạo thêm class "ma" với precision=0, kéo kết quả xuống.
    eval_labels = sorted(set(y_true))
    acc_top1 = accuracy_score(y_true, y_pred_top1)
    precision = precision_score(y_true, y_pred_top1, average='macro', labels=eval_labels, zero_division=0)
    recall = recall_score(y_true, y_pred_top1, average='macro', labels=eval_labels, zero_division=0)
    acc_top5 = correct_top5_count / total_queries if total_queries > 0 else 0
    avg_time = elapsed / total_queries if total_queries > 0 else 0

    metrics = {
        "Total_Images": total_queries,
        "Top-1_Accuracy": round(acc_top1, 4),
        "Top-5_Accuracy": round(acc_top5, 4),
        "Macro_Precision": round(precision, 4),
        "Macro_Recall": round(recall, 4),
        "Total_Time_Seconds": round(elapsed, 2),
        "Avg_Time_Per_Query_ms": round(avg_time * 1000, 2)
    }

    print("\n=== KẾT QUẢ ĐÁNH GIÁ HỆ THỐNG ===")
    print(f"Tổng số ảnh đánh giá: {total_queries}")
    print(f"Top-1 Accuracy : {acc_top1:.4f}")
    print(f"Top-5 Accuracy : {acc_top5:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Tổng thời gian : {elapsed:.1f}s")
    print(f"TB mỗi query   : {avg_time*1000:.1f}ms")

    # --- LƯU KẾT QUẢ ---
    
    # 1. Lưu dưới dạng JSON
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            "metrics": metrics,
            "details": results_details
        }, f, indent=4, ensure_ascii=False)
        
    # 2. Lưu dưới dạng CSV
    with open('evaluation_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
        
        writer.writerow([]) # Dòng trống phân cách
        writer.writerow(["file_name", "true_id", "pred_top1_id", "top_5_ids", "is_top1_correct", "is_top5_correct"])
        for item in results_details:
            writer.writerow([
                item["file_name"], 
                item["true_id"], 
                item["pred_top1_id"], 
                ", ".join(map(str, item["top_5_ids"])), 
                item["is_top1_correct"], 
                item["is_top5_correct"]
            ])
            
    # 3. Lưu dưới dạng TXT
    with open('evaluation_results.txt', 'w', encoding='utf-8') as f:
        f.write("=== KẾT QUẢ ĐÁNH GIÁ HỆ THỐNG ===\n")
        f.write(f"Tổng số ảnh đánh giá: {total_queries}\n")
        f.write(f"Top-1 Accuracy : {acc_top1:.4f}\n")
        f.write(f"Top-5 Accuracy : {acc_top5:.4f}\n")
        f.write(f"Macro Precision: {precision:.4f}\n")
        f.write(f"Macro Recall   : {recall:.4f}\n")
        f.write(f"Tổng thời gian : {elapsed:.1f}s\n")
        f.write(f"TB mỗi query   : {avg_time*1000:.1f}ms\n")
        f.write("\n=== CHI TIẾT ===\n")
        for item in results_details:
            top_5_str = ", ".join(map(str, item['top_5_ids']))
            f.write(f"File: {item['file_name']} | True ID: {item['true_id']} | Top 1: {item['pred_top1_id']} | Top 5: [{top_5_str}] | Top-1 Đúng: {item['is_top1_correct']} | Top-5 Đúng: {item['is_top5_correct']}\n")

    print("\nĐã lưu kết quả đánh giá vào các file: evaluation_results.json, evaluation_results.csv, evaluation_results.txt")

if __name__ == "__main__":
    evaluate_system()