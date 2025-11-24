import os
import shutil


# Lưu ý: Python dùng dấu gạch chéo / hoặc 2 gạch chéo ngược \\
source_dir = os.path.join("dataset", "raw", "raw") 

# Đường dẫn đến thư mục MỚI, SẠCH SẼ
target_dir = "data_clean"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Từ điển ánh xạ: Tên file cũ -> Tên file mới
files_to_move = {
    "train.en": "train.en",
    "train.fr": "train.fr",
    "val.en":   "val.en",
    "val.fr":   "val.fr",
    "test_2016_flickr.en": "test_2016.en", # Đổi tên cho ngắn gọn
    "test_2016_flickr.fr": "test_2016.fr"
}

print(f" Đang dọn dẹp từ '{source_dir}' sang '{target_dir}'...")

found_count = 0
for old_name, new_name in files_to_move.items():
    old_path = os.path.join(source_dir, old_name)
    new_path = os.path.join(target_dir, new_name)
    
    if os.path.exists(old_path):
        shutil.copy(old_path, new_path)
        print(f" Đã copy: {old_name} -> {new_name}")
        found_count += 1
    else:
        print(f" KHÔNG TÌM THẤY: {old_name} (Kiểm tra lại đường dẫn!)")

if found_count == 6:
    print("\n TUYỆT VỜI! đủ vốn khởi nghiệp rồi đóa.")
    print(f" Từ giờ chỉ làm việc trong folder: '{target_dir}' thôi nhé!")
else:
    print(f"\n Có biến! Chỉ tìm thấy {found_count}/6 file. Soi lại folder.")