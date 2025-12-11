# 🇬🇧 🇫🇷 English-to-French Neural Machine Translation

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)]()

> **Dự án Dịch máy Nơ-ron (Neural Machine Translation)** triển khai từ đầu (from scratch) kiến trúc **Encoder-Decoder LSTM** để dịch tiếng Anh sang tiếng Pháp trên bộ dữ liệu **Multi30K**.

---

## 📑 Mục lục (Table of Contents)
- [Giới thiệu](#-giới-thiệu-introduction)
- [Kiến trúc Mô hình](#-kiến-trúc-mô-hình-model-architecture)
- [Cài đặt & Yêu cầu](#-cài-đặt--yêu-cầu-installation)
- [Sử dụng](#-sử-dụng-usage)
- [Kết quả Thực nghiệm](#-kết-quả-thực-nghiệm-results)
- [Cấu trúc Thư mục](#-cấu-trúc-thư-mục-project-structure)
- [Hạn chế & Hướng phát triển](#-hạn-chế--hướng-phát-triển-limitations--future-work)
- [Tài liệu Tham khảo](#-tài-liệu-tham-khảo-references)

---

## 🚀 Giới thiệu (Introduction)

Dự án này là một phần của đồ án môn học Xử lý Ngôn ngữ Tự nhiên (NLP). Mục tiêu là xây dựng một hệ thống dịch máy **Sequence-to-Sequence (Seq2Seq)** sử dụng mạng nơ-ron hồi quy (RNN/LSTM) mà không phụ thuộc vào các thư viện dịch thuật cấp cao (như Transformers hay Torchtext Legacy).

Điểm nổi bật của dự án:
* Xử lý dữ liệu văn bản thô (Raw text processing).
* Xây dựng Dataset và Dataloader tùy chỉnh trong PyTorch.
* Triển khai kỹ thuật **Teacher Forcing** và **Packed Padded Sequences**.
* Đánh giá mô hình bằng **BLEU Score**.

---

## 🧠 Kiến trúc Mô hình (Model Architecture)

Mô hình dựa trên kiến trúc **Seq2Seq** kinh điển được đề xuất bởi Sutskever et al. (2014), bao gồm:

| Thành phần | Thông số kỹ thuật | Mô tả |
| :--- | :--- | :--- |
| **Encoder** | `LSTM(emb_dim=256, hid_dim=512, n_layers=2)` | Đọc câu nguồn và nén thông tin vào Context Vector ($h_n, c_n$). |
| **Decoder** | `LSTM(emb_dim=256, hid_dim=512, n_layers=2)` | Sinh từ dựa trên Context Vector và từ dự đoán trước đó. |
| **Embedding** | `256` | Kích thước vector biểu diễn từ. |
| **Dropout** | `0.5` | Áp dụng giữa các lớp LSTM để chống Overfitting. |

**Chiến lược huấn luyện:**
* **Optimizer:** Adam ($lr=0.001$).
* **Loss Function:** `CrossEntropyLoss` (bỏ qua token `<pad>`).
* **Teacher Forcing Ratio:** $0.5$ (50% sử dụng ground truth).

---

## 🛠 Cài đặt & Yêu cầu (Installation)

1. **Clone dự án:**
   ```bash
   git clone [https://github.com/TidalWavetop1/NLP_Translate.git](https://github.com/TidalWavetop1/NLP_Translate.git)
   cd NLP_Translate

2.  **Cài đặt thư viện:**
    Dự án yêu cầu Python 3.7+ và các thư viện sau:

    ```bash
    pip install torch torchvision torchtext spacy numpy matplotlib
    ```

3.  **Tải mô hình ngôn ngữ cho SpaCy:**
    Dùng để tokenize tiếng Anh và tiếng Pháp.

    ```bash
    python -m spacy download en_core_web_sm
    python -m spacy download fr_core_news_sm
    ```

-----

## 💻 Sử dụng (Usage)

### 1\. Huấn luyện (Training)

Mở file Notebook `NLP_Translate.ipynb` hoặc chạy script (nếu có) để bắt đầu huấn luyện. Quá trình huấn luyện sẽ tự động tải dữ liệu Multi30K, xây dựng từ điển và lưu checkpoint tốt nhất vào `best_model.pth`.

### 2\. Dự đoán (Inference/Translation)

Sử dụng hàm `translate_sentence` để dịch một câu mới:

```python
import torch
# Đảm bảo bạn đã import đúng class Encoder, Decoder, Seq2Seq từ file source
from your_model_file import translate_sentence, load_model

# 1. Load Model
# (Code khởi tạo model architecture phải khớp với config lúc train)
model = load_model('best_model.pth') 

# 2. Input Sentence
src_sentence = "Two men are walking on the street."

# 3. Translate
translation, attention = translate_sentence(src_sentence, model, device)

print(f"SRC: {src_sentence}")
print(f"TRG: {' '.join(translation)}")
```

-----

## 📊 Kết quả Thực nghiệm (Results)

Sau quá trình huấn luyện trên GPU (Google Colab), mô hình đạt được các chỉ số sau trên tập Test:

| Metric | Giá trị | Nhận xét |
| :--- | :--- | :--- |
| **BLEU Score** | **\40.01** | Kết quả khả quan cho kiến trúc LSTM cơ bản (Fixed Context Vector). |
| **Train Loss** | **\~1.581** | Mô hình hội tụ tốt, không bị Overfitting nặng. |

### Ví dụ mô hình dịch (Sample Outputs)

Dưới đây là kết quả thực tế từ tập Test, minh họa các trường hợp mô hình hoạt động tốt và các hạn chế còn tồn tại:

| Loại | Tiếng Anh (Input) | Tiếng Pháp (Prediction) | Đánh giá |
| :--- | :--- | :--- | :--- |
| **Tốt** | *A man in an orange hat starring at something.* | *un homme avec un chapeau orange regardant quelque chose .* | ✅ **Chính xác:** Dịch đúng toàn bộ từ vựng và cấu trúc. |
| **Khá** | *Five people wearing winter jackets... with snowmobiles in the background.* | *cinq personnes portant des manteaux... avec des **\<unk>** en arrière-plan .* | ⚠️ **Lỗi OOV:** Cấu trúc câu phức tạp được dịch mượt mà, nhưng từ hiếm "snowmobiles" bị thay thế bằng `<unk>`. |
| **Kém** | *A girl in karate uniform breaking a stick with a front kick.* | *une fille en tenue de karaté **karaté un un avec un un** .* | ❌ **Lỗi lặp từ:** Mô hình bị mất thông tin ngữ cảnh ở đoạn hành động phức tạp, dẫn đến lặp từ vô nghĩa. |

-----

## 📂 Cấu trúc Thư mục (Project Structure)

```
NLP_Translate/
├── data_clean                # Folder chưa dũ liệu sau khi xử lý và làm sạch
├── datase                    # Folder dữ liệu thô
├── model            # Các mô hình đã huấn luyện
    ├── model_translate_(3).ipynb       # Source code sau cùng 
├── path  # Folder chứa checkpoint của mô hình tốt nhất            
└── README.md                 # Tài liệu hướng dẫn
```

-----

## ⚠️ Hạn chế & Hướng phát triển (Limitations & Future Work)

**Hạn chế:**

  * **Nút thắt cổ chai (Information Bottleneck):** Context vector cố định không thể nén hết thông tin của các câu dài (\>20 từ).
  * **Từ vựng hiếm (OOV):** Các từ không nằm trong top 10.000 từ phổ biến sẽ bị thay thế bằng `<unk>`.

**Hướng phát triển:**

  * [ ] Tích hợp cơ chế **Attention (Bahdanau/Luong)** để cải thiện hiệu năng trên câu dài.
  * [ ] Sử dụng **Beam Search Decoding** thay vì Greedy Decoding.
  * [ ] Chuyển sang kiến trúc **Transformer**.

-----

## 📚 Tài liệu Tham khảo (References)

[1] I. Sutskever, O. Vinyals, and Q. V. Le, "Sequence to sequence learning with neural networks," in *Advances in Neural Information Processing Systems 27*, 2014.

[2] K. Cho *et al.*, "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in *EMNLP*, 2014.

[3] PyTorch Documentation. "LSTM — PyTorch 2.0 documentation".

-----

