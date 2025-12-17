# 🇬🇧 🇫🇷 English-to-French Neural Machine Translation

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)]()

> **Dự án Dịch máy Nơ-ron (Neural Machine Translation)** triển khai kiến trúc **Encoder-Decoder LSTM tích hợp cơ chế Attention (Luong Attention)** để dịch tiếng Anh sang tiếng Pháp trên bộ dữ liệu **Multi30K**.

---

**👥 Nhóm thực hiện:**
- **Đỗ Nguyễn Thanh Phong** (Email: donguyenthanhphong2005@gmail.com)
- **Trịnh Minh Toàn** (Email: [Điền email bạn Toàn vào đây])

---

## 📑 Mục lục (Table of Contents)
- [Giới thiệu](#-giới-thiệu-introduction)
- [Kiến trúc Mô hình](#-kiến-trúc-mô-hình-model-architecture)
- [Cài đặt & Yêu cầu](#-cài-đặt--yêu-cầu-installation)
- [Sử dụng](#-sử-dụng-usage)
- [Kết quả Thực nghiệm](#-kết-quả-thực-nghiệm-results)
- [Cấu trúc Thư mục](#-cấu-trúc-thư-mục-project-structure)
- [Hạn chế & Hướng phát triển](#-hạn-chế--hướng-phát-triển-limitations--future-work)

---

## 🚀 Giới thiệu (Introduction)

Dự án này là đồ án môn học Xử lý Ngôn ngữ Tự nhiên (NLP). Mục tiêu là xây dựng một hệ thống dịch máy **Sequence-to-Sequence (Seq2Seq)** có khả năng xử lý ngữ cảnh tốt hơn mô hình truyền thống nhờ cơ chế **Global Attention**.

**Điểm nổi bật:**
* Sử dụng **Attention Mechanism** để giải quyết vấn đề "nút thắt cổ chai" thông tin.
* Xử lý dữ liệu văn bản thô (Raw text processing) với `spaCy`.
* Tối ưu hóa huấn luyện với **Packed Padded Sequences** và **Teacher Forcing**.
* Code hoàn toàn bằng **PyTorch** thuần (không dùng thư viện có sẵn như `torchtext.legacy`).

---

## 🧠 Kiến trúc Mô hình (Model Architecture)

Mô hình áp dụng kiến trúc **Encoder-Decoder với Global Attention** (theo Luong et al., 2015), được cấu hình tinh gọn để chạy trên Google Colab:

| Thành phần | Thông số kỹ thuật | Mô tả |
| :--- | :--- | :--- |
| **Encoder** | `LSTM(emb=128, hid=128, layers=1)` | LSTM 2 chiều (Bidirectional) hoặc 1 chiều, mã hóa câu nguồn thành chuỗi Hidden States. |
| **Attention** | `Linear(hid_dim * 2 -> hid_dim)` | Tính trọng số chú ý (Attention weights) dựa trên Decoder state và Encoder states. |
| **Decoder** | `LSTM(emb=128, hid=128, layers=1)` | Sinh từ dựa trên Context Vector động và từ dự đoán trước đó. |
| **Embedding** | `128` | Kích thước vector biểu diễn từ. |
| **Dropout** | `0.5` | Áp dụng để chống Overfitting. |

**Chiến lược huấn luyện:**
* **Optimizer:** Adam.
* **Loss Function:** `CrossEntropyLoss` (bỏ qua token `<pad>`).
* **Kỹ thuật:** Teacher Forcing (Ratio = 0.5), Clip Gradients (max=1).

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

Mở file Notebook model/model_translate_low_graph (1).ipynb trên Google Colab hoặc Jupyter Notebook để bắt đầu huấn luyện. Quá trình này sẽ:

* Tải và xử lý dữ liệu Multi30K.

* Huấn luyện mô hình qua 15-20 epochs.

* Tự động lưu checkpoint tốt nhất vào thư mục path/.

### 2\. Dự đoán (Inference/Translation)

Sử dụng hàm `translate_sentence` để dịch một câu mới:

```python
import torch
from model import load_model, translate_sentence

# 1. Load Model
model = load_model('path/best_model.pth') 

# 2. Input Sentence
src_sentence = "A man in an orange hat is walking."

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
| **BLEU Score** | **\32.67** | Kết quả khả quan cho kiến trúc LSTM cơ bản (Fixed Context Vector). |
| **Test Loss** | **\~3.114** | Mô hình hội tụ tốt, không bị Overfitting nặng. |

### Ví dụ mô hình dịch (Sample Outputs)

Dưới đây là kết quả thực tế từ tập Test, minh họa các trường hợp mô hình hoạt động tốt và các hạn chế còn tồn tại:

| Loại | Tiếng Anh (Input) | Tiếng Pháp (Prediction) | Đánh giá |
| :--- | :--- | :--- | :--- |
| **Thành công** | *A man in an orange hat starring at something.* | *un homme avec un chapeau orange regardant quelque chose.* | ✅ **Chính xác:** Cấu trúc câu chuẩn xác, Attention bắt đúng tính từ orange. |
| **Lỗi từ vựng** | *A Boston Terrier is running on lush green grass in front of a white fence.* | *un athlète de court sur sur l' herbe verte devant une clôture blanche.* | ⚠️ Từ hiếm "Boston Terrier" bị đoán sai thành "athlète". |
| **Lỗi ngữ nghĩa** | *People are fixing the roof of a house.* | *des gens sont sur le toit d' une maison.
* | ⚠️ Dịch đơn giản hóa hành động "fixing" thành "sont sur" (ở trên). |

-----

## 📂 Cấu trúc Thư mục (Project Structure)

```
NLP_Translate/
├── data_clean                # Folder chưa dũ liệu sau khi xử lý và làm sạch
├── datase                    # Folder dữ liệu thô
├── model            # Các mô hình đã huấn luyện
    ├── model_translate_low_graph (1).ipynb       # Source code sau cùng 
├── path  # Folder chứa checkpoint của mô hình tốt nhất            
└── README.md                 # Tài liệu hướng dẫn
```

-----

## ⚠️ Hạn chế & Hướng phát triển (Limitations & Future Work)

**Hạn chế:**

  * **Vốn từ vựng hạn chế:** Do giới hạn phần cứng, từ điển chỉ khoảng ~6.000 từ, dẫn đến nhiều lỗi OOV (Out-of-Vocabulary).
  * **Mô hình nhỏ:** Hidden size 128 chưa đủ mạnh để ghi nhớ các câu quá phức tạp.

**Hướng phát triển:**

  * [ ] Tăng kích thước Embedding và Hidden size (lên 256 hoặc 512) nếu có GPU mạnh hơn.
  * [ ] Sử dụng **Pre-trained Embeddings** (GloVe, FastText) để cải thiện vốn từ.
  * [ ] Nâng cấp lên kiến trúc **Transformer** (Self-Attention).

-----

## 📚 Tài liệu Tham khảo (References)

[1] I. Sutskever, O. Vinyals, and Q. V. Le, "Sequence to sequence learning with neural networks," in *Advances in Neural Information Processing Systems 27*, 2014.

[2] M.-T. Luong et al., "Effective approaches to attention-based neural machine translation," EMNLP 2015.

[3] PyTorch Documentation. "LSTM — PyTorch 2.0 documentation".

-----

