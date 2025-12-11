# 🇬🇧 🇫🇷 English-to-French Neural Machine Translation

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](./LICENSE)

> **Dự án Dịch máy Nơ-ron (Neural Machine Translation)** sử dụng kiến trúc **Encoder-Decoder LSTM** để dịch tiếng Anh sang tiếng Pháp trên bộ dữ liệu **Multi30K**.

---

## 📑 Mục lục (Table of Contents)
- [Giới thiệu](#-giới-thiệu-introduction)
- [Kiến trúc Mô hình](#-kiến-trúc-mô-hình-model-architecture)
- [Cài đặt & Yêu cầu](#-cài-đặt--yêu-cầu-installation)
- [Sử dụng](#-sử-dụng-usage)
- [Kết quả Thực nghiệm](#-kết-quả-thực-nghiệm-results)
- [Cấu trúc Thư mục](#-cấu-trúc-thư-mục-project-structure)
- [Tài liệu Tham khảo](#-tài-liệu-tham-khảo-references)

---

## 🚀 Giới thiệu (Introduction)

Dự án này triển khai một hệ thống dịch máy Sequence-to-Sequence (Seq2Seq) từ con số 0 (from scratch) bằng thư viện **PyTorch**. Mục tiêu là xây dựng một mô hình cơ bản nhưng hiệu quả để giải quyết bài toán dịch thuật giữa hai ngôn ngữ có cấu trúc khác biệt, đồng thời minh họa các khái niệm cốt lõi như:
* Mạng LSTM đa tầng (Stacked LSTM).
* Cơ chế Vector ngữ cảnh cố định (Fixed Context Vector).
* Kỹ thuật Teacher Forcing.

---

## 🧠 Kiến trúc Mô hình (Model Architecture)

Mô hình dựa trên kiến trúc **Seq2Seq** được đề xuất bởi Sutskever et al. (2014) [1], bao gồm hai thành phần chính:

| Thành phần | Chi tiết kỹ thuật |
| :--- | :--- |
| **Encoder** | • Sử dụng **LSTM 2 lớp** (2-layer LSTM).<br>• Nén câu nguồn thành **Context Vector** ($h_n, c_n$).<br>• Embedding Dimension: 256. |
| **Decoder** | • Sử dụng **LSTM 2 lớp**.<br>• Khởi tạo trạng thái từ Context Vector.<br>• Sử dụng **Teacher Forcing** (Ratio = 0.5) trong quá trình huấn luyện. |
| **Optimization** | • Optimizer: **Adam** ($lr=0.001$).<br>• Loss Function: **CrossEntropyLoss** (bỏ qua padding). |

---
