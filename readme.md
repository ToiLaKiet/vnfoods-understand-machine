# 🍜 VNFoodsDetector
Demo video : https://drive.google.com/file/d/1aht35eIdhrLpwQzuarTi4F3vEBCagoue/view

Hệ thống phân loại món ăn Việt Nam sử dụng deep learning với các mô hình thị giác hiện đại. Dự án này tận dụng nhiều mô hình pre-trained bao gồm CLIP, DinoV2, và ResNet50 để nhận diện chính xác và cung cấp thông tin về 30 loại món ăn Việt Nam khác nhau.

## 📋 Tổng quan

VNFoodsDetector được thiết kế để phân loại các món ăn Việt Nam từ hình ảnh và cung cấp thông tin liên quan về chúng. Hệ thống khám phá nhiều phương pháp trích xuất đặc trưng—từ kỹ thuật computer vision truyền thống đến các mô hình transformer hiện đại—để đạt được hiệu suất phân loại tối ưu.

## 🎯 Tính năng

- **Đa mô hình**: Triển khai nhiều phương pháp trích xuất đặc trưng:
  - **CLIP ViT-L/14**: Mô hình ngôn ngữ-thị giác cho khả năng hiểu ảnh mạnh mẽ
  - **DinoV2**: Vision transformer tự giám sát
  - **ResNet50**: Mạng nơ-ron tích chập sâu
  - Phương pháp truyền thống: Color Histogram, HOG, Local Binary Patterns
  
- **30 loại món ăn Việt Nam**: Phân loại đa dạng các món ăn bao gồm:
  - Bánh bèo, Bánh bột lọc, Bánh căn, Bánh canh, Bánh chưng, Bánh cuốn
  - Bánh đúc, Bánh giò, Bánh khọt, Bánh mì, Bánh pía, Bánh tét
  - Bánh tráng nướng, Bánh xèo
  - Bún bò Huế, Bún đậu mắm tôm, Bún mắm, Bún riêu, Bún thịt nướng
  - Cá kho tộ, Canh chua, Cao lầu, Cháo lòng
  - Cơm tấm, Gỏi cuốn, Hủ tiếu, Mì Quảng
  - Nem chua, Phở, Xôi xéo

- **Demo web**: Flask backend kết hợp React frontend để dự đoán theo thời gian thực

- **Cung cấp thông tin**:  Không chỉ phân loại—còn cung cấp thông tin chi tiết về từng món ăn

## 📊 Dataset

Dự án sử dụng **30VNFoods dataset**, bao gồm:
- 30 loại món ăn Việt Nam
- Hình ảnh được chia thành tập Train, Validate và Test
- Biểu diễn đa dạng cho mỗi loại món ăn

## 🏗️ Cấu trúc dự án

```
VNFoodsDetector/
├── CLIP-ViT-L: 14/              # Triển khai mô hình CLIP và embeddings
├── DinoV2/                      # Triển khai mô hình DinoV2
├── ResNet50/                    # Triển khai mô hình ResNet50
├── Color Histogram/             # Đặc trưng dựa trên màu sắc truyền thống
├── Histogram Of Gradients/      # Trích xuất đặc trưng HOG
├── Local Binary Patterns/       # Đặc trưng kết cấu LBP
├── Local Binary Patterns + Color Histogram/  # Đặc trưng kết hợp
└── Demo/                        # Ứng dụng web
    ├── backend.py               # Flask API server
    ├── index.html               # Frontend entry point
    └── models/                  # Các file mô hình đã train
```

## 🚀 Bắt đầu

### Yêu cầu cài đặt

```bash
pip install torch torchvision
pip install transformers
pip install flask flask-cors
pip install pillow numpy pandas
pip install scikit-learn joblib
pip install tqdm
```

### Chạy Demo

1. **Khởi động Flask backend:**
```bash
cd Demo
python backend.py
```

2. **Truy cập giao diện web:**
   - Mở trình duyệt và truy cập URL được cung cấp
   - Upload ảnh món ăn Việt Nam
   - Chọn mô hình ưa thích (CLIP ViT-L/14, DinoV2, hoặc ResNet50)
   - Nhận kết quả dự đoán ngay lập tức kèm thông tin món ăn

### Huấn luyện mô hình

Mỗi thư mục mô hình chứa Jupyter notebooks cho:
1. Trích xuất đặc trưng từ 30VNFoods dataset
2. Huấn luyện SVM classifier trên embeddings đã trích xuất
3. Đánh giá mô hình và các chỉ số hiệu suất

Quy trình ví dụ:
```bash
# Di chuyển đến thư mục mô hình mong muốn
cd ResNet50

# Chạy notebook trích xuất embedding
jupyter notebook 30vnfoods-resnet50-embedding.ipynb
```

## 🎯 Hiệu suất mô hình

Dự án triển khai và so sánh nhiều phương pháp: 

| Phương pháp | Loại | Ưu điểm chính |
|-------------|------|---------------|
| **CLIP ViT-L/14** | Vision-Language | Khả năng zero-shot mạnh, hiểu ngữ nghĩa tốt |
| **DinoV2** | Self-supervised ViT | Chất lượng đặc trưng xuất sắc không cần nhãn |
| **ResNet50** | CNN | Baseline vững chắc, inference hiệu quả |
| **Color Histogram** | Truyền thống | Nhanh, phân biệt dựa trên màu sắc |
| **HOG** | Truyền thống | Phát hiện hình dạng và cạnh |
| **LBP** | Truyền thống | Phân tích kết cấu |

## 🛠️ Công nghệ sử dụng

- **Deep Learning Frameworks**: PyTorch, TensorFlow
- **Mô hình Pre-trained**: Hugging Face Transformers, PyTorch Hub
- **Machine Learning**: Scikit-learn (SVM classifiers)
- **Web Framework**: Flask, React
- **Xử lý ảnh**: PIL, OpenCV
- **Xử lý dữ liệu**: NumPy, Pandas

## 📈 Cải tiến tương lai

- [ ] Mở rộng thêm các món ăn Việt Nam
- [ ] Hỗ trợ đa ngôn ngữ (Việt, Anh)
- [ ] Phân loại video real-time
- [ ] Triển khai ứng dụng di động
- [ ] Tích hợp thông tin dinh dưỡng
- [ ] Gợi ý công thức nấu ăn
- [ ] Tìm quán ăn gần đây có món đó

## 🤝 Đóng góp

Rất hoan nghênh các đóng góp!  Bạn có thể:
- Báo cáo lỗi
- Đề xuất tính năng mới
- Cải thiện tài liệu
- Thêm hỗ trợ cho nhiều món ăn hơn

## 📄 License

Dự án này là mã nguồn mở và có sẵn theo [MIT License](LICENSE).

## 👥 Tác giả

**ToiLaKiet** - [GitHub Profile](https://github.com/ToiLaKiet)

## 🙏 Lời cảm ơn

- Người tạo **30VNFoods Dataset** đã cung cấp dữ liệu huấn luyện
- **OpenAI** cho mô hình CLIP
- **Meta AI** cho DinoV2
- **Microsoft** cho ResNet50
- Cộng đồng mã nguồn mở cho các công cụ và thư viện

---

⭐ Nếu bạn thấy dự án hữu ích, hãy cho một star nhé! 

🍲 Được xây dựng với ❤️ dành cho ẩm thực Việt Nam
