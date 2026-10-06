# 🍜 VNFoodsUnderstandMachine

[![VNFoodsUnderstandMachine Demo](https://img.youtube.com/vi/37cnGm0-b7o/maxresdefault.jpg)](https://youtu.be/37cnGm0-b7o?si=OlRrceHn4wkdTVxt)

Watch directly: [https://youtu.be/37cnGm0-b7o](https://youtu.be/37cnGm0-b7o?si=OlRrceHn4wkdTVxt)

A Vietnamese food classification system powered by deep learning and modern vision models. This project uses multiple pretrained models, including CLIP, DINOv2, and ResNet50, to accurately identify and provide information about 30 different Vietnamese dishes.

## 📋 Overview

VNFoodsDetector is designed to classify Vietnamese dishes from images and provide relevant information about them. The system explores various feature extraction methods—from traditional computer vision techniques to modern transformer models—to optimize classification performance.

## 🎯 Features

- **Multiple Models**: Implements several feature extraction methods:
  - **CLIP ViT-L/14**: A vision-language model with strong image understanding capabilities
  - **DINOv2**: A self-supervised vision transformer
  - **ResNet50**: A deep convolutional neural network
  - Traditional methods: Color Histogram, HOG, and Local Binary Patterns

- **30 Vietnamese Dishes**: Classifies a diverse range of dishes, including:
  - Bánh bèo, Bánh bột lọc, Bánh căn, Bánh canh, Bánh chưng, Bánh cuốn
  - Bánh đúc, Bánh giò, Bánh khọt, Bánh mì, Bánh pía, Bánh tét
  - Bánh tráng nướng, Bánh xèo
  - Bún bò Huế, Bún đậu mắm tôm, Bún mắm, Bún riêu, Bún thịt nướng
  - Cá kho tộ, Canh chua, Cao lầu, Cháo lòng
  - Cơm tấm, Gỏi cuốn, Hủ tiếu, Mì Quảng
  - Nem chua, Phở, Xôi xéo

- **Web Demo**: A Flask backend combined with a React frontend for real-time predictions

- **Dish Information**: Provides detailed information about each dish alongside classification results

## 📊 Dataset

The project uses the **30VNFoods dataset**, which includes:

- 30 Vietnamese food categories
- Images split into training, validation, and test sets
- Diverse images representing each food category

## 🏗️ Project Structure

```text
VNFoodsDetector/
├── CLIP-ViT-L: 14/              # CLIP model implementation and embeddings
├── DinoV2/                     # DINOv2 model implementation
├── ResNet50/                   # ResNet50 model implementation
├── Color Histogram/            # Traditional color-based features
├── Histogram Of Gradients/     # HOG feature extraction
├── Local Binary Patterns/      # LBP texture features
├── Local Binary Patterns + Color Histogram/  # Combined features
└── Demo/                       # Web application
    ├── backend.py              # Flask API server
    ├── index.html              # Frontend entry point
    └── models/                 # Trained model files
```

## 🚀 Getting Started

### Installation Requirements

```bash
pip install torch torchvision
pip install transformers
pip install flask flask-cors
pip install pillow numpy pandas
pip install scikit-learn joblib
pip install tqdm
```

### Running the Demo

1. **Start the Flask backend:**

   ```bash
   cd Demo
   python backend.py
   ```

2. **Access the web interface:**
   - Open your browser and navigate to the provided URL
   - Upload an image of a Vietnamese dish
   - Select your preferred model (CLIP ViT-L/14, DINOv2, or ResNet50)
   - Receive an instant prediction along with information about the dish

### Model Training

Each model directory contains Jupyter notebooks for:

1. Extracting features from the 30VNFoods dataset
2. Training an SVM classifier on the extracted embeddings
3. Evaluating the model and its performance metrics

Example workflow:

```bash
# Navigate to the desired model directory
cd ResNet50

# Run the embedding extraction notebook
jupyter notebook 30vnfoods-resnet50-embedding.ipynb
```

## 🎯 Model Performance

The project implements and compares several methods:

| Method | Type | Key Strengths |
|--------|------|---------------|
| **CLIP ViT-L/14** | Vision-Language | Strong zero-shot capabilities and semantic understanding |
| **DINOv2** | Self-supervised ViT | High-quality features learned without labels |
| **ResNet50** | CNN | Strong baseline and efficient inference |
| **Color Histogram** | Traditional | Fast, color-based discrimination |
| **HOG** | Traditional | Captures shapes and edges |
| **LBP** | Traditional | Texture analysis |

## 🛠️ Technology Stack

- **Deep Learning Frameworks**: PyTorch, TensorFlow
- **Pretrained Models**: Hugging Face Transformers, PyTorch Hub
- **Machine Learning**: Scikit-learn (SVM classifiers)
- **Web Frameworks**: Flask, React
- **Image Processing**: PIL, OpenCV
- **Data Processing**: NumPy, Pandas

## 📈 Future Improvements

- [ ] Expand support to more Vietnamese dishes
- [ ] Add multilingual support (Vietnamese and English)
- [ ] Implement real-time video classification
- [ ] Deploy a mobile application
- [ ] Integrate nutritional information
- [ ] Recommend recipes
- [ ] Find nearby restaurants serving the identified dish

## 🤝 Contributing

Contributions are welcome! You can:

- Report bugs
- Suggest new features
- Improve documentation
- Add support for more dishes

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👥 Author

**ToiLaKiet** — [GitHub Profile](https://github.com/ToiLaKiet)

## 🙏 Acknowledgments

- The creators of the **30VNFoods Dataset** for providing the training data
- **OpenAI** for CLIP
- **Meta AI** for DINOv2
- **Microsoft** for ResNet50
- The open-source community for its tools and libraries

---

⭐ If you find this project useful, please give it a star!

🍲 Built with ❤️ for Vietnamese cuisine
