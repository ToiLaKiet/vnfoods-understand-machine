import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import { model } from '@tensorflow/tfjs';
// ✅ Định nghĩa các model + endpoint tương ứng ở backend
const MODEL_DEFINITIONS = {
  resnet50: {
    key: 'resnet50',
    label: 'ResNet50',
    endpoint: 'http://127.0.0.1:5000/predict-resnet50'
  },
  vitl14: {
    key: 'vitl14',
    label: 'CLIP ViT-L14',
    endpoint: 'http://127.0.0.1:5000/predict-vitl14'
  },
  dinov2: {
    key: 'dinov2',
    label: 'DinoV2',
    endpoint: 'http://127.0.0.1:5000/predict-dinov2'
  }
};

export default function App() {
  const [classifying, setClassifying] = useState(false);
  const [image, setImage] = useState(null);           // dataURL của ảnh
  const [predictions, setPredictions] = useState([]);
  const [selectedFood, setSelectedFood] = useState(null);
  const [foodInfo, setFoodInfo] = useState({});
  const [error, setError] = useState(null);
  const [unknownMessage, setUnknownMessage] = useState('');
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({
    totalScans: 0,
    averageConfidence: 0,
    topFood: null
  });
  const [currentModelKey, setCurrentModelKey] = useState('resnet50');

  const fileInputRef = useRef(null);
  const imageRef = useRef(null);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = () => {
    const saved = localStorage.getItem('foodHistory');
    if (saved) {
      const parsedHistory = JSON.parse(saved);
      setHistory(parsedHistory);
      updateStats(parsedHistory);
    }
  };
  const handleChangeImage = () => {
    // xóa dữ liệu foodInfo trong localStorage (nếu có)
    localStorage.removeItem('foodInfo');
  
    // reset state liên quan
    setPredictions([]);
    setSelectedFood(null);
    setFoodInfo({});
    setUnknownMessage('');
    setError(null);
  
    // mở dialog chọn ảnh
    fileInputRef.current?.click();
  };
  

  const updateStats = (historyData) => {
    if (historyData.length === 0) {
      setStats({
        totalScans: 0,
        averageConfidence: 0,
        topFood: null
      });
      return;
    }

    const total = historyData.length;
    const avgConf =
      historyData.reduce((sum, item) => sum + parseFloat(item.confidence), 0) /
      total;

    const foodCounts = {};
    historyData.forEach((item) => {
      foodCounts[item.className] = (foodCounts[item.className] || 0) + 1;
    });

    const topFood = Object.keys(foodCounts).reduce((a, b) =>
      foodCounts[a] > foodCounts[b] ? a : b
    );

    setStats({
      totalScans: total,
      averageConfidence: avgConf.toFixed(1),
      topFood
    });
  };

  const saveToHistory = (prediction) => {
    const newEntry = {
      ...prediction,
      timestamp: new Date().toISOString(),
      model: currentModelKey
    };

    if (history.length > 10) {
      setHistory(history.slice(0, 9));
    }
    const newHistory = [newEntry, ...history].slice(0, 10);
    setHistory(newHistory);
    localStorage.setItem('foodHistory', JSON.stringify(newHistory));
    updateStats(newHistory);
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target.result); // dataURL
        setPredictions([]);
        setSelectedFood(null);
        setError(null);
        setUnknownMessage('');
      };
      reader.readAsDataURL(file);
    }
  };

  const loadFoodInfo = async (className) => {
    const fileName = className.toLowerCase().replace(/\s+/g, '_');
    console.log(fileName);
    try {
      const response = await fetch(`/food_info/${fileName}.txt`);
      if (response.ok) {
        const text = await response.text();
        return text;
      }
      return null;
    } catch (err) {
      console.log(`Không tìm thấy thông tin cho: ${className}`);
      return null;
    }
  };

  const classifyImage = async () => {
    if (!image || !imageRef.current) return;

    try {
      setClassifying(true);
      setError(null);
      setUnknownMessage('');
      setSelectedFood(null);

      const currentModel = MODEL_DEFINITIONS[currentModelKey];
      if (!currentModel) {
        console.log('currentModel', currentModelKey);
        setError('Mô hình không hợp lệ.');
        setClassifying(false);
        return;
      }

      // Gửi request lên API của model hiện tại
      // Ở đây mình giả sử backend nhận body:
      const im = image.split(',')[1];
      const response = await fetch(currentModel.endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'  
        },
        body: JSON.stringify({
          image: im,               // dataURL ảnh
          model: currentModel.key
        })
      });
      if (!response.ok) {
        throw new Error(`API trả về mã lỗi ${response.status}`);
      }

      // Giả sử API trả về mảng { className, confidence } với confidence là %
      const text = await response.text();
      const json = JSON.parse(text);

    //   if (!rawPredictions || rawPredictions.length === 0) {
    //     setPredictions([]);
    //     setUnknownMessage(
    //       'Món ăn này nằm ngoài tầm hiểu biết của hệ thống.'
    //     );
    //     setClassifying(false);
    //     return;
    //   }

    //  // rawPredictions: { "Banh beo": 0.238, "Banh xeo": 0.52, ... }

    // // 1) Convert object → array [{ className, confidence }, ...]
    // const predictionsArray = Object.entries(rawPredictions).map(
    //   ([className, prob]) => ({
    //     className,
    //     confidence: Number((prob * 100).toFixed(2)) // convert 0.238 → 23.8 (%)
    //   })
    // );

    // // 2) Sort giảm dần theo confidence
    // const sorted = predictionsArray.sort(
    //   (a, b) => b.confidence - a.confidence
    // );

    // const top = sorted[0];

    // if (!top || top.confidence < 0) {
    //   setPredictions([]);
    //   setUnknownMessage('Món ăn này nằm ngoài tầm hiểu biết của hệ thống.');
    //   setClassifying(false);
    //   return;
    // }

    // // 3) Lọc các món có confidence >= 30%
    // const filtered = sorted.filter((p) => p.confidence >= 10);

    // // 4) Load info cho từng món
    // const infoPromises = filtered.map((pred) =>
    //   loadFoodInfo(pred.className).then((info) => ({
    //     className: pred.className,
    //     confidence: pred.confidence,
    //     info
    //   }))
    // );

    // // 5) Chờ tất cả xong rồi set state
    // const results = await Promise.all(infoPromises);
    // setPredictions(results);
    // setClassifying(false);


    //   const foodInfoResults = await Promise.all(infoPromises);
    //   const newFoodInfo = {};
    //   foodInfoResults.forEach(({ className, info }) => {
    //     if (info) {
    //       newFoodInfo[className] = info;
    //     }
    //   });

    //   setFoodInfo(newFoodInfo);
    //   setPredictions(filtered);

    //   if (filtered.length > 0) {
    //     saveToHistory(filtered[0]);
    //   }

    //   setClassifying(false);
    const rawPredictions = json['Probabilities'];
    console.log('Raw predictions:', rawPredictions);

    if (!rawPredictions || Object.keys(rawPredictions).length === 0) {
      setPredictions([]);
      setUnknownMessage('Món ăn này nằm ngoài tầm hiểu biết của hệ thống.');
      setClassifying(false);
      return;
    }

    // 1) Convert object → array [{ className, confidence }, ...]
    const predictionsArray = Object.entries(rawPredictions).map(
      ([className, prob]) => ({
        className,
        confidence: Number((prob * 100).toFixed(2)) // 0.238 → 23.8 (%)
      })
    );

    // 2) Sort giảm dần theo confidence
    const sorted = predictionsArray.sort((a, b) => b.confidence - a.confidence);

    // 3) Lấy top-1
    const top = sorted[0];

    // Nếu top-1 quá thấp thì coi như không biết
    if (!top || top.confidence < 10) { // ngưỡng 10%, tuỳ bạn chỉnh
      setPredictions([]);
      setUnknownMessage('Món ăn này nằm ngoài tầm hiểu biết của hệ thống.');
      setClassifying(false);
      return;
    }

    // 4) Lấy top-5 để hiển thị
    const top5 = sorted.slice(0, 5);

    // 5) Chỉ load info cho top-1
    const topInfoText = await loadFoodInfo(top.className);

    const newFoodInfo = {};
    if (topInfoText) {
      newFoodInfo[top.className] = topInfoText;
    }
    setFoodInfo(newFoodInfo);

    // 6) predictions: vẫn là mảng top-5 (không cần info trong từng phần tử)
    setPredictions(top5);

    // Lưu lịch sử với món top-1
    saveToHistory(top);

    setClassifying(false);
    } catch (err) {
      console.error('Helloooo:', err.message);
      setError('Lỗi khi phân loại ảnh. Vui lòng thử lại.');
      setClassifying(false);
    }
  };

  const handleViewDetails = (prediction) => {
    setSelectedFood(prediction);
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('foodHistory');
    setStats({
      totalScans: 0,
      averageConfidence: 0,
      topFood: null
    });
  };

  const currentModelLabel =
    MODEL_DEFINITIONS[currentModelKey]?.label || 'Model';

  return (
    <div className="ic-container">
      {/* Header Section */}
      <div className="ic-topbar">
        <div className="ic-logo">
          <span className="ic-logo-icon">🍜</span>
          <span className="ic-logo-text">
            UIT | CS231 | Hệ thống phân loại món ăn Việt Nam
          </span>
        </div>
        <div className="ic-model-info">
          <span className="ic-model-badge">
            {classifying ? '🔍 Đang phân tích...' : `✓ ${currentModelLabel}`}
          </span>
        </div>
      </div>

      <div className="ic-main">
        {/* Stats Cards */}
        <div className="ic-stats-grid">
          <div className="ic-stat-card">
            <div className="ic-stat-icon">📊</div>
            <div className="ic-stat-value">{stats.totalScans}</div>
            <div className="ic-stat-label">Tổng số lượt quét</div>
          </div>
          <div className="ic-stat-card">
            <div className="ic-stat-icon">🎯</div>
            <div className="ic-stat-value">{stats.averageConfidence}%</div>
            <div className="ic-stat-label">Độ tin cậy Trung Bình</div>
          </div>
          <div className="ic-stat-card">
            <div className="ic-stat-icon">🏆</div>
            <div className="ic-stat-value">{stats.topFood || '-'}</div>
            <div className="ic-stat-label">Món ăn xuất hiện nhiều nhất</div>
          </div>
        </div>

        <div className="ic-columns">
          {/* Left Column - Upload & Results */}
          <div className="ic-left">
            <div className="ic-card">
              <div className="ic-card-header">
                <h2 className="ic-card-title">📸 Tải ảnh lên</h2>

                {/* Chọn mô hình */}
                <select
                  className="ic-model-select"
                  value={currentModelKey}
                  onChange={(e) => {
                    setCurrentModelKey(e.target.value);
                    setPredictions([]);
                    setUnknownMessage('');
                    setError(null);
                  }}
                >
                  {Object.values(MODEL_DEFINITIONS).map((m) => (
                    <option key={m.key} value={m.key}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </div>

              {error && <div className="ic-error">{error}</div>}

              <>
                <div className="ic-upload-section">
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className={
                      'ic-upload-box ' + (!image ? 'ic-upload-box--empty' : '')
                    }
                  >
                    {image ? (
                      <div className="ic-image-container">
                        <img
                          ref={imageRef}
                          src={image}
                          alt="Uploaded"
                          className="ic-uploaded-image"
                          crossOrigin="anonymous"
                        />
                      </div>
                    ) : (
                      <div className="ic-upload-prompt">
                        <div className="ic-upload-icon">📤</div>
                        <p className="ic-upload-text">
                          Nhấp để tải ảnh món ăn
                        </p>
                        <p className="ic-upload-hint">PNG, JPG, JPEG</p>
                      </div>
                    )}
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="ic-hidden-input"
                  />
                </div>

                {image && (
                  <div className="ic-button-group">
                    <button
                      onClick={handleChangeImage}
                      className="ic-btn ic-btn--change"
                    >
                      📷 Đổi ảnh
                    </button>
                    {!classifying &&
                      predictions.length === 0 &&
                      !unknownMessage && (
                        <button
                          onClick={classifyImage}
                          className="ic-btn ic-btn--classify"
                        >
                          🔍 Phân loại ngay
                        </button>
                      )}
                  </div>
                )}

                {classifying && (
                  <div className="ic-classifying">
                    <div className="ic-spinner"></div>
                    <p className="ic-classifying-text">
                      Đang phân tích món ăn...
                    </p>
                  </div>
                )}

                {/* Thông điệp ngoài tầm hiểu biết */}
                {unknownMessage && !classifying && (
                  <div className="ic-unknown-box">{unknownMessage}</div>
                )}
              </>
            </div>

            {/* Results Section */}
            {predictions.length > 0 && !classifying && (
              <div className="ic-card">
                <div className="ic-card-header">
                  <h2 className="ic-card-title">🎯 Kết quả phân loại</h2>
                  <div className="ic-result-count">
                    {predictions.length} kết quả
                  </div>
                </div>
                <div className="ic-predictions">
                  {predictions.map((pred, index) => (
                    <div key={index} className="ic-prediction-item">
                      <div className="ic-prediction-rank">{index + 1}</div>
                      <div className="ic-prediction-content">
                        <div className="ic-prediction-header">
                          <span className="ic-prediction-label">
                            {pred.className}
                          </span>
                          <span className="ic-prediction-score">
                            {pred.confidence}%
                          </span>
                        </div>
                        <div className="ic-progress-bg">
                          <div
                            className="ic-progress-fill"
                            style={{ width: `${pred.confidence}%` }}
                          />
                        </div>
                        {foodInfo[pred.className] && (
                          <button
                            onClick={() => handleViewDetails(pred)}
                            className="ic-btn ic-btn--detail"
                          >
                            📖 Chi tiết
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Column - History + Tips */}
          <div className="ic-right">
            {/* History Card */}
            <div className="ic-card">
              <div className="ic-card-header">
                <h2 className="ic-card-title">📜 Lịch sử quét</h2>
                {history.length > 0 && (
                  <button
                    onClick={clearHistory}
                    className="ic-btn ic-btn--clear"
                  >
                    🗑️ Xóa
                  </button>
                )}
              </div>
              <div className="ic-history-list">
                {history.length === 0 ? (
                  <div className="ic-empty-history">
                    <div className="ic-empty-icon">📭</div>
                    <p className="ic-empty-text">Chưa có lịch sử quét</p>
                  </div>
                ) : (
                  history.map((item, index) => (
                    <div key={index} className="ic-history-item">
                      {/* <img
                        src={item.image}
                        alt=""
                        className="ic-history-thumb"
                      /> */}
                      <div className="ic-history-content">
                        <div className="ic-history-title">
                          {item.className}
                        </div>
                        <div className="ic-history-meta">
                          <span className="ic-history-conf">
                            {item.confidence}%
                          </span>
                          <span className="ic-history-conf">
                            {item.model}
                          </span>
                          <span className="ic-history-time">
                            {new Date(item.timestamp).toLocaleTimeString(
                              'vi-VN'
                            )}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Tips Card */}
            <div className="ic-card">
              <div className="ic-card-header">
                <h2 className="ic-card-title">💡 Mẹo sử dụng</h2>
              </div>
              <div className="ic-tips">
                <div className="ic-tip-item">
                  <span className="ic-tip-icon">📷</span>
                  <span className="ic-tip-text">
                    Chụp ảnh rõ nét, ánh sáng tốt
                  </span>
                </div>
                <div className="ic-tip-item">
                  <span className="ic-tip-icon">🍽️</span>
                  <span className="ic-tip-text">
                    Món ăn chiếm phần lớn khung hình
                  </span>
                </div>
                <div className="ic-tip-item">
                  <span className="ic-tip-icon">🎯</span>
                  <span className="ic-tip-text">
                    Tránh chụp nhiều món cùng lúc
                  </span>
                </div>
                <div className="ic-tip-item">
                  <span className="ic-tip-icon">✨</span>
                  <span className="ic-tip-text">
                    Góc chụp từ trên xuống cho kết quả tốt nhất
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Detail Modal */}
      {selectedFood && foodInfo[selectedFood.className] && (
        <div
          className="ic-detail-modal"
          onClick={() => setSelectedFood(null)}
        >
          <div
            className="ic-detail-card"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setSelectedFood(null)}
              className="ic-close-btn"
            >
              ✕
            </button>
            <h2 className="ic-detail-title">
              🍽️ {selectedFood.className}
            </h2>
            <div className="ic-detail-content">
              <div className="ic-confidence-badge">
                Độ chính xác: {selectedFood.confidence}%
              </div>
              <div className="ic-detail-text">
                {foodInfo[selectedFood.className]}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="ic-footer">
        <p>HIIIIIII, chúng tôi là chuyên gia ẩm thực !!</p>
      </div>
    </div>
  );
}
