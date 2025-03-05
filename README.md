# Real-Time Classification of Metal & Non-Metal Objects Using Deep Learning

## 📌 Overview
This project focuses on **real-time detection and classification** of **metal and non-metal objects** using **deep learning and computer vision**. It aims to enhance **waste segregation and recycling** efficiency by accurately identifying materials.

We integrate two models:
- **YOLOv8 (Object Detection Model):** Detects objects placed on a surface.
- **Custom CNN Model (Classification Model):** Categorizes objects into **Steel, Copper, Rusted, Iron, Aluminum, and Non-Metal**.






## 🚀 Features
- **Real-time object detection & classification**
- **YOLOv8 for precise object detection**
- **CNN-based classification into six categories**
- **Color-coded bounding boxes for visualization**





## 📊 Dataset
- **Metal category:** Images collected from manual photography and online sources (Flipkart, Meesho).
- **Non-Metal category:** Images sourced from **TrashNet (Kaggle)** and manual photography.
- **Dataset Size:**
  - **MVN Model:** 12,000 images (2,000 per class)
  - **OD Model:** 414 manually annotated images





## 🏗 Model Architecture
### **1️⃣ YOLOv8 (Object Detection Model)**
- Detects objects on a surface and provides location coordinates.
- Uses **LabelIMG** for annotation.
- **96% detection accuracy**.

### **2️⃣ CNN (Metal vs. Non-Metal Classification)**
- The CNN architecture was adapted from a reference model used for **vehicle color recognition**, modified to classify metal and non-metal objects.
- Classifies objects into **six categories** based on color information.
- **79% classification accuracy**.





## 🔄 Workflow
<p align="center">
  <img src="https://github.com/user-attachments/assets/73bdd9f3-ad26-4df9-9a58-bb0fcb48ed2f" width="600">
</p>


1. **Object Detection (YOLOv8):**
   - Capture **real-time video feed**.
   - YOLOv8 detects objects on the floor and provides bounding box coordinates.

2. **Object Cropping (OpenCV):**
   - Extract detected objects based on bounding box coordinates.
   - Preprocess images for classification.

3. **Object Classification (CNN Model):**
   - Classifies cropped objects into **Steel, Copper, Rusted, Iron, Aluminum, or Non-Metal**.
   - Uses a CNN model adapted from a reference vehicle color recognition model.

4. **Result Visualization:**
   - Assigns **color-coded bounding boxes** based on classification results.
   - Displays real-time classification output on screen.






## 📈 Results & Performance
<p align="center">
  <img src="https://github.com/user-attachments/assets/08aca8eb-bae9-45e2-a3f1-d46bdbc11f88" width="600">
</p>

- **YOLOv8 Model:** `96% accuracy` in object detection.
- **CNN Model:** `79% accuracy` in classification.
- **Challenges:** Similar-colored metals & lighting conditions impact classification.





## 🎥 Demo






## 🔮 Future Improvements
- Improve dataset to enhance accuracy.
- Optimize classification for similar-colored metals.
- Deploy as a **real-time edge AI system**.





## 📜 References
- [Reference Paper on CNN for Vehicle Color Recognition](https://www.researchgate.net/publication/283279784_Vehicle_Color_Recognition_using_Convolutional_Neural_Network)





## 📂 File Structure
```
metal-classification/
├── MVN_model/
│   ├── mean_image.png          # Mean image used for preprocessing
│   ├── MVN.py                  # Classification model script (CNN)
│   ├── MVN-35-0.7969.onnx      # Trained CNN model
│
├── Object_detection/
│   ├── OD.py                   # Object detection script (YOLOv8)
│   ├── best.onnx               # Trained YOLOv8 model
│
├── GUI.py                      # Main GUI application using Tkinter
├── icon.ico                     # Application icon
```





## ⚙️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/rajkajare/metal-classification.git
   cd Real-Time-Classification-of-Metal-And-Non-Metal-Objects-Using-Deep-Learning
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python GUI.py
   ```




---
💡 **Enhancing waste segregation with AI-driven metal classification.** 🚀
