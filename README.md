# 👁️ Glaucoma Detection Web App  
### AI-Powered Retinal Image Analysis using CNN + VGG16

This project is an AI-based web application that detects **glaucoma from retinal fundus images**.  
Using a hybrid **CNN + VGG16 deep learning model**, the system predicts whether the uploaded image shows **Glaucomatous** signs with high accuracy and generates a downloadable **PDF medical-style report**.

---

## 🚀 Features

### 🔍 **1. AI-Based Glaucoma Detection**
- Hybrid CNN + VGG16 model  
- Fine-tuned using retinal fundus dataset  
- Outputs:
  - **Classification (Normal / Glaucoma)**
  - **Confidence score**
  - **Probability distribution bar chart**

### 🖼️ **2. Retina Image Upload**
- JPG/PNG support  
- Automatic preprocessing  
  - Resize to 256×256  
  - Pixel normalization  

### 📝 **3. Auto-Generated PDF Report**
PDF includes:
- Uploaded retina image  
- Prediction result  
- Confidence  
- Probability bar chart  
- AI-generated medical note for the patient  

### 🌐 **4. Clean Web Interface (Streamlit)**
- User-friendly UI  
- Responsive  
- Mobile + desktop support  

---

## 🧠 Model Architecture

### **Base Model:**  
- Pre-trained **VGG16** (ImageNet) as the feature extractor

### **Fine-tuning:**  
- Last convolution layers retrained on glaucoma dataset  

### **Custom Layers Added:**  
- Dense layers  
- Dropout  
- Softmax output for 2 classes  

### **Training Setup:**  
- **Epochs:** 25  
- **Optimizer:** Adam  
- **Loss:** Categorical Crossentropy  
- **Dataset Split:** 70% train / 20% validation / 10% test  

---

## 📂 Project Structure

📦 Glaucoma_detection_webapp
│
├── app.py # Main Streamlit app
├── combine_cnn_model_finetuned.keras # Deep learning model (Git LFS)
├── style.css # UI styling
├── requirements.txt # Dependencies
├── README.md # Documentation
├── images/ # UI assets
└── reports/ # Generated PDF reports


---

## ⚙️ Installation

### 🔧 1. Clone the Repository
```bash
git clone https://github.com/Dishanthegde/Glaucoma_detection_webapp.git
cd Glaucoma_detection_webapp

pip install -r requirements.txt
streamlit run app.py

```

🔁 End-to-End Pipeline

User uploads retinal fundus image

Image is preprocessed

Resize 256×256

Normalize pixel values

AI model predicts:

Class (Normal / Glaucoma)

Probability & confidence

App displays result + bar chart

User downloads PDF medical report

📄 PDF Report Includes

✔ Uploaded retinal image
✔ AI prediction result
✔ Confidence percentage
✔ Probability chart
✔ AI-generated medical note
✔ Timestamp

⚠️ Medical Disclaimer

This tool is not a substitute for professional medical diagnosis.
It is designed to assist with preliminary screening, not replace an ophthalmologist.

👨‍💻 Developed By

Dishant Hegde
Final Year Project – Glaucoma Detection using AI

⭐ Support This Project

If you found this useful, please ⭐ star the GitHub repository!
