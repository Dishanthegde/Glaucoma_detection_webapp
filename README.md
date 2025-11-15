👁️ Glaucoma Detection Web App
AI-Powered Retinal Image Analysis using CNN + VGG16

This project is an AI-based web application that detects glaucoma from retinal fundus images.
Using a hybrid CNN + VGG16 deep learning model, the system predicts whether the uploaded image shows Glaucomatous signs with high accuracy and generates a downloadable PDF medical-style report.

🚀 Features
🔍 1. AI-Based Detection

Hybrid CNN + VGG16 model

Fine-tuned on glaucoma retinal datasets

Outputs:

Prediction (Glaucoma / Normal)

Confidence score

Probability distribution bar chart

🖼️ 2. Image Upload System

Upload retina images in JPG/PNG format

Automatic pre-processing (resize, normalize)

📝 3. Auto-Generated PDF Report

Includes:

Uploaded image

Diagnosis result

Confidence

Probability bar chart

AI-generated physician-style note

🌐 4. Clean Web Interface (Streamlit)

User-friendly

Responsive

Works on mobile & desktop

🧠 Model Architecture
Base Model

Pre-trained VGG16 (ImageNet weights)

Used as feature extractor

Custom CNN Layers

Dense layers + Dropout

Softmax output for 2 classes

Training Setup

Epochs: 25

Optimizer: Adam

Loss: Categorical Crossentropy

Dataset Split:

70% Training

20% Validation

10% Testing

🗂️ Project Structure
📦 Glaucoma_detection_webapp
│
├── app.py                        # Main Streamlit application
├── combine_cnn_model_finetuned.keras   # Trained model (tracked using Git LFS)
├── requirements.txt              # Python dependencies
├── style.css                     # Custom styles for UI
├── README.md                     # Project documentation
├── images/                       # Assets for UI
└── reports/                      # Auto-generated PDF reports

💻 Installation & Usage
🔧 1. Clone the Repository
git clone https://github.com/Dishanthegde/Glaucoma_detection_webapp.git
cd Glaucoma_detection_webapp

📦 2. Install Dependencies
pip install -r requirements.txt

▶️ 3. Run the App
streamlit run app.py

📘 How It Works (Pipeline)

User uploads a retina image

Image is preprocessed (resize 256×256, normalize 0-1)

Model performs prediction using CNN + VGG16

App visualizes results + probabilities

User downloads an AI-generated PDF report

📄 Example Output (PDF Report)

The report contains:

Patient retina image

Prediction

Confidence score

Probability bar chart

AI-generated medical warning

Timestamp

📊 Sample Bar Chart

The probability visualization shows confidence for:

Normal

Glaucoma

Helps users understand how certain the model is.

🌍 Deployment

This project can be deployed using:

✔️ Streamlit Cloud (Easy and Free)

Just push code to GitHub → Deploy with one click at
https://streamlit.io/cloud

✔️ Local Deployment

Run streamlit run app.py

✔️ Docker (Optional)

Containerize for hospital use or offline devices.
