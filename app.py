import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import datetime
import os
import requests
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.platypus import Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib.units import inch

# -----------------------------------------------------------
# STREAMLIT CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Glaucoma Detector",
    page_icon="👁️",
    layout="wide"
)

# -----------------------------------------------------------
# CSS STYLING
# -----------------------------------------------------------
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #001f3f, #004d66, #00796b);
            color: white;
            background-attachment: fixed;
        }
        .title {
            text-align: center;
            font-size: 2.6rem;
            font-weight: 700;
            color: #e0f7fa;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            color: #b2ebf2;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }
        .result-card {
            background: rgba(255, 255, 255, 0.12);
            padding: 1rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 0 20px rgba(0,0,0,0.25);
            margin-bottom: 0.8rem;
        }
        .footer {
            text-align: center;
            color: #b2ebf2;
            margin-top: 2rem;
            font-size: 0.9rem;
        }
        .stProgress > div > div > div {
            background-color: #26c6da;
        }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# MODEL AUTO-DOWNLOAD
# -----------------------------------------------------------
MODEL_PATH = "combine_cnn_model_finetuned.keras"
MODEL_URL = st.secrets.get("MODEL_URL")   # must be defined in Streamlit Cloud Secrets

if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model... please wait.")
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)

# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------
st.sidebar.header("📊 Model & Dataset Info")
st.sidebar.markdown("""
**Model:** VGG16 + CNN Hybrid  
**Input Image Size:** 256×256 px  
**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  
**Accuracy:** ~87–92%
""")

st.sidebar.markdown("---")
st.sidebar.header("🧩 Dataset Summary")
st.sidebar.markdown("""
- **Glaucoma:** 570 images  
- **Normal:** 622 images  
- **Split:** 70/20/10  
- **Source:** ACRIMA + RIM-ONE  
""")

st.sidebar.markdown("---")
st.sidebar.header("📘 About the Project")
st.sidebar.markdown("""
This app uses a **CNN + VGG16 model** to detect Glaucoma from retinal images.  
Developed as part of a **Final Year Project** for early detection of eye diseases.
""")


# -----------------------------------------------------------
# PDF GENERATION
# -----------------------------------------------------------
def generate_pdf_report(image, result, confidence, prediction, class_names):
    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, "glaucoma_report.pdf")
    
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title', parent=styles['Heading1'], alignment=1, textColor=colors.HexColor("#004d66"))
    normal_style = ParagraphStyle('normal', parent=styles['BodyText'], fontSize=12, leading=16)
    
    # Save retina image temp
    img_path = os.path.join(temp_dir, "retina_image_temp.jpg")
    image.save(img_path)
    
    story.append(Paragraph("AI Glaucoma Detection Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Uploaded Retina Image:</b>", normal_style))
    story.append(RLImage(img_path, width=4 * inch, height=4 * inch))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(f"<b>Prediction:</b> {result}", normal_style))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", normal_style))
    story.append(Spacer(1, 8))

    if result.lower() == "glaucoma":
        story.append(Paragraph(
            "⚠ The image shows features consistent with <b>Glaucoma</b>. Consultation with an ophthalmologist is recommended.",
            normal_style
        ))
    else:
        story.append(Paragraph(
            "✅ The retina image appears <b>Normal</b>. Regular checkups are advised.",
            normal_style
        ))

    story.append(Spacer(1, 20))

    data = [prediction[0].tolist()]
    drawing = Drawing(400, 200)
    bc = VerticalBarChart()
    bc.x = 50
    bc.y = 30
    bc.height = 125
    bc.width = 300
    bc.data = data
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = 1
    bc.categoryAxis.categoryNames = class_names
    bc.bars[0].fillColor = colors.HexColor("#26c6da")
    
    drawing.add(bc)
    story.append(Paragraph("<b>Prediction Confidence Chart:</b>", normal_style))
    story.append(drawing)

    story.append(Paragraph(
        "<i>Report generated by AI Glaucoma Detector | Dept. of CSE | Final Year Project 2025</i>",
        ParagraphStyle('footer', parent=styles['Normal'], alignment=1, fontSize=10, textColor=colors.gray)
    ))

    doc.build(story)
    return pdf_path


# -----------------------------------------------------------
# HEADER
# -----------------------------------------------------------
st.markdown("<h1 class='title'>👁️ Glaucoma Detection using CNN</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a retina image to detect Glaucoma or Normal Eye.</p>", unsafe_allow_html=True)


# -----------------------------------------------------------
# IMAGE UPLOAD + PREDICTION
# -----------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload retina image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((256, 256))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    prediction = model.predict(img_array)
    class_names = ["Glaucoma", "Normal"]
    confidence = float(np.max(prediction)) * 100
    result = class_names[np.argmax(prediction)]

    # Display result card
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:1.4rem;'>🧠 Prediction: <b>{result}</b></p>", unsafe_allow_html=True)
    st.progress(confidence / 100)
    st.markdown(f"**Confidence:** {confidence:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(image, caption="Uploaded Retina Image", width=350)

    with col2:
        st.write("### Actions:")
        
        if st.button("📄 Generate AI Report (PDF)"):
            with st.spinner("Generating report..."):
                pdf_path = generate_pdf_report(image, result, confidence, prediction, class_names)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Glaucoma Report",
                    data=f,
                    file_name="Glaucoma_Report.pdf",
                    mime="application/pdf"
                )
        
        if st.button("🔁 Try Another Image"):
            st.experimental_rerun()

else:
    st.info("👆 Upload a retina image to begin analysis.")


# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("<div class='footer'>Developed by <b>Hanumantha and Team</b> | Dept. of CSE | Final Year Project 2025 🌍</div>", unsafe_allow_html=True)
