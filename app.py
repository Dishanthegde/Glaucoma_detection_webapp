import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # kept for future use
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
    layout="wide",
)

# -----------------------------------------------------------
# THEME SWITCH (Dark / Light)
# -----------------------------------------------------------
theme = st.sidebar.selectbox("🎨 Theme", ["Dark", "Light"], index=0)

if theme == "Dark":
    css_bg = """
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #001f3f, #004d66, #00796b);
            color: white;
            background-attachment: fixed;
        }
        [data-testid="stHeader"] {
            background: transparent;
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
        .stButton > button {
            border-radius: 0.75rem;
            background: #0ea5e9;
            color: white;
            border: none;
            padding: 0.4rem 0.9rem;
            font-weight: 600;
        }
        .stButton > button:hover {
            background: #0284c7;
        }
    """
else:
    # Light theme: bright background, dark text, high contrast
    css_bg = """
        /* MAIN BACKGROUND & TEXT */
        [data-testid="stAppViewContainer"] {
            background: #f9fafb;
            color: #111827;
        }
        [data-testid="stHeader"] {
            background: #f9fafb;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }

        /* Ensure text in main area is dark */
        section.main p,
        section.main span,
        section.main label,
        section.main li,
        section.main h1,
        section.main h2,
        section.main h3 {
            color: #111827 !important;
        }

        .title {
            text-align: center;
            font-size: 2.6rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            text-align: center;
            color: #4b5563;
            margin-bottom: 1.2rem;
            font-size: 1.05rem;
        }

        .result-card {
            background: #ffffff;
            padding: 1rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(15,23,42,0.08);
            margin-bottom: 0.8rem;
            border: 1px solid #e5e7eb;
            color: #111827;
        }

        .footer {
            text-align: center;
            color: #6b7280;
            margin-top: 2rem;
            font-size: 0.9rem;
        }

        .stProgress > div > div > div {
            background-color: #2563eb;
        }

        .stButton > button {
            border-radius: 0.75rem;
            background: #2563eb;
            color: white;
            border: none;
            padding: 0.4rem 0.9rem;
            font-weight: 600;
        }
        .stButton > button:hover {
            background: #1d4ed8;
        }

        /* ---------- FILE UPLOADER STYLING ---------- */
        

    div[data-testid="stFileUploader"] label {
        color: #1e293b !important;      /* Dark gray */
        font-weight: 600 !important;
        font-size: 1rem !important;
    }


        /* Dropzone container */
        [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
            background-color: #eef2ff;          /* light indigo */
            border-radius: 0.9rem;
            border: 2px dashed #2563eb;
        }

        /* Text inside dropzone: "Drag and drop file here", size, etc. */
        [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] div {
            color: #111827 !important;
            font-weight: 500;
        }

        /* The "Browse files" button on the right */
        [data-testid="stFileUploader"] button {
            background-color: #2563eb !important;
            color: #ffffff !important;
            border-radius: 0.5rem !important;
            border: none !important;
            font-weight: 600 !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background-color: #1d4ed8 !important;
            color: #ffffff !important;
        }

        /* Info / warning boxes (st.info, st.error, etc.) */
        div.stAlert {
            background-color: #dbeafe !important;  /* light blue */
            color: #111827 !important;
            border-left: 4px solid #2563eb !important;
        }
        div.stAlert p,
        div.stAlert span {
            color: #111827 !important;
            font-weight: 500;
        }
    """

st.markdown(f"<style>{css_bg}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------
# MODEL AUTO-DOWNLOAD (for Streamlit Cloud)
# -----------------------------------------------------------
MODEL_PATH = "combine_cnn_model_finetuned.keras"

MODEL_URL = None
try:
    MODEL_URL = st.secrets.get("MODEL_URL", None)
except Exception:
    MODEL_URL = None

if not os.path.exists(MODEL_PATH):
    if MODEL_URL:
        st.info("📥 Downloading glaucoma model from remote storage... please wait.")
        try:
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()
    else:
        st.error(
            "❌ Model file not found and MODEL_URL is not set.\n\n"
            "If you are running locally, please place `combine_cnn_model_finetuned.keras` "
            "in the app directory.\nIf you are on Streamlit Cloud, set `MODEL_URL` in Secrets."
        )
        st.stop()

# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------------------------------------
# SIMPLE RETINA-LIKE CHECK
# -----------------------------------------------------------
def is_retina_like(pil_img, brightness_margin: float = 0.05) -> bool:
    """
    Heuristic to reject obvious non-retina images.
    - Fundus photos are roughly square.
    - Center tends to be brighter than corners.
    """
    w, h = pil_img.size
    ratio = max(w, h) / max(1, min(w, h))
    if ratio > 1.3:
        return False

    img = pil_img.resize((256, 256))
    arr = np.array(img).astype("float32") / 255.0
    gray = arr.mean(axis=2)

    H, W = gray.shape
    center = gray[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4]
    center_mean = float(center.mean())

    tl = gray[0 : H // 4, 0 : W // 4]
    tr = gray[0 : H // 4, 3 * W // 4 : W]
    bl = gray[3 * H // 4 : H, 0 : W // 4]
    br = gray[3 * H // 4 : H, 3 * W // 4 : W]
    corners = np.concatenate([tl.flatten(), tr.flatten(), bl.flatten(), br.flatten()])
    corner_mean = float(corners.mean())

    return center_mean > corner_mean + brightness_margin

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
    title_style = ParagraphStyle(
        "title",
        parent=styles["Heading1"],
        alignment=1,
        textColor=colors.HexColor("#004d66"),
    )
    normal_style = ParagraphStyle(
        "normal", parent=styles["BodyText"], fontSize=12, leading=16
    )

    img_path = os.path.join(temp_dir, "retina_image_temp.jpg")
    image.save(img_path)

    story.append(Paragraph("AI Glaucoma Detection Report", title_style))
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            f"<b>Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            normal_style,
        )
    )
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Uploaded Retina Image:</b>", normal_style))
    story.append(RLImage(img_path, width=4 * inch, height=4 * inch))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Prediction:</b> {result}", normal_style))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", normal_style))
    story.append(Spacer(1, 8))

    if result.lower() == "glaucoma":
        story.append(
            Paragraph(
                "The uploaded image shows features consistent with <b>Glaucoma</b>. "
                "It is recommended to consult an ophthalmologist for further evaluation.",
                normal_style,
            )
        )
    else:
        story.append(
            Paragraph(
                "The uploaded image appears <b>Normal</b>. "
                "However, regular comprehensive eye examinations are still advised.",
                normal_style,
            )
        )

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
    bc.valueAxis.valueStep = 0.2
    bc.categoryAxis.categoryNames = class_names
    bc.barWidth = 20
    bc.bars[0].fillColor = colors.HexColor("#26c6da")
    drawing.add(bc)

    story.append(Paragraph("<b>Prediction Confidence Chart:</b>", normal_style))
    story.append(drawing)
    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "<i>Report generated by AI Glaucoma Detector | "
            "Dept. of CSE | Final Year Project 2025</i>",
            ParagraphStyle(
                "footer",
                parent=styles["Normal"],
                alignment=1,
                fontSize=10,
                textColor=colors.gray,
            ),
        )
    )

    doc.build(story)
    return pdf_path

# -----------------------------------------------------------
# HEADER
# -----------------------------------------------------------
st.markdown(
    "<h1 class='title'>👁️ Glaucoma Detection using CNN</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='subtitle'>Upload a retina image to detect Glaucoma or Normal Eye.</p>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------
# IMAGE UPLOAD + PREDICTION
# -----------------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload retina image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Basic retina-like validation
    if not is_retina_like(image):
        st.error(
            "❌ The uploaded image does not appear to be a retinal (fundus) image.\n\n"
            "Please upload a valid eye fundus image for glaucoma analysis."
        )
        st.stop()

    img_resized = image.resize((256, 256))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    prediction = model.predict(img_array)
    class_names = ["Glaucoma", "Normal"]
    confidence = float(np.max(prediction)) * 100
    result = class_names[np.argmax(prediction)]

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:1.4rem;'>🧠 Prediction: <b>{result}</b></p>",
        unsafe_allow_html=True,
    )
    st.progress(confidence / 100)
    st.markdown(f"**Confidence:** {confidence:.2f}%", unsafe_allow_html=False)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(image, caption="Uploaded Retina Image", width=350)

    with col2:
        st.write("### Actions:")
        if st.button("📄 Generate AI Report (PDF)"):
            with st.spinner("Generating report..."):
                pdf_path = generate_pdf_report(
                    image, result, confidence, prediction, class_names
                )
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Glaucoma Report",
                    data=f,
                    file_name="Glaucoma_Report.pdf",
                    mime="application/pdf",
                )

        if st.button("🔁 Try Another Image"):
            st.experimental_rerun()

else:
    st.info("👆 Upload a retina image to begin analysis.")

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown(
    "<div class='footer'>Developed by <b>Hanumantha and Team</b> | "
    "Dept. of CSE | Final Year Project 2025 🌍</div>",
    unsafe_allow_html=True,
)
