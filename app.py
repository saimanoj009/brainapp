import streamlit as st
import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os
import base64
import time

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ─── Load model & classes ────────────────────────────────────────────────────
model = load_model("brain_tumor_model.keras")
model.predict(np.zeros((1, 224, 224, 3)))

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ─── Background image helper ─────────────────────────────────────────────────
def get_base64_image(img_path):
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_DIR, "assets", "bg.jpg")
img_base64 = get_base64_image(IMG_PATH)

# ─── Prediction helpers ───────────────────────────────────────────────────────
def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.reshape(img, (1, 224, 224, 3))

def predict_image(img_path):
    pred = model.predict(preprocess(img_path))[0]
    return class_names[np.argmax(pred)], float(np.max(pred))

def predict_full(img_path):
    return model.predict(preprocess(img_path))[0]

# ─── Grad-CAM ────────────────────────────────────────────────────────────────
def generate_gradcam(img_path, model):
    base_model = model.layers[0]
    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    if last_conv_layer is None:
        return img

    img_norm = img / 255.0
    input_tensor = tf.convert_to_tensor(np.expand_dims(img_norm, axis=0), dtype=tf.float32)
    conv_model = tf.keras.Model(inputs=base_model.input, outputs=last_conv_layer.output)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(input_tensor)
        preds = model(input_tensor)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return img

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.5, heatmap, 0.8, 0)

# ─── Bounding box ─────────────────────────────────────────────────────────────
def draw_bounding_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img

# ─── Enhanced PDF report ──────────────────────────────────────────────────────
def generate_pdf(label, conf, probs, original_path, heatmap_img):
    pdf_path = "brain_tumor_report.pdf"
    doc = SimpleDocTemplate(pdf_path, leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                 fontSize=22, alignment=TA_CENTER,
                                 textColor=colors.HexColor("#1a1a2e"),
                                 spaceAfter=4)
    subtitle_style = ParagraphStyle("Sub", parent=styles["Normal"],
                                    fontSize=11, alignment=TA_CENTER,
                                    textColor=colors.HexColor("#6b7280"),
                                    spaceAfter=16)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"],
                                   fontSize=13, textColor=colors.HexColor("#374151"),
                                   spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                fontSize=11, textColor=colors.HexColor("#374151"),
                                spaceAfter=4)

    elements = []

    elements.append(Paragraph("Brain Tumor Detection Report", title_style))
    elements.append(Paragraph("AI-powered MRI analysis · Grad-CAM visualization", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb"), spaceAfter=16))

    is_tumor = label.lower() != "notumor"
    result_color = colors.HexColor("#dc2626") if is_tumor else colors.HexColor("#16a34a")
    result_bg = colors.HexColor("#fef2f2") if is_tumor else colors.HexColor("#f0fdf4")
    result_label = label.upper()

    result_data = [
        [Paragraph(f"<b>Prediction</b>", body_style), Paragraph(f"<b>Confidence</b>", body_style)],
        [Paragraph(f'<font color="#{("dc2626" if is_tumor else "16a34a")}" size="14"><b>{result_label}</b></font>', body_style),
         Paragraph(f'<font size="14"><b>{conf*100:.2f}%</b></font>', body_style)],
    ]
    result_table = Table(result_data, colWidths=[3*inch, 3*inch])
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), result_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#6b7280")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [result_bg]),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("BOX", (0, 0), (-1, -1), 1, result_color),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("Class Probabilities", section_style))

    class_colors = {
        "glioma":      colors.HexColor("#dc2626"),
        "meningioma":  colors.HexColor("#d97706"),
        "notumor":     colors.HexColor("#16a34a"),
        "pituitary":   colors.HexColor("#7c3aed"),
    }
    breakdown_data = [["Class", "Probability", "Confidence"]]
    for i, p in enumerate(probs):
        name = class_names[i]
        bar_chars = int(p * 20)
        bar = "█" * bar_chars + "░" * (20 - bar_chars)
        breakdown_data.append([name.capitalize(), bar, f"{p:.4f}"])

    breakdown_table = Table(breakdown_data, colWidths=[1.5*inch, 3.0*inch, 1.5*inch])
    breakdown_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#374151")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("ALIGN", (2, 0), (2, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(breakdown_table)
    elements.append(Spacer(1, 16))

    heatmap_path = "heatmap.jpg"
    cv2.imwrite(heatmap_path, heatmap_img)

    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb"), spaceAfter=12))
    elements.append(Paragraph("MRI Scan Images", section_style))

    img_data = [[
        RLImage(original_path, width=2.8*inch, height=2.8*inch),
        RLImage(heatmap_path, width=2.8*inch, height=2.8*inch),
    ], [
        Paragraph("<b>Original MRI</b>", ParagraphStyle("cap", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10, textColor=colors.HexColor("#6b7280"))),
        Paragraph("<b>Grad-CAM Heatmap</b>", ParagraphStyle("cap", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10, textColor=colors.HexColor("#6b7280"))),
    ]]
    img_table = Table(img_data, colWidths=[3*inch, 3*inch])
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 20))

    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb"), spaceAfter=8))
    elements.append(Paragraph(
        "⚠️ This report is generated by an AI model and is intended for informational purposes only. "
        "It does not constitute a medical diagnosis. Please consult a qualified medical professional.",
        ParagraphStyle("Warn", parent=styles["Normal"], fontSize=9,
                       textColor=colors.HexColor("#92400e"),
                       backColor=colors.HexColor("#fffbeb"),
                       borderPadding=(6, 8, 6, 8),
                       borderRadius=4)
    ))

    doc.build(elements)
    return pdf_path

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# ─── CSS: full animated redesign ──────────────────────────────────────────────
bg_css = f"""
.stApp {{
    background-image: url("data:image/jpg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
""" if img_base64 else ""

st.markdown(f"""
<style>
/* ── Reset & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

{bg_css}

.stApp {{
    background-color: #080c14;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
}}

/* Overlay */
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(4, 8, 20, 0.82);
    backdrop-filter: blur(2px);
    z-index: -1;
}}

/* ── Animated stars bg ── */
.stApp::after {{
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(1px 1px at 20% 15%, rgba(255,255,255,0.35) 0%, transparent 100%),
        radial-gradient(1px 1px at 70% 30%, rgba(255,255,255,0.25) 0%, transparent 100%),
        radial-gradient(1px 1px at 40% 60%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 80% 75%, rgba(255,255,255,0.2) 0%, transparent 100%),
        radial-gradient(1px 1px at 10% 80%, rgba(255,255,255,0.25) 0%, transparent 100%),
        radial-gradient(1px 1px at 55% 90%, rgba(255,255,255,0.2) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 10%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 30% 45%, rgba(255,255,255,0.15) 0%, transparent 100%);
    z-index: -1;
    animation: twinkle 6s ease-in-out infinite alternate;
}}

@keyframes twinkle {{
    0%  {{ opacity: 0.4; }}
    100%{{ opacity: 1;   }}
}}

/* ── Main block ── */
.block-container {{
    background: rgba(10, 15, 30, 0.75) !important;
    border: 1px solid rgba(99, 179, 237, 0.12) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(16px) !important;
    padding: 2.5rem 2rem !important;
    box-shadow: 0 0 60px rgba(99, 179, 237, 0.06), 0 0 120px rgba(129, 140, 248, 0.04);
    animation: fadeUp 0.7s ease both;
}}

@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* ── Typography ── */
h1, h2, h3, h4, p, label, span, div {{ color: #e2e8f0 !important; }}

/* ── Upload area ── */
[data-testid="stFileUploader"] {{
    border: 1.5px dashed rgba(99,179,237,0.4) !important;
    border-radius: 14px !important;
    padding: 18px !important;
    background: rgba(99,179,237,0.04) !important;
    transition: border-color 0.3s, box-shadow 0.3s;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: rgba(99,179,237,0.8) !important;
    box-shadow: 0 0 20px rgba(99,179,237,0.15);
}}
[data-testid="stFileUploader"] section {{ background: transparent !important; }}
[data-testid="stFileUploader"] button {{
    background: linear-gradient(135deg, #1d4ed8, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: transform 0.2s, box-shadow 0.2s;
}}
[data-testid="stFileUploader"] button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(99,179,237,0.4);
}}

/* ── Images ── */
img {{
    border-radius: 12px !important;
    transition: transform 0.3s, box-shadow 0.3s !important;
}}
img:hover {{
    transform: scale(1.03) !important;
    box-shadow: 0 0 30px rgba(99,179,237,0.4) !important;
}}

/* ── Alerts ── */
.stSuccess {{ background: rgba(16,185,129,0.1) !important; border: 1px solid rgba(16,185,129,0.3) !important; border-radius: 10px !important; color: #6ee7b7 !important; }}
.stError   {{ background: rgba(239,68,68,0.1)  !important; border: 1px solid rgba(239,68,68,0.3)  !important; border-radius: 10px !important; color: #fca5a5 !important; }}
.stWarning {{ background: rgba(245,158,11,0.1) !important; border: 1px solid rgba(245,158,11,0.3) !important; border-radius: 10px !important; color: #fde68a !important; }}

/* ── Download button ── */
.stDownloadButton > button {{
    background: linear-gradient(135deg, #1d4ed8 0%, #6366f1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.02em;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35);
    width: 100% !important;
}}
.stDownloadButton > button:hover {{
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.5) !important;
}}

/* ── Glass card ── */
.glass-card {{
    background: rgba(255,255,255,0.04);
    border: 0.5px solid rgba(99,179,237,0.2);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 14px;
    animation: fadeUp 0.6s ease both;
}}
.glass-card h3 {{
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(148,163,184,0.8) !important;
    margin-bottom: 10px;
}}

/* ── Confidence display ── */
.conf-number {{
    font-size: 52px;
    font-weight: 600;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}}
.conf-bar-track {{
    height: 4px;
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    margin-top: 10px;
    overflow: hidden;
}}
.conf-bar-fill {{
    height: 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    animation: barGrow 1.2s cubic-bezier(0.34,1.56,0.64,1) both;
}}
@keyframes barGrow {{
    from {{ width: 0 !important; }}
}}

/* ── Class breakdown bars ── */
.bar-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    animation: fadeUp 0.5s ease both;
}}
.bar-class {{
    font-size: 13px;
    font-weight: 500;
    width: 100px;
    color: #94a3b8 !important;
    text-transform: capitalize;
}}
.bar-outer {{
    flex: 1;
    height: 5px;
    background: rgba(255,255,255,0.07);
    border-radius: 5px;
    overflow: hidden;
}}
.bar-inner {{
    height: 5px;
    border-radius: 5px;
    animation: barGrow 1s cubic-bezier(0.34,1.56,0.64,1) both;
}}
.bar-val {{
    font-size: 12px;
    color: #64748b !important;
    width: 36px;
    text-align: right;
}}

/* ── Pulse badge ── */
.pulse-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 100px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.05em;
}}
.pulse-badge.safe {{
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.35);
    color: #34d399 !important;
}}
.pulse-badge.danger {{
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.35);
    color: #f87171 !important;
}}
.pulse-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    animation: pulseDot 1.5s ease-in-out infinite;
}}
.pulse-dot.safe   {{ background: #34d399; box-shadow: 0 0 8px rgba(52,211,153,0.6); }}
.pulse-dot.danger {{ background: #f87171; box-shadow: 0 0 8px rgba(248,113,113,0.6); }}
@keyframes pulseDot {{
    0%, 100% {{ transform: scale(1);   opacity: 1; }}
    50%       {{ transform: scale(1.5); opacity: 0.6; }}
}}

/* ── Section header ── */
.section-label {{
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(148,163,184,0.6) !important;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.section-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.07);
}}

/* ── Spinner ── */
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.spinner {{
    width: 18px; height: 18px;
    border: 2px solid rgba(99,179,237,0.2);
    border-top-color: #38bdf8;
    border-radius: 50%;
    display: inline-block;
    animation: spin 0.8s linear infinite;
    vertical-align: middle;
    margin-right: 6px;
}}

/* ── Footer ── */
footer {{ visibility: hidden; }}
.footer-custom {{
    text-align: center;
    font-size: 12px;
    color: rgba(148,163,184,0.45) !important;
    margin-top: 36px;
    padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.05);
}}
.footer-custom a {{ color: rgba(99,179,237,0.7) !important; text-decoration: none; }}
.footer-custom a:hover {{ color: #38bdf8 !important; }}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 0.5rem 0 1.5rem;">
  <div style="font-size:48px; margin-bottom:6px;">🧠</div>
  <h1 style="font-size:32px; font-weight:600; margin:0;
             background:linear-gradient(90deg,#38bdf8,#818cf8,#c084fc);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    Brain Tumor Detection
  </h1>
  <p style="color:rgba(148,163,184,0.6)!important; font-size:14px; margin-top:6px;">
    AI-powered MRI analysis &nbsp;·&nbsp; Grad-CAM visualization
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Upload ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Upload MRI scan</div>', unsafe_allow_html=True)
file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
st.markdown(
    "<p style='text-align:center;font-size:12px;color:rgba(148,163,184,0.4)!important;"
    "margin-top:6px;'>Supported: JPG · PNG · JPEG</p>",
    unsafe_allow_html=True
)

# ─── Processing ───────────────────────────────────────────────────────────────
if file:
    img = Image.open(file)

    with st.spinner(""):
        st.markdown("<p style='color:#38bdf8!important;font-size:14px;text-align:center;'>"
                    "<span class='spinner'></span>Analysing MRI scan…</p>", unsafe_allow_html=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            img.save(temp.name)
            label, conf = predict_image(temp.name)
            probs = predict_full(temp.name)
            heatmap_img = generate_gradcam(temp.name, model)
            heatmap_img = draw_bounding_box(heatmap_img)
            temp_path = temp.name

    time.sleep(0.3)
    st.success("✅ Analysis complete")

    # Images row
    st.markdown('<div class="section-label" style="margin-top:1.2rem;">MRI images</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'><h3>🖼️ Original scan</h3></div>", unsafe_allow_html=True)
        st.image(img, use_container_width=True)
    with col2:
        st.markdown("<div class='glass-card'><h3>🔥 Grad-CAM heatmap</h3></div>", unsafe_allow_html=True)
        st.image(heatmap_img, use_container_width=True, channels="BGR")

    # Prediction + confidence row
    st.markdown('<div class="section-label" style="margin-top:1.2rem;">Results</div>', unsafe_allow_html=True)
    res_col, conf_col = st.columns(2)

    is_tumor = label.lower() != "notumor"
    badge_cls = "danger" if is_tumor else "safe"
    badge_icon = "🚨" if is_tumor else "✅"

    with res_col:
        st.markdown(f"""
        <div class="glass-card">
          <h3>Prediction</h3>
          <div class="pulse-badge {badge_cls}">
            <span class="pulse-dot {badge_cls}"></span>
            {badge_icon} {label.upper()}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with conf_col:
        bar_w = int(conf * 100)
        st.markdown(f"""
        <div class="glass-card">
          <h3>Confidence</h3>
          <div class="conf-number">{conf*100:.1f}%</div>
          <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width:{bar_w}%;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Class probabilities
    st.markdown('<div class="section-label" style="margin-top:1.2rem;">Class probabilities</div>', unsafe_allow_html=True)
    class_colors_map = {
        "glioma":      ("#dc2626", "#fca5a5"),
        "meningioma":  ("#d97706", "#fde68a"),
        "notumor":     ("#16a34a", "#86efac"),
        "pituitary":   ("#7c3aed", "#c4b5fd"),
    }

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    for i, p in enumerate(probs):
        name = class_names[i]
        col_dark, col_light = class_colors_map.get(name, ("#64748b", "#94a3b8"))
        pct = int(p * 100)
        delay = 0.1 * i
        st.markdown(f"""
        <div class="bar-row" style="animation-delay:{delay}s;">
          <span class="bar-class">{name.capitalize()}</span>
          <div class="bar-outer">
            <div class="bar-inner" style="width:{pct}%; background:linear-gradient(90deg,{col_dark},{col_light}); animation-delay:{delay}s;"></div>
          </div>
          <span class="bar-val">{p:.3f}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.warning("⚠️ AI-based result only — not a medical diagnosis. Consult a qualified physician.")

    # PDF download
    st.markdown('<div class="section-label" style="margin-top:1rem;">Report</div>', unsafe_allow_html=True)
    pdf_file = generate_pdf(label, conf, probs, temp_path, heatmap_img)
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄  Download Full Report (PDF)",
            data=f,
            file_name="brain_tumor_report.pdf",
            mime="application/pdf",
        )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-custom">
  © 2026 Brain Tumor Detection System &nbsp;·&nbsp; Sai Manoj &nbsp;·&nbsp;
  <a href="mailto:saimanoj0914@gmail.com">saimanoj0914@gmail.com</a>
</div>
""", unsafe_allow_html=True)