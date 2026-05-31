cat > /mnt/user-data/outputs/app.py << 'PYEOF'
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
from reportlab.lib.enums import TA_CENTER

# ── Load model & classes ──────────────────────────────────────────────────────
model = load_model("brain_tumor_model.keras")
model.predict(np.zeros((1, 224, 224, 3)))

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ── Background helper ─────────────────────────────────────────────────────────
def get_base64_image(img_path):
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_DIR, "assets", "bg.jpg")
img_base64 = get_base64_image(IMG_PATH)

# ── Prediction helpers ────────────────────────────────────────────────────────
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

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
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

def draw_bounding_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img

# ── PDF ───────────────────────────────────────────────────────────────────────
def generate_pdf(label, conf, probs, original_path, heatmap_img):
    pdf_path = "brain_tumor_report.pdf"
    doc = SimpleDocTemplate(pdf_path, leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"], fontSize=22,
                                 alignment=TA_CENTER, textColor=colors.HexColor("#1a1a2e"), spaceAfter=4)
    sub_style = ParagraphStyle("S", parent=styles["Normal"], fontSize=11,
                               alignment=TA_CENTER, textColor=colors.HexColor("#6b7280"), spaceAfter=16)
    sec_style = ParagraphStyle("H", parent=styles["Heading2"], fontSize=13,
                               textColor=colors.HexColor("#374151"), spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("B", parent=styles["Normal"], fontSize=11,
                                textColor=colors.HexColor("#374151"), spaceAfter=4)

    elements = []
    elements.append(Paragraph("Brain Tumor Detection Report", title_style))
    elements.append(Paragraph("AI-powered MRI analysis · Grad-CAM visualization", sub_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb"), spaceAfter=16))

    is_tumor = label.lower() != "notumor"
    hex_color = "dc2626" if is_tumor else "16a34a"
    result_bg = colors.HexColor("#fef2f2") if is_tumor else colors.HexColor("#f0fdf4")
    result_border = colors.HexColor("#dc2626") if is_tumor else colors.HexColor("#16a34a")

    result_data = [
        [Paragraph("<b>Prediction</b>", body_style), Paragraph("<b>Confidence</b>", body_style)],
        [Paragraph(f'<font color="#{hex_color}" size="14"><b>{label.upper()}</b></font>', body_style),
         Paragraph(f'<font size="14"><b>{conf*100:.2f}%</b></font>', body_style)],
    ]
    rt = Table(result_data, colWidths=[3*inch, 3*inch])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), result_bg),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BOX", (0,0), (-1,-1), 1, result_border),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    elements.append(rt)
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("Class Probabilities", sec_style))
    bd = [["Class", "Probability", "Score"]]
    for i, p in enumerate(probs):
        bar = "█" * int(p*20) + "░" * (20 - int(p*20))
        bd.append([class_names[i].capitalize(), bar, f"{p:.4f}"])
    bt = Table(bd, colWidths=[1.5*inch, 3*inch, 1.5*inch])
    bt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f3f4f6")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f9fafb")]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
    ]))
    elements.append(bt)
    elements.append(Spacer(1, 16))

    heatmap_path = "heatmap.jpg"
    cv2.imwrite(heatmap_path, heatmap_img)
    cap_style = ParagraphStyle("cap", parent=styles["Normal"], alignment=TA_CENTER,
                               fontSize=10, textColor=colors.HexColor("#6b7280"))
    img_data = [
        [RLImage(original_path, width=2.8*inch, height=2.8*inch),
         RLImage(heatmap_path, width=2.8*inch, height=2.8*inch)],
        [Paragraph("<b>Original MRI</b>", cap_style), Paragraph("<b>Grad-CAM Heatmap</b>", cap_style)],
    ]
    it = Table(img_data, colWidths=[3*inch, 3*inch])
    it.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    elements.append(it)
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb"), spaceAfter=8))
    elements.append(Paragraph(
        "This report is generated by an AI model for informational purposes only. "
        "Not a medical diagnosis. Consult a qualified physician.",
        ParagraphStyle("W", parent=styles["Normal"], fontSize=9,
                       textColor=colors.HexColor("#92400e"),
                       backColor=colors.HexColor("#fffbeb"),
                       borderPadding=(6,8,6,8))
    ))
    doc.build(elements)
    return pdf_path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

bg_css = f'.stApp {{ background-image: url("data:image/jpg;base64,{img_base64}"); background-size: cover; background-position: center; background-attachment: fixed; }}' if img_base64 else ""

# ── Full CSS with 3D animations ───────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');

* {{ box-sizing: border-box; margin: 0; padding: 0; }}

{bg_css}

.stApp {{
    background-color: #030712;
    font-family: 'Inter', sans-serif;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: radial-gradient(ellipse at 20% 20%, rgba(56,189,248,0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(139,92,246,0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(6,182,212,0.04) 0%, transparent 70%),
                rgba(3,7,18,0.88);
    z-index: -1;
}}

.block-container {{
    background: rgba(8, 15, 35, 0.85) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 24px !important;
    backdrop-filter: blur(20px) !important;
    padding: 3rem 2.5rem !important;
    box-shadow:
        0 0 0 1px rgba(56,189,248,0.05),
        0 25px 50px rgba(0,0,0,0.6),
        inset 0 1px 0 rgba(255,255,255,0.05) !important;
    animation: containerIn 0.8s cubic-bezier(0.16,1,0.3,1) both;
}}

@keyframes containerIn {{
    from {{ opacity:0; transform: perspective(1000px) rotateX(8deg) translateY(30px); }}
    to   {{ opacity:1; transform: perspective(1000px) rotateX(0deg) translateY(0); }}
}}

h1,h2,h3,h4,p,label,span,div,li {{ color: #f1f5f9 !important; }}

.stSuccess p, .stSuccess div {{ color: #6ee7b7 !important; }}
.stError   p, .stError div   {{ color: #fca5a5 !important; }}
.stWarning p, .stWarning div {{ color: #fde68a !important; }}

.stSuccess {{ background: rgba(16,185,129,0.08) !important; border:1px solid rgba(16,185,129,0.3) !important; border-radius:12px !important; }}
.stError   {{ background: rgba(239,68,68,0.08)  !important; border:1px solid rgba(239,68,68,0.3)  !important; border-radius:12px !important; }}
.stWarning {{ background: rgba(245,158,11,0.08) !important; border:1px solid rgba(245,158,11,0.3) !important; border-radius:12px !important; }}

[data-testid="stFileUploader"] {{
    border: 2px dashed rgba(56,189,248,0.4) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    background: rgba(56,189,248,0.03) !important;
    transition: all 0.3s ease;
    animation: float3d 6s ease-in-out infinite;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: rgba(56,189,248,0.9) !important;
    background: rgba(56,189,248,0.07) !important;
    transform: perspective(800px) rotateX(-2deg) translateY(-3px);
    box-shadow: 0 20px 40px rgba(56,189,248,0.15), 0 0 0 1px rgba(56,189,248,0.3);
}}
[data-testid="stFileUploader"] section {{ background: transparent !important; }}
[data-testid="stFileUploader"] small {{ display:none !important; }}
[data-testid="stFileUploader"] label {{ color: #e2e8f0 !important; font-size:15px !important; font-weight:500 !important; }}
[data-testid="stFileUploader"] button {{
    background: linear-gradient(135deg, #0ea5e9, #8b5cf6) !important;
    color: white !important; border:none !important;
    border-radius:10px !important; font-weight:600 !important;
    letter-spacing:0.04em !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.4);
    transition: all 0.3s ease !important;
}}
[data-testid="stFileUploader"] button:hover {{
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 8px 30px rgba(14,165,233,0.6) !important;
}}

img {{
    border-radius: 14px !important;
    transition: all 0.4s cubic-bezier(0.34,1.56,0.64,1) !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4) !important;
}}
img:hover {{
    transform: perspective(600px) rotateY(5deg) rotateX(-3deg) scale(1.04) !important;
    box-shadow: 0 20px 60px rgba(56,189,248,0.35),
                0 0 0 1px rgba(56,189,248,0.3) !important;
}}

.stDownloadButton > button {{
    background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 50%, #06b6d4 100%) !important;
    background-size: 200% 200% !important;
    color: white !important; border:none !important;
    border-radius:14px !important; padding:14px 32px !important;
    font-size:15px !important; font-weight:700 !important;
    letter-spacing:0.05em !important; width:100% !important;
    box-shadow: 0 8px 30px rgba(14,165,233,0.4),
                0 0 0 1px rgba(255,255,255,0.05);
    animation: btnPulse 3s ease-in-out infinite;
    transition: all 0.3s ease !important;
}}
.stDownloadButton > button:hover {{
    transform: perspective(600px) rotateX(-4deg) translateY(-3px) scale(1.02) !important;
    box-shadow: 0 16px 50px rgba(14,165,233,0.6),
                0 0 80px rgba(139,92,246,0.3) !important;
}}

@keyframes btnPulse {{
    0%,100% {{ box-shadow: 0 8px 30px rgba(14,165,233,0.4); }}
    50%      {{ box-shadow: 0 8px 40px rgba(139,92,246,0.5), 0 0 60px rgba(14,165,233,0.2); }}
}}

@keyframes float3d {{
    0%,100% {{ transform: perspective(800px) rotateX(0deg) translateY(0); }}
    50%     {{ transform: perspective(800px) rotateX(1deg) translateY(-4px); }}
}}

@keyframes cardIn {{
    from {{ opacity:0; transform: perspective(800px) rotateY(-15deg) translateX(-20px); }}
    to   {{ opacity:1; transform: perspective(800px) rotateY(0deg) translateX(0); }}
}}

@keyframes cardInRight {{
    from {{ opacity:0; transform: perspective(800px) rotateY(15deg) translateX(20px); }}
    to   {{ opacity:1; transform: perspective(800px) rotateY(0deg) translateX(0); }}
}}

@keyframes slideUp {{
    from {{ opacity:0; transform: translateY(24px); }}
    to   {{ opacity:1; transform: translateY(0); }}
}}

@keyframes barGrow {{
    from {{ width: 0 !important; }}
}}

@keyframes rotateBorder {{
    0%   {{ border-color: rgba(56,189,248,0.6) rgba(56,189,248,0.1) rgba(56,189,248,0.1) rgba(56,189,248,0.1); }}
    25%  {{ border-color: rgba(56,189,248,0.1) rgba(139,92,246,0.6) rgba(56,189,248,0.1) rgba(56,189,248,0.1); }}
    50%  {{ border-color: rgba(56,189,248,0.1) rgba(56,189,248,0.1) rgba(6,182,212,0.6) rgba(56,189,248,0.1); }}
    75%  {{ border-color: rgba(56,189,248,0.1) rgba(56,189,248,0.1) rgba(56,189,248,0.1) rgba(139,92,246,0.6); }}
    100% {{ border-color: rgba(56,189,248,0.6) rgba(56,189,248,0.1) rgba(56,189,248,0.1) rgba(56,189,248,0.1); }}
}}

@keyframes pulseDot {{
    0%,100% {{ transform:scale(1); opacity:1; box-shadow: 0 0 0 0 currentColor; }}
    50%     {{ transform:scale(1.4); opacity:0.7; box-shadow: 0 0 0 6px transparent; }}
}}

@keyframes scanLine {{
    0%   {{ top: 0%; opacity:1; }}
    100% {{ top: 100%; opacity:0.2; }}
}}

@keyframes orbit {{
    from {{ transform: rotate(0deg) translateX(28px) rotate(0deg); }}
    to   {{ transform: rotate(360deg) translateX(28px) rotate(-360deg); }}
}}

.card-3d {{
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 18px;
    padding: 20px 22px;
    margin-bottom: 16px;
    transition: all 0.4s cubic-bezier(0.34,1.56,0.64,1);
    position: relative;
    overflow: hidden;
}}
.card-3d::before {{
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(56,189,248,0.06) 0%, rgba(139,92,246,0.04) 100%);
    pointer-events: none;
}}
.card-3d:hover {{
    transform: perspective(800px) rotateX(-3deg) rotateY(3deg) translateY(-5px);
    border-color: rgba(56,189,248,0.5);
    box-shadow: 0 20px 60px rgba(0,0,0,0.5),
                0 0 40px rgba(56,189,248,0.15),
                inset 0 1px 0 rgba(255,255,255,0.07);
}}

.card-label {{
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(148,163,184,0.7) !important;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 7px;
}}
.card-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,0.3), transparent);
}}

.hero-title {{
    font-family: 'Orbitron', monospace;
    font-size: 38px;
    font-weight: 900;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 40%, #c084fc 70%, #38bdf8 100%);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    animation: gradientShift 4s ease infinite;
    line-height: 1.2;
    margin-bottom: 8px;
}}

@keyframes gradientShift {{
    0%,100% {{ background-position: 0% 50%; }}
    50%     {{ background-position: 100% 50%; }}
}}

.hero-sub {{
    text-align: center;
    font-size: 14px !important;
    color: rgba(148,163,184,0.65) !important;
    letter-spacing: 0.06em;
    margin-bottom: 2rem;
}}

.conf-number {{
    font-family: 'Orbitron', monospace;
    font-size: 56px;
    font-weight: 900;
    background: linear-gradient(135deg, #38bdf8, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    letter-spacing: -0.02em;
}}

.conf-bar-track {{
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 6px;
    margin-top: 14px;
    overflow: hidden;
    position: relative;
}}
.conf-bar-fill {{
    height: 6px;
    border-radius: 6px;
    background: linear-gradient(90deg, #0ea5e9, #8b5cf6, #06b6d4);
    background-size: 200% 100%;
    animation: barGrow 1.4s cubic-bezier(0.34,1.56,0.64,1) both,
               gradientShift 3s ease infinite;
    position: relative;
}}
.conf-bar-fill::after {{
    content: '';
    position: absolute;
    right: 0; top: 0; bottom: 0;
    width: 20px;
    background: rgba(255,255,255,0.4);
    border-radius: 6px;
    filter: blur(4px);
}}

.pulse-badge {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 12px 20px;
    border-radius: 50px;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.08em;
    font-family: 'Orbitron', monospace;
    margin-top: 6px;
}}
.pulse-badge.safe {{
    background: rgba(16,185,129,0.1);
    border: 1.5px solid rgba(16,185,129,0.4);
    color: #34d399 !important;
    box-shadow: 0 0 30px rgba(16,185,129,0.1), inset 0 1px 0 rgba(255,255,255,0.05);
}}
.pulse-badge.danger {{
    background: rgba(239,68,68,0.1);
    border: 1.5px solid rgba(239,68,68,0.4);
    color: #f87171 !important;
    box-shadow: 0 0 30px rgba(239,68,68,0.1), inset 0 1px 0 rgba(255,255,255,0.05);
    animation: dangerPulse 2s ease-in-out infinite;
}}
@keyframes dangerPulse {{
    0%,100% {{ box-shadow: 0 0 30px rgba(239,68,68,0.1); }}
    50%     {{ box-shadow: 0 0 50px rgba(239,68,68,0.25); }}
}}

.pulse-dot {{
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
    animation: pulseDot 1.8s ease-in-out infinite;
}}
.pulse-dot.safe   {{ background:#34d399; color:#34d399; }}
.pulse-dot.danger {{ background:#f87171; color:#f87171; }}

.bar-row {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
    animation: slideUp 0.5s ease both;
}}
.bar-class {{
    font-size: 13px;
    font-weight: 600;
    width: 100px;
    color: #cbd5e1 !important;
    text-transform: capitalize;
    letter-spacing: 0.03em;
}}
.bar-outer {{
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
}}
.bar-inner {{
    height: 8px;
    border-radius: 8px;
    animation: barGrow 1.2s cubic-bezier(0.34,1.56,0.64,1) both;
    position: relative;
}}
.bar-inner::after {{
    content: '';
    position: absolute;
    right: 0; top: 0; bottom: 0;
    width: 12px;
    background: rgba(255,255,255,0.5);
    border-radius: 8px;
    filter: blur(3px);
}}
.bar-val {{
    font-size: 13px;
    font-weight: 600;
    color: #94a3b8 !important;
    width: 42px;
    text-align: right;
    font-family: 'Orbitron', monospace;
    font-size: 11px;
}}

.scan-wrapper {{
    position: relative;
    border-radius: 14px;
    overflow: hidden;
}}
.scan-line {{
    position: absolute;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.9), transparent);
    animation: scanLine 2.5s ease-in-out infinite;
    pointer-events: none;
    z-index: 10;
    box-shadow: 0 0 10px rgba(56,189,248,0.8);
}}

.section-divider {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 1.8rem 0 1rem;
}}
.section-divider-text {{
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(148,163,184,0.5) !important;
    white-space: nowrap;
}}
.section-divider-line {{
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,0.3), transparent);
}}

.warning-box {{
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 13px !important;
    color: #fde68a !important;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 12px 0;
    animation: slideUp 0.6s ease 0.5s both;
}}

.footer-custom {{
    text-align: center;
    font-size: 12px !important;
    color: rgba(148,163,184,0.35) !important;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
    letter-spacing: 0.04em;
}}
.footer-custom a {{ color: rgba(56,189,248,0.6) !important; text-decoration: none; }}
.footer-custom a:hover {{ color: #38bdf8 !important; }}

footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ── Animated particle canvas header ──────────────────────────────────────────
st.markdown("""
<div style="position:relative; text-align:center; padding: 1rem 0 2.5rem; overflow:hidden;">
  <canvas id="particleCanvas" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;opacity:0.5;"></canvas>
  <div style="position:relative; z-index:1;">
    <div style="display:inline-flex;align-items:center;justify-content:center;width:72px;height:72px;
                border-radius:50%;background:rgba(14,165,233,0.12);border:1.5px solid rgba(14,165,233,0.3);
                margin-bottom:14px;position:relative;animation:float3d 4s ease-in-out infinite;">
      <span style="font-size:32px;">🧠</span>
      <div style="position:absolute;width:8px;height:8px;border-radius:50%;background:#38bdf8;
                  top:6px;right:6px;animation:orbit 3s linear infinite;"></div>
    </div>
    <div class="hero-title">Brain Tumor Detection</div>
    <div class="hero-sub">AI · MRI Analysis · Grad-CAM · Real-time Diagnosis</div>
  </div>
</div>
<script>
(function() {
  var c = document.getElementById('particleCanvas');
  if (!c) return;
  var ctx = c.getContext('2d');
  var W = c.offsetWidth, H = c.offsetHeight;
  c.width = W; c.height = H;
  var pts = [];
  for (var i = 0; i < 40; i++) {
    pts.push({ x: Math.random()*W, y: Math.random()*H,
               vx: (Math.random()-0.5)*0.4, vy: (Math.random()-0.5)*0.4,
               r: Math.random()*1.5+0.5 });
  }
  function draw() {
    ctx.clearRect(0,0,W,H);
    pts.forEach(function(p) {
      p.x += p.vx; p.y += p.vy;
      if (p.x<0||p.x>W) p.vx*=-1;
      if (p.y<0||p.y>H) p.vy*=-1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(56,189,248,0.7)';
      ctx.fill();
    });
    for (var i=0;i<pts.length;i++) for (var j=i+1;j<pts.length;j++) {
      var dx=pts[i].x-pts[j].x, dy=pts[i].y-pts[j].y, d=Math.sqrt(dx*dx+dy*dy);
      if (d<90) {
        ctx.beginPath();
        ctx.moveTo(pts[i].x,pts[i].y);
        ctx.lineTo(pts[j].x,pts[j].y);
        ctx.strokeStyle='rgba(56,189,248,'+(0.15*(1-d/90))+')';
        ctx.lineWidth=0.5;
        ctx.stroke();
      }
    }
    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-divider">
  <div class="section-divider-line"></div>
  <span class="section-divider-text">Upload MRI scan</span>
  <div class="section-divider-line" style="background:linear-gradient(90deg,transparent,rgba(56,189,248,0.3));"></div>
</div>
""", unsafe_allow_html=True)

file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

st.markdown("""
<p style="text-align:center;font-size:12px;color:rgba(148,163,184,0.4)!important;
   margin-top:8px;letter-spacing:0.06em;">
  SUPPORTED &nbsp;·&nbsp; JPG &nbsp;·&nbsp; PNG &nbsp;·&nbsp; JPEG
</p>
""", unsafe_allow_html=True)

# ── Processing ────────────────────────────────────────────────────────────────
if file:
    img = Image.open(file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        img.save(temp.name)
        label, conf = predict_image(temp.name)
        probs = predict_full(temp.name)
        heatmap_img = generate_gradcam(temp.name, model)
        heatmap_img = draw_bounding_box(heatmap_img)
        temp_path = temp.name

    st.success("Analysis complete")

    # ── Images ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-divider">
      <div class="section-divider-line"></div>
      <span class="section-divider-text">MRI scan images</span>
      <div class="section-divider-line" style="background:linear-gradient(90deg,transparent,rgba(56,189,248,0.3));"></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card-3d" style="animation: cardIn 0.7s ease both;">
          <div class="card-label">Original scan</div>
          <div class="scan-wrapper"><div class="scan-line"></div>
        """, unsafe_allow_html=True)
        st.image(img, width=290)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card-3d" style="animation: cardInRight 0.7s ease 0.1s both;">
          <div class="card-label">Grad-CAM heatmap</div>
          <div class="scan-wrapper">
        """, unsafe_allow_html=True)
        st.image(heatmap_img, width=290, channels="BGR")
        st.markdown("</div></div>", unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-divider">
      <div class="section-divider-line"></div>
      <span class="section-divider-text">Diagnosis results</span>
      <div class="section-divider-line" style="background:linear-gradient(90deg,transparent,rgba(56,189,248,0.3));"></div>
    </div>
    """, unsafe_allow_html=True)

    is_tumor = label.lower() != "notumor"
    badge_cls = "danger" if is_tumor else "safe"
    badge_icon = "⚠" if is_tumor else "✓"

    res_col, conf_col = st.columns(2)

    with res_col:
        st.markdown(f"""
        <div class="card-3d" style="animation: cardIn 0.6s ease 0.2s both;">
          <div class="card-label">Prediction result</div>
          <div class="pulse-badge {badge_cls}">
            <span class="pulse-dot {badge_cls}"></span>
            {badge_icon} &nbsp;{label.upper()}
          </div>
          <p style="font-size:12px!important;color:rgba(148,163,184,0.5)!important;
             margin-top:12px;letter-spacing:0.04em;">
            {"Tumor detected — consult a physician" if is_tumor else "No tumor detected in scan"}
          </p>
        </div>
        """, unsafe_allow_html=True)

    with conf_col:
        bar_w = int(conf * 100)
        st.markdown(f"""
        <div class="card-3d" style="animation: cardInRight 0.6s ease 0.3s both;">
          <div class="card-label">Confidence score</div>
          <div class="conf-number">{conf*100:.1f}%</div>
          <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width:{bar_w}%;"></div>
          </div>
          <p style="font-size:11px!important;color:rgba(148,163,184,0.4)!important;
             margin-top:10px;letter-spacing:0.05em;">
            MODEL CERTAINTY SCORE
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Probabilities ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-divider">
      <div class="section-divider-line"></div>
      <span class="section-divider-text">Class probabilities</span>
      <div class="section-divider-line" style="background:linear-gradient(90deg,transparent,rgba(56,189,248,0.3));"></div>
    </div>
    """, unsafe_allow_html=True)

    class_cfg = {
        "glioma":     ("linear-gradient(90deg,#dc2626,#ef4444)", "#fca5a5"),
        "meningioma": ("linear-gradient(90deg,#d97706,#f59e0b)", "#fde68a"),
        "notumor":    ("linear-gradient(90deg,#059669,#10b981)", "#6ee7b7"),
        "pituitary":  ("linear-gradient(90deg,#7c3aed,#a78bfa)", "#c4b5fd"),
    }

    st.markdown("<div class='card-3d' style='animation:slideUp 0.6s ease 0.4s both;'>", unsafe_allow_html=True)
    for i, p in enumerate(probs):
        name = class_names[i]
        grad, text_col = class_cfg.get(name, ("linear-gradient(90deg,#475569,#64748b)", "#94a3b8"))
        pct = max(int(p * 100), 1)
        delay = 0.08 * i
        st.markdown(f"""
        <div class="bar-row" style="animation-delay:{delay}s;">
          <span class="bar-class" style="color:{text_col}!important;">{name.capitalize()}</span>
          <div class="bar-outer">
            <div class="bar-inner" style="width:{pct}%;background:{grad};animation-delay:{delay}s;"></div>
          </div>
          <span class="bar-val">{p:.3f}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Warning ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="warning-box">
      <span style="font-size:18px;flex-shrink:0;">⚠</span>
      <span style="color:#fde68a!important;font-size:13px!important;">
        <strong style="color:#fbbf24!important;">Medical disclaimer:</strong>
        This AI analysis is for informational purposes only and does not constitute a medical diagnosis.
        Always consult a qualified neurologist or radiologist for clinical interpretation.
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── PDF download ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-divider">
      <div class="section-divider-line"></div>
      <span class="section-divider-text">Download report</span>
      <div class="section-divider-line" style="background:linear-gradient(90deg,transparent,rgba(56,189,248,0.3));"></div>
    </div>
    """, unsafe_allow_html=True)

    pdf_file = generate_pdf(label, conf, probs, temp_path, heatmap_img)
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄   Download Full Diagnostic Report (PDF)",
            data=f,
            file_name="brain_tumor_report.pdf",
            mime="application/pdf",
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-custom">
  © 2026 Brain Tumor Detection System &nbsp;·&nbsp; Sai Manoj &nbsp;·&nbsp;
  <a href="mailto:saimanoj0914@gmail.com">saimanoj0914@gmail.com</a>
</div>
""", unsafe_allow_html=True)
PYEOF
echo "done"
