"""
Streamlit Web Application: Plant & Rice Leaf Disease Diagnostic Engine
Author: Arjuna Fransesco
"""

import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="🌾 Plant & Rice Leaf Disease Diagnostic Engine",
    page_icon="🌾",
    layout="wide"
)

# Custom Theme Styling
st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #14532d; margin-bottom: 5px; }
    .sub-title { font-size: 1.1rem; color: #475569; margin-bottom: 25px; }
    .diagnosis-box { background: #f0fdf4; border: 1px solid #86efac; border-radius: 10px; padding: 20px; }
    .warning-box { background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌾 Plant & Rice Leaf Pathology Diagnostic Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Developed by <b>Arjuna Fransesco</b> | Computer Vision & Agricultural AI Portfolio</div>', unsafe_allow_html=True)

DISEASE_INFO = {
    "Bacterial_Blight": {
        "title": "Bacterial Blight (Xanthomonas oryzae)",
        "icon": "⚠️",
        "description": "Water-soaked lesions on leaf margins, expanding into necrotic straw-colored stripes.",
        "action": "Apply copper oxychloride (2.5 g/L). Avoid excess nitrogen fertilizers and ensure field drainage."
    },
    "Brown_Spot": {
        "title": "Brown Spot (Bipolaris oryzae)",
        "icon": "🍂",
        "description": "Oval brown necrotic spots with yellow halos across the leaf surface.",
        "action": "Spray Mancozeb (2 g/L) or Tricyclazole. Supplement soil with potassium and silicon nutrients."
    },
    "Leaf_Blast": {
        "title": "Rice Leaf Blast (Magnaporthe oryzae)",
        "icon": "🚨",
        "description": "Diamond / spindle-shaped lesions with grayish centers and reddish-brown borders.",
        "action": "Deploy systemic fungicides (Isoprothiolane / Tricyclazole 75% WP). Adjust crop density."
    },
    "Healthy_Leaf": {
        "title": "Healthy Crop Tissue",
        "icon": "✅",
        "description": "Vibrant uniform chlorophyll pigmentation with zero necrosis.",
        "action": "Maintain routine NPK fertilization and balanced irrigation regimen."
    },
    "Tungro_Virus": {
        "title": "Rice Tungro Virus (RTV)",
        "icon": "🦗",
        "description": "Yellow-orange discoloration spreading downward from leaf tips with stunted tillering.",
        "action": "Target insect vectors (Green Leafhoppers) using Imidacloprid (0.3 ml/L)."
    }
}


@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, "models", "plant_leaf_disease_model.joblib"))
    with open(os.path.join(base_dir, "reports", "metrics.json")) as f:
        metrics = json.load(f)
    return model, metrics

model, metrics = load_assets()

# Sidebar: Diagnostic Biomarker Sliders
st.sidebar.header("🔬 Leaf Biomarker & Color Indices")
greenness = st.sidebar.slider("Mean Greenness Index (HSV)", 0.20, 0.95, 0.48, step=0.01)
chlorosis = st.sidebar.slider("Chlorosis / Yellowing Index", 0.00, 0.85, 0.35, step=0.01)
lesion_ratio = st.sidebar.slider("Lesion Necrosis Area Ratio", 0.00, 0.85, 0.28, step=0.01)
glcm_contrast = st.sidebar.slider("GLCM Texture Contrast", 5.0, 75.0, 38.5, step=0.5)

input_df = pd.DataFrame([{
    "mean_greenness_index": greenness,
    "chlorosis_index": chlorosis,
    "lesion_area_ratio": lesion_ratio,
    "glcm_contrast": glcm_contrast
}])

predicted_class = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]
confidence = float(np.max(probabilities) * 100)

info = DISEASE_INFO.get(predicted_class, {})

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Diagnosed Pathology", predicted_class.replace("_", " "))
with col2:
    st.metric("Model Confidence", f"{confidence:.1f}%")
with col3:
    st.metric("Test Accuracy", f"{metrics['Test_Accuracy_Pct']}%")
with col4:
    st.metric("Macro F1-Score", f"{metrics['Macro_F1_Score']}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🩺 Clinical Pathology & Agronomy Advice", "📊 Multi-Class Probability Distribution", "📑 Model Specifications"])

with tab1:
    st.subheader(f"{info.get('icon', '🌱')} {info.get('title', predicted_class)}")
    st.write(f"**Pathological Symptoms:** {info.get('description', '')}")
    st.info(f"**💡 Recommended Treatment:** {info.get('action', '')}")

with tab2:
    prob_df = pd.DataFrame({
        "Disease Category": [c.replace("_", " ") for c in model.classes_],
        "Probability (%)": [p * 100 for p in probabilities]
    }).sort_values("Probability (%)", ascending=True)
    
    st.subheader("Class Probabilities")
    st.bar_chart(data=prob_df.set_index("Disease Category"))

with tab3:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.json(metrics)
    with col_b:
        st.image("reports/confusion_matrix.png", use_container_width=True)
