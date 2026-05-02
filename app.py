import streamlit as st
import pandas as pd
import json
import torch
from torchvision import models, transforms
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ----------------
RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDS_DIR = RESULTS_DIR / "predictions"

CLASS_NAMES = [
    "good",
    "moderate",
    "severe",
    "unhealthy",
    "unhealthy for sensitive groups",
    "very unhealthy",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AQI Few-Shot Benchmark",
    page_icon="🌍",
    layout="wide",
)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🌍 AQI Few-Shot Benchmark")
st.sidebar.caption("Research-grade image classification demo")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Info", "🖼️ Image → Predict", "🏆 Leaderboard", "📊 Confusion Matrices", "📁 Prediction Browser"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with PyTorch & Streamlit")

# ================= INFO =================
if page == "🏠 Info":
    st.markdown("# 🌍 Few-Shot AQI Image Classification")
    st.markdown(
        """
        <div style='font-size:18px; color:#555'>
        A reproducible benchmark evaluating CNN and Transformer models
        under an extreme few-shot learning setting.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧪 Training Protocol")
    st.markdown("""
    - **12-shot training / 4-shot validation** per class  
    - Fixed per-class split (fully reproducible)  
    - Two-stage optimization:
        - Frozen backbone
        - Full fine-tuning  
    - **Macro-F1** as the primary evaluation metric
    """)

    st.markdown("### 🧠 Models Evaluated")
    st.markdown("""
    - ResNet-18 / 34  
    - MobileNet-V2  
    - DenseNet-121  
    - EfficientNet-B0 / B2 / B3  
    - ConvNeXt-Tiny / Small  
    - Vision Transformer (ViT-B/16)
    """)

    st.markdown("### 📦 Generated Artifacts")
    st.markdown("""
    - Model checkpoints (`.pt`)
    - Confusion matrices (JSON + plots)
    - Per-image predictions (CSV)
    - Leaderboards & publication-ready tables
    """)

# ================= IMAGE → PREDICT =================
elif page == "🖼️ Image → Predict":
    st.markdown("# 🖼️ Image → AQI Prediction")
    st.caption("Upload an image and run inference using a trained model")

    model_files = sorted(MODELS_DIR.glob("*finetune.pt"))
    if not model_files:
        st.error("No finetuned models found in `results/models/`")
        st.stop()

    model_file = st.selectbox("🧠 Select Model", model_files)

    uploaded = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded and model_file:
        col1, col2 = st.columns([1, 1])

        with col1:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Input Image", use_column_width=True)

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES)),
        )

        state = torch.load(model_file, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

        pred = probs.argmax()

        with col2:
            st.markdown("### ✅ Prediction Result")
            st.markdown(
                f"""
                <div style='font-size:26px; font-weight:700; color:#2E7D32'>
                {CLASS_NAMES[pred]}
                </div>
                <div style='font-size:16px; color:#555'>
                Confidence: {probs[pred]:.3f}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### 📊 Class Probabilities")
            st.bar_chart(pd.Series(probs, index=CLASS_NAMES))

# ================= LEADERBOARD =================
elif page == "🏆 Leaderboard":
    st.markdown("# 🏆 Model Leaderboard")
    st.caption("Comparison across architectures and training regimes")

    summary_files = list(RESULTS_DIR.glob("*summary*.csv"))
    if not summary_files:
        st.error("No summary CSV found in `results/`")
        st.stop()

    df = pd.read_csv(summary_files[0])

    metric = st.selectbox(
        "📌 Sort by metric",
        ["test_macro_f1", "test_acc", "val_macro_f1", "val_acc"],
    )

    st.dataframe(
        df.sort_values(metric, ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=520,
    )

# ================= CONFUSION MATRICES =================
elif page == "📊 Confusion Matrices":
    st.markdown("# 📊 Confusion Matrices")
    st.caption("Error distribution across AQI categories")

    json_files = sorted(METRICS_DIR.glob("*test_report.json"))
    if not json_files:
        st.error("No confusion matrix JSON files found.")
        st.stop()

    selected = st.selectbox("Select Model Report", json_files)

    with open(selected) as f:
        cm = json.load(f)["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

# ================= PREDICTION BROWSER =================
elif page == "📁 Prediction Browser":
    st.markdown("# 📁 Prediction Browser")
    st.caption("Inspect per-image predictions and model errors")

    pred_files = sorted(PREDS_DIR.glob("*.csv"))
    if not pred_files:
        st.error("No prediction CSVs found.")
        st.stop()

    selected = st.selectbox("Select Prediction File", pred_files)
    df = pd.read_csv(selected)

    if st.checkbox("🔍 Show only misclassified samples"):
        df = df[df["true_label"] != df["predicted_label"]]

    st.dataframe(df, use_container_width=True, height=600)
