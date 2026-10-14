"""
SolarSentinel AI — Interactive Web Application
Demonstrates cDCGAN image generation, EfficientNet-B0 defect classification,
model comparisons, and evaluation dashboards for Generative AI Lab Evaluation 2.
"""

import os
import glob
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import torch

import config
from src.predict import SolarSentinelPredictor
from src.preprocessing import denormalize_gan_tensor
from src.gan_model import ConditionalGenerator

# ---------------------------------------------------------
# Page Configuration & Custom CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="SolarSentinel AI | cGAN + EfficientNet-B0",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Styling */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 24px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .main-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }
    .badge-synthetic {
        display: inline-block;
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-real {
        display: inline-block;
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Navigation & System Status
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1509391365360-2e959784a276?w=600&auto=format&fit=crop&q=80", caption="Solar Photovoltaic Inspection", use_container_width=True)
    st.title("SolarSentinel AI")
    st.caption("Generative AI Lab Evaluation 2 — cGAN + EfficientNet-B0")

    menu = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔍 Image Prediction",
            "🎨 cGAN Generation",
            "⚖️ Model Comparison",
            "📈 Testing & Evaluation",
            "⚙️ Pipeline & Diagnostics"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("System Health")
    device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    st.markdown(f"**Hardware Device:** `{device_name}`")

    # Status indicators for saved models
    base_exists = os.path.exists(config.BASELINE_CKPT)
    aug_exists = os.path.exists(config.AUGMENTED_CKPT)
    gan_exists = os.path.exists(config.GENERATOR_CKPT)

    st.markdown(f"- Baseline Model: {'🟢 Ready' if base_exists else '🔴 Missing'}")
    st.markdown(f"- GAN-Augmented: {'🟢 Ready' if aug_exists else '🟡 Awaiting'}")
    st.markdown(f"- cDCGAN Generator: {'🟢 Ready' if gan_exists else '🟡 Awaiting'}")

    st.caption("SolarSentinel AI © 2026")

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">☀️ SolarSentinel AI</div>
    <div class="main-subtitle">
        Automated Solar Panel Electroluminescence Defect Classification via Conditional DCGAN Augmentation and EfficientNet-B0
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 1: DASHBOARD
# ---------------------------------------------------------
if menu == "📊 Dashboard":
    st.header("Project Overview & Architecture")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="card">
            <div class="metric-label">Dataset Images</div>
            <div class="metric-value">13,414</div>
            <div style="color: #94a3b8; font-size: 0.8rem;">PVEL-AD Benchmark</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <div class="metric-label">Target Classes</div>
            <div class="metric-value">8</div>
            <div style="color: #94a3b8; font-size: 0.8rem;">6 Defects + Good + Material</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
            <div class="metric-label">Classifier Backbone</div>
            <div class="metric-value">4.01M</div>
            <div style="color: #94a3b8; font-size: 0.8rem;">EfficientNet-B0 Parameters</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="card">
            <div class="metric-label">Generative Architecture</div>
            <div class="metric-value">cDCGAN</div>
            <div style="color: #94a3b8; font-size: 0.8rem;">Defect-Conditioned Generator</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("System Architecture & Workflow")
    st.markdown("""
    SolarSentinel AI implements an **end-to-end Generative AI pipeline** that addresses extreme class imbalance in solar cell inspection:
    
    1. **PVELAD Electroluminescence Dataset**: 13,414 high-resolution EL images partitioned into a strict **70:15:15** split (9,389 train, 2,012 val, 2,013 held-out test).
    2. **Conditional DCGAN (cDCGAN)**: Synthesizes high-fidelity synthetic EL images conditioned on target minority defects (`crack`, `star_crack`, `thick_line`, `finger`, `black_core`, `other_defect`).
    3. **Quality Validation Gate**: Filters generated images for non-finite values and mode collapse before injection into training.
    4. **Dual Controlled Experiments**:
       - **Experiment A (Baseline)**: EfficientNet-B0 trained solely on real images.
       - **Experiment B (GAN-Augmented)**: EfficientNet-B0 trained on real + validated synthetic images.
    5. **Zero-Leakage Held-Out Evaluation**: Both models evaluated on the identical, untouched real test set.
    """)

    # Class distribution plot
    if os.path.exists(config.DATASET_INDEX_CSV):
        st.subheader("Dataset Class Distribution (PVEL-AD)")
        df_idx = pd.read_csv(config.DATASET_INDEX_CSV)
        counts = {c: int(df_idx[c].sum()) for c in config.CLASS_NAMES}
        
        fig, ax = plt.subplots(figsize=(10, 3.5))
        bars = ax.bar(list(counts.keys()), list(counts.values()), color='#0284c7', edgecolor='#38bdf8', alpha=0.85)
        ax.set_ylabel("Number of Samples")
        ax.set_title("Severe Class Imbalance in Real PVEL-AD Dataset", fontweight='bold')
        plt.xticks(rotation=20, ha='right')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f"{yval:,}", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

# ---------------------------------------------------------
# TAB 2: IMAGE PREDICTION
# ---------------------------------------------------------
elif menu == "🔍 Image Prediction":
    st.header("Real-Time Solar Panel Defect Inference")

    # Model selector
    ckpt_options = {}
    if os.path.exists(config.BASELINE_CKPT):
        ckpt_options["Baseline EfficientNet-B0 (Exp A)"] = config.BASELINE_CKPT
    if os.path.exists(config.AUGMENTED_CKPT):
        ckpt_options["GAN-Augmented EfficientNet-B0 (Exp B)"] = config.AUGMENTED_CKPT

    if not ckpt_options:
        st.warning("⚠️ No trained classifier checkpoints found in `checkpoints/`. Please run training first.")
        st.stop()

    selected_model_name = st.selectbox("Select Model Checkpoint:", list(ckpt_options.keys()))
    selected_ckpt = ckpt_options[selected_model_name]

    col_in, col_out = st.columns([1, 1.2])

    with col_in:
        st.subheader("Input Image")
        input_mode = st.radio("Select Input Source:", ["Upload My Own Image", "Pick Sample from Real Dataset"], horizontal=True)

        input_img = None
        if input_mode == "Upload My Own Image":
            uploaded_file = st.file_uploader("Upload Solar Panel EL Image (JPG, PNG)", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                try:
                    input_img = Image.open(uploaded_file).convert('RGB')
                    st.image(input_img, caption="Uploaded Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
        else:
            # Curated samples from train_PVELAD or test_PVELAD
            sample_candidates = glob.glob(os.path.join(config.PROJECT_ROOT, "train_PVELAD", "train", "JPEGImages", "*.jpg"))[:12]
            if sample_candidates:
                sample_names = [os.path.basename(p) for p in sample_candidates]
                chosen_name = st.selectbox("Choose sample image:", sample_names)
                chosen_path = os.path.join(config.PROJECT_ROOT, "train_PVELAD", "train", "JPEGImages", chosen_name)
                input_img = Image.open(chosen_path).convert('RGB')
                st.image(input_img, caption=f"Dataset Sample: {chosen_name}", use_container_width=True)
            else:
                st.info("Sample directory not found. Please upload an image.")

    with col_out:
        st.subheader("Classification Results")
        if input_img is not None:
            if st.button("🚀 Analyze Solar Panel", type="primary"):
                with st.spinner("Analyzing electroluminescence patterns..."):
                    predictor = SolarSentinelPredictor(checkpoint_path=selected_ckpt)
                    res = predictor.predict(input_img)

                # Diagnosis Card
                card_bg = "#7f1d1d" if res['is_defective'] else "#064e3b"
                card_border = "#ef4444" if res['is_defective'] else "#10b981"
                status_text = "⚠️ DEFECT DETECTED" if res['is_defective'] else "✅ HEALTHY / PASS"

                st.markdown(f"""
                <div style="background-color: {card_bg}; border: 2px solid {card_border}; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                    <h3 style="margin-top:0; color: white;">{status_text}</h3>
                    <p style="color: #f1f5f9; font-size: 1.05rem; margin-bottom: 6px;"><b>Interpretation:</b> {res['interpretation']}</p>
                    <p style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0;"><b>Inference Latency:</b> {res['latency_ms']:.2f} ms</p>
                </div>
                """, unsafe_allow_html=True)

                # Probabilities Bar Chart
                probs = res['probabilities']
                prob_df = pd.DataFrame(list(probs.items()), columns=['Class', 'Probability']).sort_values('Probability', ascending=True)

                fig, ax = plt.subplots(figsize=(8, 4.5))
                bar_colors = ['#ef4444' if c in config.DEFECT_CLASSES and p >= 0.5 else '#38bdf8' for c, p in zip(prob_df['Class'], prob_df['Probability'])]
                bars = ax.barh(prob_df['Class'], prob_df['Probability'], color=bar_colors, height=0.6)
                ax.axvline(0.5, color='#f59e0b', linestyle='--', label='Decision Threshold (0.5)')
                ax.set_xlim([0.0, 1.0])
                ax.set_xlabel("Sigmoid Multi-Label Probability")
                ax.set_title("Defect Category Probabilities", fontweight='bold')
                ax.legend(loc='lower right')
                for bar in bars:
                    w = bar.get_width()
                    ax.text(w + 0.02, bar.get_y() + bar.get_height()/2.0, f"{w:.1%}", va='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("👈 Please upload or select a solar panel EL image to run classification.")

# ---------------------------------------------------------
# TAB 3: GAN IMAGE GENERATION
# ---------------------------------------------------------
elif menu == "🎨 cGAN Generation":
    st.header("Conditional DCGAN (cDCGAN) Image Generation")
    st.markdown("""
    Use the trained Generator network to create synthetic solar panel electroluminescence images on demand.
    Each generated image is conditioned on a specific defect category and watermarked as synthetic.
    """)

    if not os.path.exists(config.GENERATOR_CKPT):
        st.warning("⚠️ Generator checkpoint not found at `checkpoints/generator.pth`. Please run GAN training first.")
        st.stop()

    col_ctrl, col_display = st.columns([1, 1.5])

    with col_ctrl:
        st.subheader("Generation Controls")
        condition_class = st.selectbox(
            "Select Defect Condition:",
            config.CLASS_NAMES,
            index=0
        )
        cls_idx = config.CLASS_TO_IDX[condition_class]

        seed_option = st.number_input("Random Noise Seed:", min_value=0, max_value=999999, value=42)
        num_imgs = st.slider("Number of Variations:", min_value=1, max_value=4, value=2)

        gen_button = st.button("✨ Generate Synthetic Images", type="primary")

    with col_display:
        st.subheader("Synthetic Output")
        if gen_button:
            with st.spinner(f"Generating {num_imgs} synthetic image(s) conditioned on '{condition_class}'..."):
                torch.manual_seed(seed_option)
                ckpt = torch.load(config.GENERATOR_CKPT, map_location='cpu')
                netG = ConditionalGenerator(
                    latent_dim=ckpt.get('latent_dim', config.GAN_LATENT_DIM),
                    num_classes=ckpt.get('num_classes', config.NUM_CLASSES),
                    embed_dim=ckpt.get('embed_dim', config.GAN_EMBED_DIM)
                )
                netG.load_state_dict(ckpt['generator_state_dict'])
                netG.eval()

                z = torch.randn(num_imgs, config.GAN_LATENT_DIM)
                c = torch.tensor([cls_idx] * num_imgs, dtype=torch.long)

                with torch.no_grad():
                    fakes = netG(z, c)
                    fakes = denormalize_gan_tensor(fakes)

            img_cols = st.columns(num_imgs)
            for i in range(num_imgs):
                with img_cols[i]:
                    img_np = fakes[i].permute(1, 2, 0).numpy()
                    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                    st.image(img_pil, caption=f"Condition: {condition_class}", use_container_width=True)
                    st.markdown('<span class="badge-synthetic">Synthetic — AI Generated</span>', unsafe_allow_html=True)
        else:
            st.info("👈 Configure the defect class and click **✨ Generate Synthetic Images** to produce GAN outputs.")

    st.markdown("---")
    st.subheader("Sample Generation Gallery")
    sample_grids = sorted(glob.glob(os.path.join(config.GENERATED_SAMPLES_DIR, "progress_grid_epoch_*.png")))
    if sample_grids:
        latest_grid = sample_grids[-1]
        st.image(latest_grid, caption=f"cDCGAN Training Checkpoint Progress ({os.path.basename(latest_grid)})", use_container_width=True)

# ---------------------------------------------------------
# TAB 4: MODEL COMPARISON
# ---------------------------------------------------------
elif menu == "⚖️ Model Comparison":
    st.header("Controlled Experiment Comparison")
    st.markdown("""
    Direct empirical comparison of **Experiment A (Baseline)** vs **Experiment B (GAN-Augmented)** on the **identical, untouched held-out real test set**.
    """)

    summary_csv = config.MODEL_COMPARISON_CSV
    if os.path.exists(summary_csv):
        comp_df = pd.read_csv(summary_csv)

        st.subheader("Per-Class Performance Comparison Table")
        st.dataframe(comp_df.style.highlight_max(subset=['Augmented_F1', 'Augmented_Recall'], color='#065f46'), use_container_width=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            f1_plot = os.path.join(config.PLOTS_DIR, "model_comparison_f1.png")
            if os.path.exists(f1_plot):
                st.image(f1_plot, caption="F1-Score Comparison Across Defect Classes", use_container_width=True)
        with col_c2:
            recall_plot = os.path.join(config.PLOTS_DIR, "model_comparison_recall.png")
            if os.path.exists(recall_plot):
                st.image(recall_plot, caption="Defect Recall Comparison Across Classes", use_container_width=True)

        report_md = os.path.join(config.REPORTS_DIR, "model_comparison_report.md")
        if os.path.exists(report_md):
            with open(report_md, 'r', encoding='utf-8') as f:
                st.markdown(f.read())
    else:
        st.info("ℹ️ Comparative metrics have not been generated yet. Run `python run_pipeline.py` to evaluate both experiments.")

# ---------------------------------------------------------
# TAB 5: TESTING & EVALUATION
# ---------------------------------------------------------
elif menu == "📈 Testing & Evaluation":
    st.header("Comprehensive Evaluation Visualizations")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("Confusion Matrices (Test Set)")
        cm_path = os.path.join(config.PLOTS_DIR, "baseline_confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Baseline Per-Class Confusion Matrices", use_container_width=True)
        else:
            st.info("Confusion matrix plot not found.")

    with col_v2:
        st.subheader("Multi-Label ROC-AUC Curves")
        roc_path = os.path.join(config.PLOTS_DIR, "baseline_roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="Baseline Multi-Label ROC Curves", use_container_width=True)
        else:
            st.info("ROC curves plot not found.")

    col_v3, col_v4 = st.columns(2)
    with col_v3:
        st.subheader("cDCGAN Adversarial Training Dynamics")
        gan_loss_path = os.path.join(config.PLOTS_DIR, "gan_loss_curve.png")
        if os.path.exists(gan_loss_path):
            st.image(gan_loss_path, caption="Generator vs Discriminator Loss Over Epochs", use_container_width=True)
        else:
            st.info("GAN loss curve not found.")

    with col_v4:
        st.subheader("Classifier Training History")
        clf_curves = os.path.join(config.PLOTS_DIR, "baseline_training_curves.png")
        if os.path.exists(clf_curves):
            st.image(clf_curves, caption="EfficientNet-B0 Training and Validation Curves", use_container_width=True)
        else:
            st.info("Classifier training curves not found.")

# ---------------------------------------------------------
# TAB 6: PIPELINE & DIAGNOSTICS
# ---------------------------------------------------------
elif menu == "⚙️ Pipeline & Diagnostics":
    st.header("Pipeline Orchestration & Test Status")

    st.subheader("Component Verification Checklist")
    status_items = [
        ("Dataset Index & Verified Splits", os.path.exists(config.SPLITS_INDEX_CSV)),
        ("Baseline Checkpoint (`baseline_efficientnet_b0.pth`)", os.path.exists(config.BASELINE_CKPT)),
        ("cDCGAN Generator Checkpoint (`generator.pth`)", os.path.exists(config.GENERATOR_CKPT)),
        ("cDCGAN Discriminator Checkpoint (`discriminator.pth`)", os.path.exists(config.DISCRIMINATOR_CKPT)),
        ("Synthetic Dataset Manifest", os.path.exists(os.path.join(config.GENERATED_SAMPLES_DIR, "synthetic_manifest.csv"))),
        ("GAN-Augmented Checkpoint (`gan_augmented_efficientnet_b0.pth`)", os.path.exists(config.AUGMENTED_CKPT)),
        ("Model Comparison Summary CSV", os.path.exists(config.MODEL_COMPARISON_CSV)),
        ("Formal Test Report (`test_report.md`)", os.path.exists(config.TEST_REPORT_MD)),
        ("Final Lab Evaluation Report", os.path.exists(config.FINAL_EVAL_REPORT_MD))
    ]

    for label, is_ready in status_items:
        icon = "✅" if is_ready else "⏳"
        st.markdown(f"{icon} **{label}**: {'Ready' if is_ready else 'Awaiting Execution'}")

    st.markdown("---")
    st.subheader("Execution Commands for Demonstration")
    st.code("""
# 1. Run Complete End-to-End Pipeline
python run_pipeline.py

# 2. Run Automated Verification Tests
python run_tests.py

# 3. Train cDCGAN Separately
python src/train_gan.py --epochs 15

# 4. Generate Synthetic Minority Defect Images
python src/generate_images.py --num-per-class 30

# 5. Train Experiment B (GAN-Augmented)
python src/train_classifier.py --mode augmented --epochs 5

# 6. Evaluate Models on Held-Out Test Set
python src/evaluate.py --checkpoint checkpoints/baseline_efficientnet_b0.pth --name Baseline
python src/evaluate.py --checkpoint checkpoints/gan_augmented_efficientnet_b0.pth --name GAN_Augmented

# 7. Compare Experiments
python src/compare_models.py

# 8. Launch Streamlit Application
streamlit run app.py
    """, language="bash")
