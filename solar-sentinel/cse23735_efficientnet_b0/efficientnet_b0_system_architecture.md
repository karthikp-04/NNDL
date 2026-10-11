# SolarSentinel AI: EfficientNet-B0 System Architecture

This document presents the complete system architecture diagram for **SolarSentinel AI**, customized specifically for the **EfficientNet-B0** Electroluminescence (EL) Defect Classification pipeline. The layout directly mirrors the modular 4-layer reference architecture.

---

## 1. System Architecture Visual Diagram

![SolarSentinel AI EfficientNet-B0 Architecture](C:\Users\ASHIN\.gemini\antigravity-ide\brain\1abcc754-601a-48fa-84da-e27bb8c6bf0d\efficientnet_b0_architecture_diagram_1786422744406.png)

---

## 2. Interactive / Native Mermaid Architecture Flowchart

Below is the Mermaid flowchart representation of the system architecture, styled to match the color themes of the 4 primary sub-systems (Amber Data Layer, Purple DL Pipeline, Blue Frontend, Green Backend Inference API):

```mermaid
graph TD
    %% Styling Classes
    classDef dataLayer fill:#2a1c00,stroke:#f39c12,stroke-width:2px,color:#fff;
    classDef dlPipeline fill:#21002b,stroke:#a29bfe,stroke-width:2px,color:#fff;
    classDef frontendApp fill:#001d3d,stroke:#00b4d8,stroke-width:2px,color:#fff;
    classDef backendApi fill:#002810,stroke:#2ecc71,stroke-width:2px,color:#fff;
    classDef nodeBox fill:#1a1a24,stroke:#444,stroke-width:1px,color:#fff;

    %% 1. DATA LAYER
    subgraph DATA_LAYER ["1. DATA LAYER"]
        D1[("🛢️ Raw PVEL-AD Dataset<br/>(13,414 EL Images & Annotations)")] ::: nodeBox
        D2["🖼️ Data Augmentation & Normalization<br/>(Resize 224x224, Flips, Rotations, ColorJitter)"] ::: nodeBox
        D3[("📊 Processed 8-Class Dataset<br/>(Multi-Label Encoded Matrix)")] ::: nodeBox
        D1 --> D2 --> D3
    end
    class DATA_LAYER dataLayer;

    %% 2. DEEP LEARNING PIPELINE
    subgraph DL_PIPELINE ["2. DEEP LEARNING PIPELINE"]
        P1["🧠 EfficientNet-B0 Training Pipeline<br/>(MBConv Backbone, 4.02M Params)"] ::: nodeBox
        P2(("🎯 BCEWithLogitsLoss + Pos Weight<br/>(AdamW & Cosine Annealing)")) ::: nodeBox
        P3["📄 Trained Weights (.pth & .onnx)<br/>(48.7 MB PyTorch / 16 MB ONNX)"] ::: nodeBox
        P1 --> P2 --> P3
    end
    class DL_PIPELINE dlPipeline;

    %% 3. FRONTEND WEB APP
    subgraph FRONTEND_APP ["3. FRONTEND WEB APP"]
        F1["📊 Results Dashboard"] ::: nodeBox
        F2["☁️ EL Image Upload"] ::: nodeBox
        F3["⚛️ SolarSentinel Web App<br/>(HTML5 / CSS3 / JavaScript)"] ::: nodeBox
        F4["</> REST API JSON & Overlays"] ::: nodeBox
        F1 & F2 --> F3 --> F4
    end
    class FRONTEND_APP frontendApp;

    %% 4. BACKEND INFERENCE API
    subgraph BACKEND_API ["4. BACKEND INFERENCE API"]
        B1["🔌 Flask / FastAPI Server"] ::: nodeBox
        B2["🔥 PyTorch & ONNX Runtime Engine<br/>(140.6ms CPU Latency)"] ::: nodeBox
        B3["🎯 Grad-CAM Heatmap Generator<br/>(Defect Localization)"] ::: nodeBox
        B1 --> B2 --> B3
    end
    class BACKEND_API backendApi;

    %% Inter-Module Connections
    D3 --> P1
    P3 --> B2
    F4 --> B1
```

---

## 3. Detailed Component Breakdown

### 🟡 Layer 1: DATA LAYER (Data Preparation Pipeline)
1. **Raw PVEL-AD Dataset**:
   - **Input**: 13,414 high-resolution Near-Infrared Electroluminescence (EL) photovoltaic cell images.
   - **Source XML & Folders**: XML annotations from `train_PVELAD` combined with defect-free samples from `othertypes_PVELAD`.
2. **Data Augmentation & Multi-Label Encoding**:
   - Spatial transforms: `Resize((224, 224))`, `RandomHorizontalFlip(0.5)`, `RandomVerticalFlip(0.5)`, `RandomRotation(15°)`.
   - Photometric transforms: `ColorJitter(brightness=0.2, contrast=0.2)`.
   - Standardization: Normalized with ImageNet mean (`[0.485, 0.456, 0.406]`) and std (`[0.229, 0.224, 0.225]`).
3. **Processed 8-Class Dataset**:
   - Targets: Multi-label binary vectors representing 8 distinct categories:  
     `crack`, `finger`, `black_core`, `thick_line`, `star_crack`, `other_defect`, `good`, `material`.

---

### 🟣 Layer 2: DEEP LEARNING PIPELINE (Training & Optimization)
1. **EfficientNet-B0 Training Pipeline**:
   - **Architecture**: Mobile Inverted Bottleneck Convolution (`MBConv`) with Squeeze-and-Excitation (SE) channel attention blocks.
   - **Parameters**: 4.02 Million parameters; **0.39 GFLOPs**.
   - **Classification Head**: Global Average Pooling (GAP) ➔ Dropout (0.3) ➔ Linear (1280 ➔ 8 classes).
2. **BCEWithLogitsLoss & Optimization**:
   - **Loss Function**: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`) incorporating positive class weighting (`pos_weight`) to resolve severe imbalance in rare defects (e.g., `star_crack`, `other_defect`).
   - **Optimizer & Scheduler**: AdamW (`lr=1e-3`, `weight_decay=1e-4`) paired with `CosineAnnealingLR` (`eta_min=1e-6`).
3. **Trained Checkpoints**:
   - **PyTorch Model (`best_efficientnet_b0.pth`)**: 48.7 MB binary checkpoint (Epoch 10: **86.25% exact accuracy**, **0.7882 Macro F1**).
   - **ONNX Model (`efficientnet_b0_solar.onnx`)**: 16.0 MB optimized graph export (`opset_version=14`).

---

### 🔵 Layer 3: FRONTEND WEB APP (User Interface & Control)
1. **Results Dashboard & User Image Upload**:
   - Drag-and-drop interface for single/batch EL panel inspection.
   - Analytical widgets displaying classification metrics, confusion matrices, and confidence thresholds.
2. **SolarSentinel Web Application**:
   - Built with responsive HTML5, CSS3 glassmorphism design, and asynchronous JavaScript fetch engine.
3. **REST API Client**:
   - Serializes user uploaded images as multi-part form data and decodes JSON responses containing class probability distributions and defect overlays.

---

### 🟢 Layer 4: BACKEND INFERENCE API (Production Serving & Visualization)
1. **Flask / FastAPI Server**:
   - Endpoint routing for `/predict`, `/evaluate`, and `/heatmap`.
   - Asynchronous request handling and image validation.
2. **PyTorch & ONNX Runtime Engine**:
   - Dual execution provider supporting PyTorch CPU/GPU inference and ONNX Runtime execution (`~140.6 ms` per image latency on standard CPU).
3. **Grad-CAM Generator**:
   - Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the final MBConv feature maps to generate visual heatmaps pin-pointing exact micro-crack and defect locations.

---

## 4. Module Interaction & Data Flow

```
+------------------+         8-Class Data          +-------------------------+
|                  | ----------------------------> |                         |
|  1. DATA LAYER   |                               | 2. DEEP LEARNING        |
|                  |                               |    PIPELINE             |
+------------------+                               +-------------------------+
                                                                |
                                                                | Trained Weights (.pth / .onnx)
                                                                v
+------------------+         REST API (HTTP)       +-------------------------+
|                  | ----------------------------> |                         |
|  3. FRONTEND     |                               | 4. BACKEND INFERENCE    |
|     WEB APP      | <---------------------------- |    API                  |
+------------------+       JSON & Defect Maps      +-------------------------+
```
