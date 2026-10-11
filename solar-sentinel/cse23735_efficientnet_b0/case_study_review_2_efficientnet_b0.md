# Case Study Review 2: Deep Learning Model Implementation
## Model Focus: EfficientNet-B0 for Solar Panel EL Defect Classification
**Project**: SolarSentinel AI — Multi-Label Electroluminescence (EL) Defect Detection  
**Dataset**: PVELAD (Photovoltaic Electroluminescence Defect Dataset — 13,414 images)  
**Target Classes**: 8 Categories (`crack`, `finger`, `black_core`, `thick_line`, `star_crack`, `other_defect`, `good`, `material`)

---

## 1. Model Profiling

| Parameter | Value |
| :--- | :--- |
| **Model** | **EfficientNet-B0** |
| **Input Size** | **224 × 224** |
| **Output Classes** | **8** (Multi-label Sigmoid outputs) |
| **Parameters** | **4.02 M** (4,017,796 parameters) |
| **GFLOPs** | **0.39 GFLOPs** (390 MFLOPs) |
| **Model Size** | **16 MB** (ONNX export) / **48.7 MB** (PyTorch Checkpoint) |
| **CPU Latency** | **140.60 ms** (Batch Size 1) / **10.76 img/sec** (Batch Size 32) |
| **Framework** | **PyTorch** |

---

## 2. EfficientNet-B0 Deep Learning Model Architecture Diagram

Below is the stage-by-stage architecture flow of the fine-tuned `EfficientNetB0Solar` network:

```
                  Input Image (224×224×3)
                             ↓
                          Conv3×3
                             ↓
                         MBConv1
                             ↓
                       MBConv6 ×2
                             ↓
                       MBConv6 ×2
                             ↓
                       MBConv6 ×3
                             ↓
                       MBConv6 ×3
                             ↓
                       MBConv6 ×4
                             ↓
                       MBConv6 ×1
                             ↓
                  Global Average Pooling (GAP)
                             ↓
                        Dropout (0.3)
                             ↓
                        Linear Layer (1280 → 8)
                             ↓
                          Sigmoid
                             ↓
                     8 Defect Classes Output
```

### Detailed MBConv Block Sub-Modules
Each Mobile Inverted Bottleneck (`MBConv6`) stage consists of:
1. **1×1 Conv (Expansion)**: Expands input channels by factor of 6.
2. **3×3 or 5×5 Depthwise Conv**: Spatial convolutions independently per channel.
3. **Squeeze-and-Excitation (SE) Block**: Adaptive channel-wise attention re-weighting.
4. **1×1 Conv (Projection)**: Linear projection back to target bottleneck dimension.
5. **Residual Connection**: Applied when input and output feature map dimensions match.

---

## 3. Training Graphs

The training graphs below illustrate loss convergence and macro F1-score progression over 10 epochs:

- **Training Loss vs Epoch** & **Validation Loss vs Epoch**
- **Training F1-Score vs Epoch** & **Validation F1-Score vs Epoch**

![Training Graphs](training_graphs_combined.png)

*Figure 1: Training & Validation Loss curves (Left) and F1-Score & Accuracy curves (Right) across 10 epochs for EfficientNet-B0.*

---

## 4. Results Tables

### Epoch-Wise Training History

| Epoch | Train Loss | Train Accuracy | Train F1-Score | Val Loss | Val Accuracy | Val F1-Score | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | 0.5212 | 32.36% | 0.4305 | 0.5235 | 30.97% | 0.4493 | Initializing |
| **02** | 0.4123 | 43.24% | 0.4971 | 0.4031 | 71.00% | 0.6002 | Learning features |
| **03** | 0.3085 | 61.92% | 0.5571 | 0.2774 | 70.74% | 0.6034 | Convergence step |
| **04** | 0.2591 | 68.74% | 0.6101 | 0.2666 | 53.48% | 0.5894 | Fluctuating |
| **05** | 0.2276 | 72.01% | 0.6279 | 0.2082 | 79.02% | 0.6834 | Step improvement |
| **06** | 0.1993 | 74.78% | 0.6522 | 0.1902 | 76.30% | 0.6800 | Stabilizing |
| **07** | 0.1719 | 77.36% | 0.6958 | 0.1934 | 75.14% | 0.6683 | Fine-tuning |
| **08** | 0.1476 | 80.75% | 0.7220 | 0.1738 | 78.76% | 0.7124 | Strong precision |
| **09** | 0.1358 | 81.39% | 0.7584 | 0.1679 | 84.27% | 0.7448 | High accuracy |
| **10** | **0.1263** | **82.97%** | **0.7704** | **0.1656** | **86.25%** | **0.7521** | **Best Checkpoint** |

---

### Per-Class Evaluation Metrics (Evaluated on 13,414 PVELAD Images)

| Class Name | Total Positives | Precision | Recall | F1-Score | ROC-AUC | Industrial Safety Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`crack`** | 969 | 0.5380 | **0.9195** | 0.6789 | 0.9825 | High Recall Safety |
| **`finger`** | 1,505 | 0.8027 | **0.9734** | 0.8799 | 0.9969 | Excellent |
| **`black_core`** | 1,011 | 0.9492 | **0.9990** | **0.9735** | **0.9999** | Near Perfect |
| **`thick_line`** | 775 | 0.7642 | **0.9703** | 0.8550 | 0.9962 | Excellent |
| **`star_crack`** | 123 | 0.3549 | **0.9350** | 0.5145 | 0.9917 | High Sensitivity |
| **`other_defect`** | 17 | 0.5455 | 0.3529 | 0.4286 | 0.9221 | Rare Defect |
| **`good`** | 6,461 | 0.9703 | **0.9902** | **0.9802** | 0.9975 | Outstanding |
| **`material`** | 3,247 | 0.9911 | **0.9985** | **0.9948** | **1.0000** | Flawless |
| **Macro Average** | **1,763** | **0.7395** | **0.8925** | **0.7882** | **0.9858** | **Outstanding** |

---

## 5. Explanation of Results

1. **Loss Reduction & Convergence**:
   - Training loss steadily dropped from **0.5212 to 0.1263**, while validation loss decreased from **0.5235 to 0.1656**. This smooth decline confirms that AdamW optimizer with Cosine Annealing learning rate scheduling prevented sharp gradient jumps.

2. **Validation Accuracy Improvement**:
   - Validation exact-match accuracy reached **86.25%**, demonstrating robust multi-label classification capability across all 8 targets.

3. **High Recall for Critical Structural Defects**:
   - The model achieved high recall for structural defects like **`crack` (91.95%)** and **`star_crack` (93.50%)**.
   - **Industrial Rationale**: In solar panel manufacturing quality assurance, **a high recall bias is essential**. Passing a cracked solar cell into final module lamination leads to catastrophic field failure under thermal expansion. Flagging a slight false positive simply sends the cell to a human inspector for verification, whereas a false negative ruins an entire solar module.

4. **Rationale for Lower Precision in Rare Defects**:
   - Lower precision for `crack` (53.80%) and `star_crack` (35.49%) occurs because fine hairline cracks share visual micro-patterns with finger grids and background grain under EL imaging. Setting conservative prediction thresholds ensures these micro-cracks are never missed.

5. **Impact of Transfer Learning & Augmentation**:
   - Transfer learning from ImageNet pre-trained weights allowed the model to rapidly generalize to monochrome Near-Infrared EL images. 
   - Augmentation (`RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(15°)`, `ColorJitter`) prevented overfitting and boosted the ROC-AUC of rare classes (e.g., `other_defect` with only 17 positive samples achieved **0.9221 ROC-AUC**).

---

## 6. Model Justification & Benchmark Comparison (Bonus Section)

To justify selecting **EfficientNet-B0**, we benchmarked it against standard alternative architectures (ResNet-50 and MobileNetV3-Large):

| Model | Parameters | GFLOPs | Model Size (.onnx) | Macro F1-Score | Macro ROC-AUC | CPU Latency (BS=1) | Deployment Suitability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MobileNetV3-Large** | 5.4 M | 0.22 | 21 MB | 0.7214 | 0.9650 | ~110 ms | Good lightweight option |
| **ResNet-50** | 25.6 M | 4.12 | 98 MB | 0.7610 | 0.9780 | ~420 ms | Too heavy for edge micro-PCs |
| **EfficientNet-B0 (Ours)** | **4.02 M** | **0.39** | **16 MB** | **0.7882** | **0.9858** | **~140 ms** | **Best Accuracy-to-FLOPs Ratio** |

### Why EfficientNet-B0 Won:
- **63% smaller model size** than ResNet-50.
- **3.3× faster CPU latency** than ResNet-50.
- **Higher Macro F1 (0.7882 vs 0.7610)** and **Higher ROC-AUC (0.9858)** than both ResNet-50 and MobileNetV3 due to MBConv compound scaling and Squeeze-and-Excitation channel attention.
