"""
SolarSentinel AI — Central Configuration Module
Defines system paths, dataset hyperparameters, model architectures, and training settings.
"""

import os
import torch

# Base Directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
GENERATED_SAMPLES_DIR = os.path.join(PROJECT_ROOT, "generated_samples")
SYNTHETIC_DATASET_DIR = os.path.join(GENERATED_SAMPLES_DIR, "synthetic_dataset")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

# Ensure required directories exist
for directory in [DATA_DIR, CHECKPOINTS_DIR, GENERATED_SAMPLES_DIR,
                  SYNTHETIC_DATASET_DIR, OUTPUTS_DIR, PLOTS_DIR, METRICS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset Paths
DATASET_INDEX_CSV = os.path.join(PROJECT_ROOT, "dataset_index.csv")
SPLITS_INDEX_CSV = os.path.join(DATA_DIR, "splits_index.csv")
TRAIN_PVELAD_DIR = os.path.join(PROJECT_ROOT, "train_PVELAD")
OTHERTYPES_PVELAD_DIR = os.path.join(PROJECT_ROOT, "othertypes_PVELAD")
TEST_PVELAD_DIR = os.path.join(PROJECT_ROOT, "test_PVELAD")

# Target Defect Classification Categories (8 Classes)
CLASS_NAMES = [
    'crack',
    'finger',
    'black_core',
    'thick_line',
    'star_crack',
    'other_defect',
    'good',
    'material'
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Minority defect classes prioritized for GAN synthetic augmentation
DEFECT_CLASSES = ['crack', 'finger', 'black_core', 'thick_line', 'star_crack', 'other_defect']
MINORITY_CLASSES = ['star_crack', 'other_defect', 'thick_line', 'crack']

# Image Preprocessing Hyperparameters
CLASSIFIER_IMG_SIZE = 224
GAN_IMG_SIZE = 64
IMAGE_CHANNELS = 3

# Normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GAN_MEAN = [0.5, 0.5, 0.5]
GAN_STD = [0.5, 0.5, 0.5]

# Hardware & Execution Environment
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0  # 0 is recommended for robust multi-threading on Windows
RANDOM_SEED = 42

# GAN Hyperparameters (Conditional DCGAN)
GAN_LATENT_DIM = 100
GAN_EMBED_DIM = 64
GAN_FEATURE_MAP_G = 64
GAN_FEATURE_MAP_D = 64
GAN_LR = 0.0002
GAN_BETA1 = 0.5
GAN_BETA2 = 0.999
GAN_EPOCHS = 15
GAN_BATCH_SIZE = 32
GAN_LABEL_SMOOTHING = 0.9

# Classifier Hyperparameters (EfficientNet-B0)
CLASSIFIER_LR = 1e-3
CLASSIFIER_WEIGHT_DECAY = 1e-4
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_EPOCHS = 5
CLASSIFIER_DROPOUT = 0.3
DECISION_THRESHOLD = 0.5

# Checkpoint Filenames
BASELINE_CKPT = os.path.join(CHECKPOINTS_DIR, "baseline_efficientnet_b0.pth")
AUGMENTED_CKPT = os.path.join(CHECKPOINTS_DIR, "gan_augmented_efficientnet_b0.pth")
GENERATOR_CKPT = os.path.join(CHECKPOINTS_DIR, "generator.pth")
DISCRIMINATOR_CKPT = os.path.join(CHECKPOINTS_DIR, "discriminator.pth")

# Evaluation & Metrics Output Paths
BASELINE_EVAL_CSV = os.path.join(METRICS_DIR, "baseline_evaluation.csv")
AUGMENTED_EVAL_CSV = os.path.join(METRICS_DIR, "augmented_evaluation.csv")
MODEL_COMPARISON_CSV = os.path.join(METRICS_DIR, "model_comparison_summary.csv")
GAN_HISTORY_CSV = os.path.join(METRICS_DIR, "gan_training_history.csv")
TEST_REPORT_MD = os.path.join(REPORTS_DIR, "test_report.md")
FINAL_EVAL_REPORT_MD = os.path.join(REPORTS_DIR, "final_lab_evaluation_report.md")
