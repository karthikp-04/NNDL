"""
SolarSentinel AI — Master End-to-End Pipeline Orchestrator
Automates environment verification, dataset preparation, reproducible splitting,
cDCGAN training, synthetic sample generation, dual classifier training (Exp A vs Exp B),
held-out test set evaluation, comparative reporting, and automated test execution.
"""

import os
import sys
import time
import argparse
from datetime import datetime
import pandas as pd
import torch

import config
from src.preprocessing import create_reproducible_splits
from src.train_gan import train_gan
from src.generate_images import generate_synthetic_samples
from src.train_classifier import train_classifier
from src.evaluate import evaluate_model
from src.compare_models import compare_experiments
import run_tests

def print_banner(step_num, title):
    print("\n" + "=" * 80)
    print(f"STEP {step_num}: {title.upper()}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="SolarSentinel AI Master Pipeline Orchestrator")
    parser.add_argument('--smoke-test', action='store_true', help="Execute fast end-to-end smoke test")
    parser.add_argument('--force-retrain-gan', action='store_true', help="Force retraining of cDCGAN")
    parser.add_argument('--force-retrain-baseline', action='store_true', help="Force retraining of baseline model")
    parser.add_argument('--force-retrain-augmented', action='store_true', help="Force retraining of augmented model")
    parser.add_argument('--gan-epochs', type=int, default=config.GAN_EPOCHS, help="GAN epochs")
    parser.add_argument('--clf-epochs', type=int, default=config.CLASSIFIER_EPOCHS, help="Classifier epochs")
    parser.add_argument('--syn-count', type=int, default=30, help="Synthetic samples per minority class")
    args = parser.parse_args()

    pipeline_start = time.time()
    print("=" * 80)
    print("SOLARSENTINEL AI — END-TO-END GENERATIVE AI PIPELINE")
    print(f"Execution Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'SMOKE-TEST' if args.smoke_test else 'FULL WORKFLOW'}")
    print(f"Target Hardware: {config.DEVICE}")
    print("=" * 80)

    # -------------------------------------------------------------
    # Step 1: Environment & Dependency Check
    # -------------------------------------------------------------
    print_banner(1, "Environment & Hardware Check")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CPU Threads Allocated: {torch.get_num_threads()}")

    # -------------------------------------------------------------
    # Step 2: Dataset Verification & Partitioning
    # -------------------------------------------------------------
    print_banner(2, "Dataset Verification & Split Creation")
    if not os.path.exists(config.DATASET_INDEX_CSV):
        print(f"ERROR: Dataset index CSV not found at {config.DATASET_INDEX_CSV}")
        print("Please ensure PVEL-AD images are linked and dataset_index.csv is generated.")
        sys.exit(1)

    df_splits = create_reproducible_splits(
        index_csv=config.DATASET_INDEX_CSV,
        splits_csv=config.SPLITS_INDEX_CSV
    )
    print(f"Validated dataset: {len(df_splits):,} total samples.")

    # -------------------------------------------------------------
    # Step 3: Train / Verify Conditional DCGAN
    # -------------------------------------------------------------
    print_banner(3, "Conditional DCGAN (cDCGAN) Training")
    if os.path.exists(config.GENERATOR_CKPT) and not args.force_retrain_gan and not args.smoke_test:
        print(f"Found existing Generator checkpoint: {config.GENERATOR_CKPT}. Skipping GAN training.")
    else:
        print("Training cDCGAN on real solar panel electroluminescence images...")
        train_gan(
            epochs=2 if args.smoke_test else args.gan_epochs,
            batch_size=config.GAN_BATCH_SIZE,
            smoke_test=args.smoke_test,
            subset_size=150 if args.smoke_test else None,
            focus_defects=True,
            device=config.DEVICE
        )

    # -------------------------------------------------------------
    # Step 4: Generate Quality-Filtered Synthetic Samples
    # -------------------------------------------------------------
    print_banner(4, "Synthetic Image Generation & Quality Filtering")
    manifest_path = os.path.join(config.GENERATED_SAMPLES_DIR, "synthetic_manifest.csv")
    if os.path.exists(manifest_path) and not args.smoke_test and not args.force_retrain_gan:
        print(f"Found existing synthetic manifest with samples at {manifest_path}. Reusing samples.")
    else:
        generate_synthetic_samples(
            num_per_defect=5 if args.smoke_test else args.syn_count,
            classes=config.MINORITY_CLASSES,
            device=config.DEVICE
        )

    # -------------------------------------------------------------
    # Step 5: Train / Verify Experiment A (Baseline EfficientNet-B0)
    # -------------------------------------------------------------
    print_banner(5, "Experiment A: Baseline Classifier (Real Data Only)")
    if os.path.exists(config.BASELINE_CKPT) and not args.force_retrain_baseline and not args.smoke_test:
        print(f"Found existing Baseline checkpoint: {config.BASELINE_CKPT}. Reusing trained weights.")
    else:
        print("Training Baseline EfficientNet-B0...")
        train_classifier(
            mode='baseline',
            epochs=1 if args.smoke_test else args.clf_epochs,
            batch_size=config.CLASSIFIER_BATCH_SIZE,
            smoke_test=args.smoke_test,
            device=config.DEVICE
        )

    # -------------------------------------------------------------
    # Step 6: Train Experiment B (GAN-Augmented EfficientNet-B0)
    # -------------------------------------------------------------
    print_banner(6, "Experiment B: GAN-Augmented Classifier (Real + Synthetic)")
    if os.path.exists(config.AUGMENTED_CKPT) and not args.force_retrain_augmented and not args.smoke_test:
        print(f"Found existing GAN-Augmented checkpoint: {config.AUGMENTED_CKPT}. Reusing weights.")
    else:
        print("Training GAN-Augmented EfficientNet-B0...")
        train_classifier(
            mode='augmented',
            epochs=1 if args.smoke_test else args.clf_epochs,
            batch_size=config.CLASSIFIER_BATCH_SIZE,
            smoke_test=args.smoke_test,
            device=config.DEVICE
        )

    # -------------------------------------------------------------
    # Step 7: Evaluate Both Models on Held-Out Test Set
    # -------------------------------------------------------------
    print_banner(7, "Held-Out Test Set Evaluation (Zero Leakage)")
    print("--- Evaluating Baseline Model ---")
    base_eval = evaluate_model(
        checkpoint_path=config.BASELINE_CKPT,
        model_name="Baseline",
        device=config.DEVICE
    )

    print("--- Evaluating GAN-Augmented Model ---")
    aug_eval = evaluate_model(
        checkpoint_path=config.AUGMENTED_CKPT,
        model_name="GAN_Augmented",
        device=config.DEVICE
    )

    # -------------------------------------------------------------
    # Step 8: Comparative Analysis (Exp A vs Exp B)
    # -------------------------------------------------------------
    print_banner(8, "Controlled Comparative Analysis")
    global_df, per_class_df = compare_experiments(
        baseline_json=os.path.join(config.METRICS_DIR, "baseline_evaluation.json"),
        augmented_json=os.path.join(config.METRICS_DIR, "gan_augmented_evaluation.json")
    )

    # -------------------------------------------------------------
    # Step 9: Run Automated Testing Suite
    # -------------------------------------------------------------
    print_banner(9, "Automated Verification Testing Suite")
    test_success = run_tests.run_all_tests()
    if not test_success:
        print("WARNING: Some test scenarios reported issues. Inspect outputs/reports/test_report.md.")

    # -------------------------------------------------------------
    # Step 10: Final Summary
    # -------------------------------------------------------------
    total_pipeline_time = time.time() - pipeline_start
    print_banner(10, "Pipeline Execution Completed")
    print(f"Total Execution Time: {total_pipeline_time:.2f} seconds ({total_pipeline_time/60.0:.2f} minutes)")
    print(f"Baseline Macro F1: {base_eval['macro_f1']:.4f}")
    print(f"Augmented Macro F1: {aug_eval['macro_f1']:.4f}")
    delta_f1 = aug_eval['macro_f1'] - base_eval['macro_f1']
    print(f"Delta Macro F1: {delta_f1:+.4f}")
    print(f"\nArtifacts generated:")
    print(f"  - Checkpoints:      {config.CHECKPOINTS_DIR}")
    print(f"  - Synthetic Images: {config.GENERATED_SAMPLES_DIR}")
    print(f"  - Metric Summaries: {config.METRICS_DIR}")
    print(f"  - Plots & Figures:  {config.PLOTS_DIR}")
    print(f"  - Final Reports:    {config.REPORTS_DIR}")
    print("\nTo launch the interactive dashboard, run:")
    print("    streamlit run app.py")

if __name__ == '__main__':
    main()
