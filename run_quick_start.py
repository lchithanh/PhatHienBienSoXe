"""Quick start script for DoAnAi project"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

print("\n=== Quick Start - DoAnAi License Plate System ===\n")
print("1. Ensure dependencies are installed:")
print("   pip install -r requirements.txt")
print("")
print("2. Reorganize dataset (70/30):")
print("   python scripts/reorganize_dataset.py")
print("")
print("3. Train model (short test or full):")
print("   python scripts/train_plate_model_final.py")
print("")
print("4. Run detection GUI:")
print("   python apps/detect_gui.py")
print("")
print("5. Run webcam detection:")
print("   python apps/detect_webcam.py")
print("")
print("If first time, use CPU or GPU as available.\n")
print("For now we support: config path fallback, new advanced OCR, enhanced training and tracking.")
