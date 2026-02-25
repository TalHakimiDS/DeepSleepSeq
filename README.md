# Sleep Stage Classification (CRNN: CNN + Bi-GRU)

## Overview
Automatic sleep-stage classification using multi-channel EEG/EOG with a Convolutional + Bidirectional GRU (CRNN) pipeline. The repository contains processed datasets, training/evaluation notebooks, and trained model weights for reproducible experiments.

## Repository contents
- [EDA.ipynb](EDA.ipynb) — exploratory data analysis and visualizations.
- [Sleep_Classification_Training.ipynb](Sleep_Classification_Training.ipynb) — preprocessing, training, evaluation pipeline (notebook).
- [train_smote_final.npz](train_smote_final.npz) — SMOTE-balanced training dataset (features + labels).
- [train_ros_final.npz](train_ros_final.npz) — ROS-balanced training dataset (features + labels).
- [test_final.npz](test_final.npz) — test dataset with real-world class distribution.
- [model_smote.pth](model_smote.pth) — trained model weights (SMOTE).
- [model_ros.pth](model_ros.pth) — trained model weights (ROS).
- [requirements.txt](requirements.txt) — Python dependencies.
- [sleep-edf-database-expanded-1.0.0](sleep-edf-database-expanded-1.0.0/) — optional raw EDF files (placed here if you download raw data).
- [README.md](README.md) — this file.

## Quick start
1. Create a Python 3.8+ environment and install deps:
```bash
# language: bash
pip install -r [requirements.txt](http://_vscodecontentref_/0)