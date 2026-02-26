# Sleep Stage Classification (CRNN: CNN + Bi-GRU)

## Overview
Automatic sleep-stage classification using multi-channel EEG/EOG with a Convolutional + Bidirectional GRU (CRNN) pipeline. The repository contains processed datasets, training/evaluation notebooks, and trained model weights for reproducible experiments.

## Dataset Availability 💾
Due to GitHub's file size limits, the processed `.npz` files and the raw Sleep-EDF dataset are hosted externally on Google Drive. 

👉 **[Click here to download the datasets]([https://drive.google.com/drive/folders/1Nyg_bC2_i0LOA_sShilYWmRwH8h870Mb?usp=share_link])**

Please download the files and place them in the root directory of this project before running the notebooks. The downloaded folder includes:
- `train_smote_final.npz` — SMOTE-balanced training dataset (features + labels).
- `train_ros_final.npz` — ROS-balanced training dataset (features + labels).
- `test_final.npz` — test dataset with real-world class distribution.
- `sleep-edf-database-expanded-1.0.0/` — optional raw EDF files.

## Repository contents
- `EDA.ipynb` — exploratory data analysis and visualizations.
- `Sleep_Classification_Training.ipynb` — preprocessing, training, evaluation pipeline (notebook).
- `model_smote.pth` — trained model weights (SMOTE).
- `model_ros.pth` — trained model weights (ROS).
- `requirements.txt` — Python dependencies.

## Quick start
1. Create a Python 3.8+ environment and install dependencies:
```bash
pip install -r requirements.txt
