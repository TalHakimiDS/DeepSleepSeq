# DeepSleepSeq: Sleep Stage Classification from EEG/EOG

Automatic sleep staging (Wake, N1, N2, N3, REM) from raw polysomnography signals. It uses a **multi-branch CNN + bidirectional GRU (CRNN)** in PyTorch, built to handle the severe class imbalance of the N1 stage. The data is the Sleep-EDF Expanded dataset.

## Model: `SleepNetSeq`

```
input: 5 consecutive 30-s epochs (2.5 min) × 3000 samples × 3 channels @ 100 Hz
                     │
   ┌─────────────────┴──────────────────┐
EEG branch (Fpz-Cz, Pz-Oz)          EOG branch (horizontal)
Conv1d(2→32, k=50, s=5) + BN + ReLU   Conv1d(1→16, k=50, s=5) + BN + ReLU
MaxPool(8)                           MaxPool(8)
   └─────────────── concat (48 ch) ──────┘
          shared Conv1d(48→64, k=8) + BN + ReLU + MaxPool(4)
          global average pooling over time  → 64-d vector per epoch
          2-layer bidirectional GRU (128 hidden) over the 5-epoch sequence
          FC(256→64) + ReLU + Dropout(0.3) → FC(64→5)
```

- **Separate branches per signal type**: EEG and EOG have different frequency content, so each gets its own learned filters before fusion.
- **Sequence context**: the Bi-GRU sees the neighbouring epochs, the same context human scorers use (sleep-stage transition rules). The model classifies the last epoch of each window (many-to-one).

## Class imbalance

N1 is rare and hard to tell apart from Wake and REM. Approaches used:
- Class-weighted cross-entropy (`compute_class_weight('balanced')`).
- Two oversampling strategies for the training set: **SMOTE** (on flattened epochs) and **random oversampling (ROS)**. The test set keeps the real-world class distribution.

## Results

Classification report on a held-out evaluation set (1,943 sequences):

| Model | Accuracy | Macro F1 | N1 recall | N3 recall | REM recall |
|---|---|---|---|---|---|
| SMOTE | 0.84 | 0.79 | 0.62 | 0.91 | 0.86 |
| ROS | 0.78 | 0.75 | **0.82** | **0.97** | 0.85 |

Statistical comparison of the two models:
- **Wilcoxon signed-rank test** on F1 across 10 contiguous chunks of held-out data: ROS mean F1 0.835 vs SMOTE 0.811 (p ≈ 0.001).
- **McNemar's test** on per-epoch correctness over the full test set (38,856 epochs): the models differ significantly (p < 0.001). SMOTE is correct more often overall (4,402 vs 2,182 discordant epochs).

The two metrics favour different models. ROS gives much higher recall on the rare N1 stage, which matters clinically. SMOTE has higher overall accuracy. Which one to choose depends on whether sensitivity or overall accuracy matters more.

An external-validation notebook (`Hospital_SleepNet.ipynb`) reprocesses the raw Sleep-EDF recordings with MNE (AASM mapping: stages 3 and 4 merged into N3).

## Data

Because of GitHub's size limits, the processed `.npz` files and the raw Sleep-EDF data are on Google Drive:
**[Download the datasets](https://drive.google.com/drive/folders/19dJG4BRJRO24tp8b33eLNJF4e8-DIG_8?usp=share_link)**

- `train_smote_final.npz`: SMOTE-balanced training set
- `train_ros_final.npz`: ROS-balanced training set
- `test_final.npz`: test set with the real class distribution
- `sleep-edf-database-expanded-1.0.0/`: optional raw EDF files

Place them in the repository root before running the notebooks.

## Repository

| File | Purpose |
|---|---|
| `EDA.ipynb` | Exploratory analysis of signals and class distribution |
| `Sleep_Classification_Training.ipynb` | Preprocessing, training, evaluation and statistical tests |
| `Hospital_SleepNet.ipynb` | Raw EDF processing with MNE, external validation |
| `model_smote.pth`, `model_ros.pth`, `model_ros11.pth` | Trained weights |
| `fig1_physiological.pdf`, `fig5_hypnogram.pdf` | Figures |

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook Sleep_Classification_Training.ipynb
```

## Tech

PyTorch · MNE · scikit-learn · imbalanced-learn · SciPy · statsmodels · NumPy · Matplotlib / Seaborn
