# Sleep Stage Classification with CNN + Bi-GRU

An academic deep-learning project exploring sleep-stage classification from multi-channel EEG/EOG signals using a convolutional and recurrent architecture.

## Technical focus
- Signal exploration and preprocessing.
- A PyTorch CNN + bidirectional GRU sequence model.
- Comparison of SMOTE and random oversampling (ROS) training variants.
- Notebook-based training, evaluation, and visualization.

## Repository guide
| File | Purpose |
| --- | --- |
| `EDA.ipynb` | Exploratory analysis and visualizations |
| `Sleep_Classification_Training.ipynb` | Training and evaluation workflow |
| `Hospital_SleepNet.ipynb` | Additional experimental notebook |
| `model_smote.pth` | Saved weights for a SMOTE training variant |
| `model_ros.pth` | Saved weights for a ROS training variant |
| `requirements.txt` | Python dependencies |

## Setup
Create a Python virtual environment, then install the listed dependencies:
```bash
pip install -r requirements.txt
```
Open the notebooks in a Jupyter-compatible environment. Review file paths, data shapes, and configuration before running cells. A fresh-environment execution and exact dependency pinning have not yet been documented.

## Data
The original project README provides an external dataset folder:
[Project dataset folder](https://drive.google.com/drive/folders/19dJG4BRJRO24tp8b33eLNJF4e8-DIG_8?usp=share_link).

Expected processed inputs include:
- `train_smote_final.npz`
- `train_ros_final.npz`
- `test_final.npz`

Raw Sleep-EDF files are optional depending on the notebook workflow. Access, contents, redistribution rights, and suitability of the external folder must be checked separately. Use only datasets you are authorized to access.

## Evaluation and limitations
The notebooks are the source of experimental results. This README does not claim a new benchmark or independently reproduced score. Before interpreting generalization, verify subject-level train/test separation, preprocessing boundaries, and that oversampling is confined to training data.

This is an academic research project, not a clinical diagnostic system.
