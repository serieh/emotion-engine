# Emotion Engine

Machine learning classification pipeline and desktop application predicting dominant human emotional states from social media usage patterns.

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)
![Model](https://img.shields.io/badge/Model-Random%20Forest%20(97%25%20Acc)-brightgreen.svg)

---

## Overview

Social media usage patterns provide distinct behavioral signals that correlate with underlying emotional states. Emotion Engine analyzes demographic indicators (age, gender), active engagement metrics (posts per day, daily usage minutes, messages sent), and social feedback metrics (likes received, comments received) to classify a user's dominant emotion across six target categories:

- **Anger**
- **Anxiety**
- **Boredom**
- **Happiness**
- **Neutral**
- **Sadness**

The project includes an end-to-end data preprocessing pipeline with anomaly correction, synthetic oversampling for class balance (ADASYN), multi-model benchmarking (Decision Tree, Random Forest, KNN, SVM), a CLI inference interface, and a standalone desktop application.

---

## Key Features

- **Automated Data Cleaning**: Detects and repairs inverted demographic columns (e.g. gender strings misplaced in age fields), removes incomplete records, and coerces numeric metrics.
- **Per-Feature Standard Scaling**: Scales numerical features independently using dedicated `StandardScaler` instances preserved with model artifacts.
- **Class Balancing with ADASYN**: Balances skewed training class distributions dynamically prior to classifier training.
- **Multi-Model Benchmark**: Evaluates Decision Trees, Random Forests, K-Nearest Neighbors, and Support Vector Classifiers against holdout test data.
- **Inference Interfaces**:
  - **CLI (`emotion-engine`)**: Single-record prediction, batch CSV processing, model training, and scenario demonstrations.
  - **Desktop GUI (`emotion-engine-gui`)**: Tkinter application with real-time prediction, confidence scoring, and random test generation.

---

## Benchmark Results

Models were trained on the cleaned training dataset and evaluated against an independent holdout test set (101 samples):

| Model | Accuracy | Macro F1 | Weighted F1 | Evaluation Summary |
|---|---|---|---|---|
| **Random Forest (100 estimators)** | **97.03%** | **0.97** | **0.97** | **Top performer; selected as production classifier** |
| **K-Nearest Neighbors (k=2)** | 96.04% | 0.95 | 0.96 | Strong local neighborhood separation |
| **Decision Tree Classifier** | 94.06% | 0.94 | 0.95 | Fast baseline decision boundaries |
| **Support Vector Classifier (RBF)** | 69.31% | 0.70 | 0.69 | Underperforms on minority class boundaries |

### Production Model Performance (Random Forest)

```text
              precision    recall  f1-score   support

       Anger       1.00      1.00      1.00         9
     Anxiety       1.00      0.95      0.98        22
     Boredom       0.94      0.94      0.94        16
   Happiness       0.93      1.00      0.96        13
     Neutral       1.00      1.00      1.00        27
     Sadness       0.93      0.93      0.93        14

    accuracy                           0.97       101
   macro avg       0.97      0.97      0.97       101
weighted avg       0.97      0.97      0.97       101
```

---

## Repository Structure

```text
emotion-engine/
├── assets/
│   ├── icon.ico                 # Application window icon
│   └── icon.png                 # High-resolution application graphic
├── data/
│   ├── train.csv                # Primary training dataset
│   └── test.csv                 # Holdout test dataset
├── models/
│   ├── .gitkeep                 # Model artifact destination directory
│   ├── encoders/.gitkeep
│   └── scalers/.gitkeep
├── releases/
│   └── v1.0.0/
│       ├── README.md            # Release asset details and instructions
│       ├── SHA256SUMS           # Integrity checksums for model bundles
│       └── emotion_engine_models_v1.0.0.tar.gz  # Release artifact bundle

├── notebooks/
│   ├── model_trainer.ipynb      # Training and exploratory data analysis notebook
│   └── gui_prototype.ipynb      # Initial GUI exploration notebook
├── src/
│   └── emotion_engine/
│       ├── __init__.py          # Public package API
│       ├── cli.py               # CLI entry point
│       ├── config.py            # Global paths, constants, and feature lists
│       ├── gui.py               # Desktop GUI application
│       ├── models.py            # Model training and benchmark comparison
│       ├── predict.py           # Single-record and batch inference engine
│       └── preprocessor.py      # Data cleaning, encoding, and scaling utilities
├── tests/
│   ├── test_cli.py              # CLI integration tests
│   ├── test_models.py           # Training pipeline tests
│   ├── test_predict.py          # Prediction and validation tests
│   └── test_preprocessor.py     # Preprocessing and anomaly correction tests
├── .gitignore                   # Standard repository ignores
├── .python-version              # Pinned Python version (3.13)
├── LICENSE                      # MIT License
├── pyproject.toml               # Package configuration and dependencies
└── uv.lock                      # Resolved dependency lockfile
```

---

## Installation

This project requires Python 3.10+ and uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Sync Dependencies

```bash
git clone <repository-url>
cd emotion-engine
uv sync
```

---

## Usage

### 1. Single Record Prediction (CLI)

Predict the dominant emotion for a specific social media user profile:

```bash
uv run emotion-engine predict \
  --age 25 \
  --gender Female \
  --platform Instagram \
  --usage 120 \
  --posts 3 \
  --likes 45 \
  --comments 10 \
  --messages 12
```

Sample output:

```text
==================================================
 Emotion Engine - Prediction Result
==================================================
 Predicted Dominant Emotion: HAPPINESS
 Confidence:                 82.00%
--------------------------------------------------
 Class Probabilities:
   Happiness    82.00% | ████████████████████████
   Neutral      10.00% | ███
   Anxiety       5.00% | █
   Boredom       3.00% | 
   Anger         0.00% | 
   Sadness       0.00% | 
==================================================
```

### 2. Batch Prediction (CSV)

Run inference on an unlabelled or benchmark CSV file:

```bash
uv run emotion-engine batch --input data/test.csv --output results.csv
```

### 3. Run Benchmark Scenarios

Run predefined verification profiles representing diverse usage habits:

```bash
uv run emotion-engine demo
```

### 4. Train Models

Retrain all candidate classifiers and compare metrics:

```bash
uv run emotion-engine train
```

### 5. Desktop Application (GUI)

On desktop environments with a display server and Tk installed:

```bash
uv run emotion-engine-gui
```

---

## Pre-Trained Model Releases

To maintain a lean repository size, binary model weights (`.joblib`) are distributed via **GitHub Releases** rather than tracked directly in git history.

### Downloading & Unpacking Model Artifacts

Model artifacts are packaged in `releases/v1.0.0/emotion_engine_models_v1.0.0.tar.gz`. When running predictions, `EmotionPredictor` automatically extracts the archive into `models/` if available.

To unpack manually:

```bash
tar -xzvf releases/v1.0.0/emotion_engine_models_v1.0.0.tar.gz -C models/
```

To verify archive checksum:

```bash
cd releases/v1.0.0 && sha256sum -c SHA256SUMS
```

### Publishing to GitHub Releases

To attach the model archive to a GitHub Release:

```bash
gh release create v1.0.0 releases/v1.0.0/emotion_engine_models_v1.0.0.tar.gz \
    --title "v1.0.0 - Production Release" \
    --notes "Trained Random Forest classifier and feature preprocessing pipeline."
```

---

## Running Tests


Run the test suite and code linters:

```bash
uv run pytest
uv run ruff check .
```

---

## Feature Schema

| Feature | Type | Range / Domain | Description |
|---|---|---|---|
| `Age` | Integer | 0 – 120 | User age in years |
| `Gender` | Categorical | `Male`, `Female` | User gender |
| `Platform` | Categorical | `Instagram`, `Twitter`, `Facebook`, `LinkedIn`, `Snapchat`, `Telegram` | Primary social media platform |
| `Daily_Usage_Time (minutes)` | Float | ≥ 0 | Active daily screen time |
| `Posts_Per_Day` | Float | ≥ 0 | Number of public posts published daily |
| `Likes_Received_Per_Day` | Float | ≥ 0 | Total daily likes on user content |
| `Comments_Received_Per_Day`| Float | ≥ 0 | Total daily comments on user content |
| `Messages_Sent_Per_Day` | Float | ≥ 0 | Total private messages dispatched daily |

---

## Limitations

- **Behavioral Correlation vs. Clinical State**: The model predicts dominant situational emotion as manifested through social media interactions. It is not an instrument for clinical psychological diagnosis.
- **Platform Bias**: Engagement habits differ significantly across demographic cohorts and platform cultures (e.g. LinkedIn professional networking vs. Snapchat ephemeral messaging).
- **Tabular Scope**: The dataset relies on quantitative interaction rates rather than natural language text processing of post content.

---

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Ahmad Serieh**  
Email: [ahmadserieh.official@gmail.com](mailto:ahmadserieh.official@gmail.com)
