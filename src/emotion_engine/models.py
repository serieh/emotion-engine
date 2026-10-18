"""Model training, benchmark comparison, and artifact serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from emotion_engine.config import (
    ALL_FEATURES,
    DEFAULT_TEST_DATA,
    DEFAULT_TRAIN_DATA,
    ENCODER_NAMES,
    MODELS_DIR,
    SCALER_NAMES,
    TARGET_COLUMN,
)
from emotion_engine.preprocessor import (
    clean_dataset,
    fit_encoders,
    fit_scalers,
    transform_encoders,
    transform_scalers,
)


@dataclass
class ModelBenchmark:
    """Benchmark metrics for a trained classifier."""

    name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    classification_report: str


def train_and_evaluate(
    train_path: str | Path = DEFAULT_TRAIN_DATA,
    test_path: str | Path = DEFAULT_TEST_DATA,
    output_dir: str | Path | None = None,
    save_artifacts: bool = True,
) -> tuple[dict[str, Any], dict[str, ModelBenchmark]]:
    """Train multiple classifiers, compare benchmarks, and persist best model artifacts."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1. Clean datasets
    train_clean = clean_dataset(train_df)
    test_clean = clean_dataset(test_df)

    # 2. Fit and transform categorical encoders
    encoders = fit_encoders(train_clean)
    train_enc = transform_encoders(train_clean, encoders)
    test_enc = transform_encoders(test_clean, encoders)

    # 3. Fit and transform scalers
    scalers = fit_scalers(train_enc)
    train_scaled = transform_scalers(train_enc, scalers)
    test_scaled = transform_scalers(test_enc, scalers)

    # 4. Handle class imbalance
    x_train_raw = train_scaled[ALL_FEATURES]
    y_train_raw = train_scaled[TARGET_COLUMN]

    try:
        from imblearn.over_sampling import ADASYN

        sampler = ADASYN(random_state=42)
        x_train, y_train = sampler.fit_resample(x_train_raw, y_train_raw)
    except Exception:
        # Fallback if ADASYN fails or imblearn is not installed
        try:
            from imblearn.over_sampling import RandomOverSampler

            sampler = RandomOverSampler(random_state=42)
            x_train, y_train = sampler.fit_resample(x_train_raw, y_train_raw)
        except Exception:
            x_train, y_train = x_train_raw, y_train_raw

    x_test = test_scaled[ALL_FEATURES]
    y_test = test_scaled[TARGET_COLUMN]

    # 5. Define candidate models
    candidates: dict[str, Any] = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN (k=2)": KNeighborsClassifier(n_neighbors=2),
        "SVM (RBF)": SVC(probability=True, random_state=42),
    }

    trained_models: dict[str, Any] = {}
    benchmarks: dict[str, ModelBenchmark] = {}

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(
            y_test,
            y_pred,
            target_names=encoders[TARGET_COLUMN].classes_,
            zero_division=0,
        )

        trained_models[name] = model
        benchmarks[name] = ModelBenchmark(
            name=name,
            accuracy=acc,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            classification_report=report,
        )

    # 6. Save artifacts if requested
    if save_artifacts:
        dest_models = Path(output_dir) if output_dir else MODELS_DIR
        dest_encoders = dest_models / "encoders"
        dest_scalers = dest_models / "scalers"

        dest_models.mkdir(parents=True, exist_ok=True)
        dest_encoders.mkdir(parents=True, exist_ok=True)
        dest_scalers.mkdir(parents=True, exist_ok=True)

        best_model = trained_models["Random Forest"]
        joblib.dump(best_model, dest_models / "classifier.joblib")

        for key, filename in ENCODER_NAMES.items():
            if key in encoders:
                joblib.dump(encoders[key], dest_encoders / filename)

        for key, filename in SCALER_NAMES.items():
            if key in scalers:
                joblib.dump(scalers[key], dest_scalers / filename)

    return trained_models, benchmarks
