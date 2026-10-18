"""Inference engine for dominant emotion prediction from social metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from emotion_engine.config import (
    ENCODER_NAMES,
    MODELS_DIR,
    SCALER_NAMES,
    SUPPORTED_GENDERS,
    SUPPORTED_PLATFORMS,
)
from emotion_engine.preprocessor import prepare_inference_frame


@dataclass
class EmotionRecord:
    """User demographics and daily social media behavior metrics."""

    age: int
    gender: str
    platform: str
    daily_usage_time: float
    posts_per_day: float
    likes_received_per_day: float
    comments_received_per_day: float
    messages_sent_per_day: float

    def to_raw_dict(self) -> dict[str, Any]:
        """Convert to dataset column key names."""
        return {
            "Age": self.age,
            "Gender": self.gender,
            "Platform": self.platform,
            "Daily_Usage_Time (minutes)": self.daily_usage_time,
            "Posts_Per_Day": self.posts_per_day,
            "Likes_Received_Per_Day": self.likes_received_per_day,
            "Comments_Received_Per_Day": self.comments_received_per_day,
            "Messages_Sent_Per_Day": self.messages_sent_per_day,
        }

    def validate(self) -> None:
        """Validate input values against domain boundaries."""
        if self.age < 0 or self.age > 120:
            raise ValueError(f"Age must be between 0 and 120, got {self.age}")
        if self.gender not in SUPPORTED_GENDERS:
            raise ValueError(
                f"Gender '{self.gender}' not supported. Expected one of: {SUPPORTED_GENDERS}"
            )
        if self.platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{self.platform}' not supported. Expected one of: {SUPPORTED_PLATFORMS}"
            )
        if self.daily_usage_time < 0:
            raise ValueError("Daily usage time cannot be negative")


@dataclass
class PredictionResult:
    """Predicted emotion and associated class probabilities."""

    predicted_emotion: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmotionPredictor:
    """Loads trained artifacts and performs emotion classification."""

    def __init__(self, models_dir: str | Path | None = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.encoders_dir = self.models_dir / "encoders"
        self.scalers_dir = self.models_dir / "scalers"
        self.model_path = self.models_dir / "classifier.joblib"

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if not self.model_path.exists():
            # Check if release bundle is available locally
            bundle_name = "emotion_engine_models_v1.0.0.tar.gz"
            release_bundle = self.models_dir.parent / "releases" / "v1.0.0" / bundle_name
            if release_bundle.exists():
                import tarfile
                with tarfile.open(release_bundle, "r:gz") as tar:
                    tar.extractall(path=self.models_dir)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. "
                "Download the release bundle from GitHub Releases, extract the archive "
                "into models/, or run 'emotion-cli train' to train locally."
            )


        self.model = joblib.load(self.model_path)


        self.encoders: dict[str, LabelEncoder] = {}
        for key, filename in ENCODER_NAMES.items():
            path = self.encoders_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing required encoder artifact: {path}")
            self.encoders[key] = joblib.load(path)

        self.scalers: dict[str, StandardScaler] = {}
        for key, filename in SCALER_NAMES.items():
            path = self.scalers_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing required scaler artifact: {path}")
            self.scalers[key] = joblib.load(path)

        self.classes_: list[str] = list(self.encoders["Dominant_Emotion"].classes_)

    def predict(self, record: EmotionRecord | dict[str, Any]) -> PredictionResult:
        """Predict the dominant emotion for a single user record."""
        if isinstance(record, EmotionRecord):
            record.validate()
            raw_dict = record.to_raw_dict()
        else:
            raw_dict = record

        df_input = prepare_inference_frame(raw_dict, self.encoders, self.scalers)

        pred_idx = self.model.predict(df_input)[0]
        emotion = self.encoders["Dominant_Emotion"].inverse_transform([pred_idx])[0]

        if hasattr(self.model, "predict_proba"):
            probs_arr = self.model.predict_proba(df_input)[0]
            confidence = float(np.max(probs_arr))
            probabilities = {
                self.classes_[i]: float(probs_arr[i]) for i in range(len(self.classes_))
            }
        else:
            confidence = 1.0
            probabilities = {emotion: 1.0}

        return PredictionResult(
            predicted_emotion=emotion,
            confidence=confidence,
            probabilities=probabilities,
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict dominant emotions for an entire DataFrame."""
        results = []
        for _, row in df.iterrows():
            rec_dict = {
                "Age": row["Age"],
                "Gender": row["Gender"],
                "Platform": row["Platform"],
                "Daily_Usage_Time (minutes)": row.get(
                    "Daily_Usage_Time (minutes)", row.get("daily_usage_time", 0)
                ),
                "Posts_Per_Day": row.get("Posts_Per_Day", row.get("posts_per_day", 0)),
                "Likes_Received_Per_Day": row.get(
                    "Likes_Received_Per_Day", row.get("likes_received_per_day", 0)
                ),
                "Comments_Received_Per_Day": row.get(
                    "Comments_Received_Per_Day", row.get("comments_received_per_day", 0)
                ),
                "Messages_Sent_Per_Day": row.get(
                    "Messages_Sent_Per_Day", row.get("messages_sent_per_day", 0)
                ),
            }
            res = self.predict(rec_dict)
            results.append((res.predicted_emotion, res.confidence))

        out_df = df.copy()
        out_df["Predicted_Emotion"] = [r[0] for r in results]
        out_df["Confidence"] = [r[1] for r in results]
        return out_df
