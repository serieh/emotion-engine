"""Emotion Engine: Machine learning emotion classification from social media usage."""

from emotion_engine.models import ModelBenchmark, train_and_evaluate
from emotion_engine.predict import EmotionPredictor, EmotionRecord, PredictionResult

__all__ = [
    "EmotionPredictor",
    "EmotionRecord",
    "PredictionResult",
    "ModelBenchmark",
    "train_and_evaluate",
]

__version__ = "1.0.0"
