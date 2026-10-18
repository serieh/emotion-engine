"""Unit tests for inference and emotion prediction."""

import pandas as pd
import pytest

from emotion_engine.predict import EmotionPredictor, EmotionRecord


@pytest.fixture
def predictor() -> EmotionPredictor:
    return EmotionPredictor()


def test_emotion_record_validation():
    # Valid
    rec = EmotionRecord(
        age=25,
        gender="Female",
        platform="Instagram",
        daily_usage_time=120.0,
        posts_per_day=3.0,
        likes_received_per_day=45.0,
        comments_received_per_day=10.0,
        messages_sent_per_day=12.0,
    )
    rec.validate()

    # Invalid age
    with pytest.raises(ValueError, match="Age must be between"):
        EmotionRecord(
            age=-5,
            gender="Female",
            platform="Instagram",
            daily_usage_time=120.0,
            posts_per_day=3.0,
            likes_received_per_day=45.0,
            comments_received_per_day=10.0,
            messages_sent_per_day=12.0,
        ).validate()

    # Invalid platform
    with pytest.raises(ValueError, match="not supported"):
        EmotionRecord(
            age=25,
            gender="Female",
            platform="TikTok",
            daily_usage_time=120.0,
            posts_per_day=3.0,
            likes_received_per_day=45.0,
            comments_received_per_day=10.0,
            messages_sent_per_day=12.0,
        ).validate()


def test_predict_single_record(predictor: EmotionPredictor):
    rec = EmotionRecord(
        age=25,
        gender="Female",
        platform="Instagram",
        daily_usage_time=120.0,
        posts_per_day=3.0,
        likes_received_per_day=45.0,
        comments_received_per_day=10.0,
        messages_sent_per_day=12.0,
    )
    result = predictor.predict(rec)
    assert result.predicted_emotion in predictor.classes_
    assert 0.0 <= result.confidence <= 1.0
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-2)


def test_predict_batch(predictor: EmotionPredictor):
    df = pd.DataFrame(
        [
            {
                "Age": 25,
                "Gender": "Female",
                "Platform": "Instagram",
                "Daily_Usage_Time (minutes)": 120.0,
                "Posts_Per_Day": 3.0,
                "Likes_Received_Per_Day": 45.0,
                "Comments_Received_Per_Day": 10.0,
                "Messages_Sent_Per_Day": 12.0,
            },
            {
                "Age": 30,
                "Gender": "Male",
                "Platform": "Twitter",
                "Daily_Usage_Time (minutes)": 90.0,
                "Posts_Per_Day": 5.0,
                "Likes_Received_Per_Day": 20.0,
                "Comments_Received_Per_Day": 25.0,
                "Messages_Sent_Per_Day": 30.0,
            },
        ]
    )
    out = predictor.predict_batch(df)
    assert "Predicted_Emotion" in out.columns
    assert "Confidence" in out.columns
    assert len(out) == 2
