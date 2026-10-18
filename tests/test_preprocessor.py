"""Unit tests for data cleaning and preprocessing utilities."""

import pandas as pd
import pytest

from emotion_engine.config import ALL_FEATURES
from emotion_engine.preprocessor import (
    clean_dataset,
    fit_encoders,
    fit_scalers,
    prepare_inference_frame,
    swap_incorrect_values,
    transform_encoders,
    transform_scalers,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "User_ID": [1, 2, 3],
            "Age": [25, "Female", 30],
            "Gender": ["Female", 28, "Male"],
            "Platform": ["Instagram", "Twitter", "Facebook"],
            "Daily_Usage_Time (minutes)": [120.0, 90.0, 60.0],
            "Posts_Per_Day": [3.0, 5.0, 2.0],
            "Likes_Received_Per_Day": [45.0, 20.0, 15.0],
            "Comments_Received_Per_Day": [10.0, 25.0, 5.0],
            "Messages_Sent_Per_Day": [12.0, 30.0, 20.0],
            "Dominant_Emotion": ["Happiness", "Anger", "Neutral"],
        }
    )


def test_swap_incorrect_values(sample_raw_df: pd.DataFrame):
    swapped = swap_incorrect_values(sample_raw_df)
    assert swapped.loc[1, "Age"] == 28
    assert swapped.loc[1, "Gender"] == "Female"


def test_clean_dataset(sample_raw_df: pd.DataFrame):
    cleaned = clean_dataset(sample_raw_df)
    assert len(cleaned) == 3
    assert cleaned["Age"].dtype == int
    assert cleaned.loc[1, "Age"] == 28
    assert cleaned.loc[1, "Gender"] == "Female"


def test_encoders_and_scalers_pipeline(sample_raw_df: pd.DataFrame):
    cleaned = clean_dataset(sample_raw_df)
    encoders = fit_encoders(cleaned)
    assert "Gender" in encoders
    assert "Platform" in encoders
    assert "Dominant_Emotion" in encoders

    encoded = transform_encoders(cleaned, encoders)
    assert encoded["Gender"].dtype != object
    assert encoded["Platform"].dtype != object

    scalers = fit_scalers(encoded)
    assert "Age" in scalers
    scaled_df = transform_scalers(encoded, scalers)
    assert scaled_df["Age"].dtype == float

    rec = {
        "Age": 25,
        "Gender": "Female",
        "Platform": "Instagram",
        "Daily_Usage_Time (minutes)": 100.0,
        "Posts_Per_Day": 2.0,
        "Likes_Received_Per_Day": 30.0,
        "Comments_Received_Per_Day": 10.0,
        "Messages_Sent_Per_Day": 15.0,
    }
    inf_df = prepare_inference_frame(rec, encoders, scalers)
    assert list(inf_df.columns) == ALL_FEATURES
    assert len(inf_df) == 1
