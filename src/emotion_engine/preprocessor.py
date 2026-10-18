"""Data preprocessing, anomaly cleaning, encoding, and scaling utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from emotion_engine.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)


def swap_incorrect_values(df: pd.DataFrame) -> pd.DataFrame:
    """Correct inverted 'Age' and 'Gender' values where Gender strings appear in Age."""
    df_copy = df.copy()
    mask = df_copy["Age"].isin(["Male", "Female"])
    if mask.any():
        df_copy.loc[mask, ["Age", "Gender"]] = df_copy.loc[mask, ["Gender", "Age"]].values
    return df_copy


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows, repair swapped columns, and ensure numeric types."""
    df_clean = df.dropna().copy()
    df_clean = swap_incorrect_values(df_clean)

    for col in NUMERICAL_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = (
                pd.to_numeric(df_clean[col], errors="coerce").fillna(0).round().astype(int)
            )

    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()

    if TARGET_COLUMN in df_clean.columns:
        df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].astype(str).str.strip()

    return df_clean


def fit_encoders(df: pd.DataFrame) -> dict[str, LabelEncoder]:
    """Fit LabelEncoders for all categorical features and the target emotion."""
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le

    if TARGET_COLUMN in df.columns:
        le_target = LabelEncoder()
        le_target.fit(df[TARGET_COLUMN])
        encoders[TARGET_COLUMN] = le_target

    return encoders


def transform_encoders(
    df: pd.DataFrame, encoders: dict[str, LabelEncoder]
) -> pd.DataFrame:
    """Transform categorical columns using pre-fitted encoders."""
    df_enc = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df_enc.columns and col in encoders:
            df_enc[col] = encoders[col].transform(df_enc[col])

    if TARGET_COLUMN in df_enc.columns and TARGET_COLUMN in encoders:
        df_enc[TARGET_COLUMN] = encoders[TARGET_COLUMN].transform(df_enc[TARGET_COLUMN])

    return df_enc


def fit_scalers(df: pd.DataFrame) -> dict[str, StandardScaler]:
    """Fit StandardScalers for each numerical feature."""
    scalers: dict[str, StandardScaler] = {}
    for col in NUMERICAL_FEATURES:
        scaler = StandardScaler()
        scaler.fit(df[[col]])
        scalers[col] = scaler
    return scalers


def transform_scalers(
    df: pd.DataFrame, scalers: dict[str, StandardScaler]
) -> pd.DataFrame:
    """Transform numerical columns using pre-fitted scalers."""
    df_scaled = df.copy()
    for col in NUMERICAL_FEATURES:
        if col in df_scaled.columns and col in scalers:
            df_scaled[col] = scalers[col].transform(df_scaled[[col]])
    return df_scaled


def prepare_inference_frame(
    record: dict[str, Any],
    encoders: dict[str, LabelEncoder],
    scalers: dict[str, StandardScaler],
) -> pd.DataFrame:
    """Transform a single raw user feature record into model-ready scaled DataFrame."""
    # Encode categoricals
    gender_val = encoders["Gender"].transform([record["Gender"]])[0]
    platform_val = encoders["Platform"].transform([record["Platform"]])[0]

    row = {
        "Age": float(record["Age"]),
        "Gender": gender_val,
        "Platform": platform_val,
        "Comments_Received_Per_Day": float(record["Comments_Received_Per_Day"]),
        "Likes_Received_Per_Day": float(record["Likes_Received_Per_Day"]),
        "Messages_Sent_Per_Day": float(record["Messages_Sent_Per_Day"]),
        "Posts_Per_Day": float(record["Posts_Per_Day"]),
        "Daily_Usage_Time (minutes)": float(record["Daily_Usage_Time (minutes)"]),
    }

    df_inf = pd.DataFrame([row])

    # Scale numeric features
    for col in NUMERICAL_FEATURES:
        if col in scalers:
            df_inf[col] = scalers[col].transform(df_inf[[col]])

    # Return exactly matching feature column ordering
    return df_inf[ALL_FEATURES]
