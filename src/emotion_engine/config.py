"""Configuration constants, feature definitions, and default paths."""

from pathlib import Path

# Paths
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
ENCODERS_DIR = MODELS_DIR / "encoders"
SCALERS_DIR = MODELS_DIR / "scalers"
ASSETS_DIR = REPO_ROOT / "assets"

DEFAULT_TRAIN_DATA = DATA_DIR / "train.csv"
DEFAULT_TEST_DATA = DATA_DIR / "test.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "classifier.joblib"

# Feature definitions
CATEGORICAL_FEATURES = ["Gender", "Platform"]
NUMERICAL_FEATURES = [
    "Age",
    "Daily_Usage_Time (minutes)",
    "Posts_Per_Day",
    "Likes_Received_Per_Day",
    "Comments_Received_Per_Day",
    "Messages_Sent_Per_Day",
]
TARGET_COLUMN = "Dominant_Emotion"

ALL_FEATURES = [
    "Age",
    "Gender",
    "Platform",
    "Comments_Received_Per_Day",
    "Likes_Received_Per_Day",
    "Messages_Sent_Per_Day",
    "Posts_Per_Day",
    "Daily_Usage_Time (minutes)",
]

SUPPORTED_PLATFORMS = [
    "Instagram",
    "Twitter",
    "Facebook",
    "LinkedIn",
    "Snapchat",
    "Telegram",
]
SUPPORTED_GENDERS = ["Male", "Female"]

# Scaler filename mapping
SCALER_NAMES = {
    "Age": "Age_scaler.joblib",
    "Daily_Usage_Time (minutes)": "Usage_scaler.joblib",
    "Posts_Per_Day": "Posts_scaler.joblib",
    "Likes_Received_Per_Day": "Likes_scaler.joblib",
    "Comments_Received_Per_Day": "Comments_scaler.joblib",
    "Messages_Sent_Per_Day": "Messages_scaler.joblib",
}

ENCODER_NAMES = {
    "Gender": "gender_encoder.joblib",
    "Platform": "platform_encoder.joblib",
    "Dominant_Emotion": "emotion_encoder.joblib",
}
