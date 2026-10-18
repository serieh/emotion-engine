# Release v1.0.0 Model Artifacts

This directory contains the pre-trained classification models, label encoders, and feature scalers for Emotion Engine v1.0.0.

## Artifact Contents

- **`emotion_engine_models_v1.0.0.tar.gz`** (~400 KB): Compressed archive containing:
  - `classifier.joblib`: Trained Random Forest multiclass classifier (97% test accuracy).
  - `encoders/`: Label encoders for `Dominant_Emotion`, `Gender`, and `Platform`.
  - `scalers/`: StandardScalers for numerical features (`Age`, `Comments`, `Likes`, `Messages`, `Posts`, `Daily_Usage`).

## Checksum Verification

```bash
sha256sum -c SHA256SUMS
```

## Publishing to GitHub Releases

When publishing this version to GitHub, attach the release archive using the GitHub CLI:

```bash
gh release create v1.0.0 releases/v1.0.0/emotion_engine_models_v1.0.0.tar.gz \
    --title "v1.0.0 - Production Release" \
    --notes "Production release of Emotion Engine with trained Random Forest model artifacts."
```

## Local Extraction

To extract the model artifacts into the working tree for inference:

```bash
tar -xzvf releases/v1.0.0/emotion_engine_models_v1.0.0.tar.gz -C models/
```
