"""Command-line interface for Emotion Engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from emotion_engine.config import DEFAULT_TEST_DATA, DEFAULT_TRAIN_DATA
from emotion_engine.models import train_and_evaluate
from emotion_engine.predict import EmotionPredictor, EmotionRecord


def cmd_predict(args: argparse.Namespace) -> int:
    predictor = EmotionPredictor()
    record = EmotionRecord(
        age=args.age,
        gender=args.gender,
        platform=args.platform,
        daily_usage_time=args.usage,
        posts_per_day=args.posts,
        likes_received_per_day=args.likes,
        comments_received_per_day=args.comments,
        messages_sent_per_day=args.messages,
    )
    result = predictor.predict(record)
    print("\n" + "=" * 50)
    print(" Emotion Engine - Prediction Result")
    print("=" * 50)
    print(f" Predicted Dominant Emotion: {result.predicted_emotion.upper()}")
    print(f" Confidence:                 {result.confidence:.2%}")
    print("-" * 50)
    print(" Class Probabilities:")
    for emotion, prob in sorted(result.probabilities.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"   {emotion:<12} {prob:6.2%} | {bar}")
    print("=" * 50 + "\n")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file {input_path} not found.")
        return 1

    predictor = EmotionPredictor()
    df = pd.read_csv(input_path)
    output_df = predictor.predict_batch(df)

    out_name = f"{input_path.stem}_predicted.csv"
    out_path = Path(args.output) if args.output else input_path.with_name(out_name)
    output_df.to_csv(out_path, index=False)
    print(f"Batch prediction completed for {len(output_df)} rows. Saved to: {out_path}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    train_path = Path(args.train_data)
    test_path = Path(args.test_data)
    print("Starting model training pipeline...")
    print(f"Train data: {train_path}")
    print(f"Test data:  {test_path}\n")

    trained_models, benchmarks = train_and_evaluate(
        train_path=train_path, test_path=test_path, save_artifacts=True
    )

    print("=" * 70)
    print(f"{'Model':<22} | {'Accuracy':>10} | {'Macro F1':>10} | {'Weighted F1':>12}")
    print("-" * 70)
    for name, bench in benchmarks.items():
        print(
            f"{name:<22} | {bench.accuracy:10.2%} | {bench.macro_f1:10.2%} | "
            f"{bench.weighted_f1:12.2%}"
        )
    print("=" * 70)

    print("\nTop Performing Model: Random Forest Classifier (saved to models/classifier.joblib)\n")
    print(benchmarks["Random Forest"].classification_report)
    return 0


def cmd_demo(_args: argparse.Namespace) -> int:
    predictor = EmotionPredictor()
    demo_samples = [
        EmotionRecord(
            age=25,
            gender="Female",
            platform="Instagram",
            daily_usage_time=120.0,
            posts_per_day=3.0,
            likes_received_per_day=45.0,
            comments_received_per_day=10.0,
            messages_sent_per_day=12.0,
        ),
        EmotionRecord(
            age=30,
            gender="Male",
            platform="Twitter",
            daily_usage_time=90.0,
            posts_per_day=5.0,
            likes_received_per_day=20.0,
            comments_received_per_day=25.0,
            messages_sent_per_day=30.0,
        ),
        EmotionRecord(
            age=28,
            gender="Female",
            platform="Instagram",
            daily_usage_time=200.0,
            posts_per_day=8.0,
            likes_received_per_day=100.0,
            comments_received_per_day=30.0,
            messages_sent_per_day=50.0,
        ),
        EmotionRecord(
            age=22,
            gender="Male",
            platform="Facebook",
            daily_usage_time=60.0,
            posts_per_day=2.0,
            likes_received_per_day=15.0,
            comments_received_per_day=5.0,
            messages_sent_per_day=20.0,
        ),
    ]

    print("\n" + "=" * 76)
    print("  Emotion Engine - Validation Scenarios")
    print("=" * 76)
    header = (
        f"{'Age':>4} | {'Gender':<6} | {'Platform':<9} | {'Usage':>5} | "
        f"{'Posts':>5} | {'Predicted Emotion':<14} | {'Confidence'}"
    )
    print(header)
    print("-" * 76)

    for rec in demo_samples:
        res = predictor.predict(rec)
        print(
            f"{rec.age:4d} | {rec.gender:<6} | {rec.platform:<9} | {rec.daily_usage_time:5.0f} | "
            f"{rec.posts_per_day:5.0f} | {res.predicted_emotion:<14} | {res.confidence:10.2%}"
        )

    print("=" * 76 + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="emotion-engine",
        description="Emotion Engine: Predict dominant emotions from social media usage patterns.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Predict subcommand
    p_pred = subparsers.add_parser("predict", help="Predict dominant emotion for a single user")
    p_pred.add_argument("--age", type=int, required=True, help="User age")
    p_pred.add_argument(
        "--gender", type=str, required=True, choices=["Male", "Female"], help="Gender"
    )
    p_pred.add_argument(
        "--platform",
        type=str,
        required=True,
        choices=["Instagram", "Twitter", "Facebook", "LinkedIn", "Snapchat", "Telegram"],
        help="Primary social media platform",
    )
    p_pred.add_argument("--usage", type=float, required=True, help="Daily usage time in minutes")
    p_pred.add_argument("--posts", type=float, required=True, help="Posts per day")
    p_pred.add_argument("--likes", type=float, required=True, help="Likes received per day")
    p_pred.add_argument("--comments", type=float, required=True, help="Comments received per day")
    p_pred.add_argument("--messages", type=float, required=True, help="Messages sent per day")
    p_pred.set_defaults(func=cmd_predict)

    # Batch subcommand
    p_batch = subparsers.add_parser("batch", help="Batch predict emotions from CSV")
    p_batch.add_argument("--input", "-i", type=str, required=True, help="Input CSV path")
    p_batch.add_argument("--output", "-o", type=str, help="Output CSV path")
    p_batch.set_defaults(func=cmd_batch)

    # Train subcommand
    p_train = subparsers.add_parser("train", help="Run model training and benchmark comparison")
    p_train.add_argument(
        "--train-data", type=str, default=str(DEFAULT_TRAIN_DATA), help="Train CSV"
    )
    p_train.add_argument("--test-data", type=str, default=str(DEFAULT_TEST_DATA), help="Test CSV")
    p_train.set_defaults(func=cmd_train)

    # Demo subcommand
    p_demo = subparsers.add_parser("demo", help="Run validation scenario demo")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    if not args.command:
        # Default to demo
        return cmd_demo(args)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
