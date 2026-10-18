"""Unit tests for model training and CLI execution."""

from emotion_engine.cli import main
from emotion_engine.models import train_and_evaluate


def test_train_and_evaluate_pipeline():
    trained_models, benchmarks = train_and_evaluate(save_artifacts=False)
    assert "Random Forest" in benchmarks
    assert "Decision Tree" in benchmarks
    assert "KNN (k=2)" in benchmarks
    assert "SVM (RBF)" in benchmarks

    rf_bench = benchmarks["Random Forest"]
    assert rf_bench.accuracy >= 0.90
    assert rf_bench.macro_f1 >= 0.90


def test_cli_demo(capsys):
    ret = main(["demo"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Emotion Engine - Validation Scenarios" in captured.out


def test_cli_predict(capsys):
    ret = main(
        [
            "predict",
            "--age",
            "25",
            "--gender",
            "Female",
            "--platform",
            "Instagram",
            "--usage",
            "120",
            "--posts",
            "3",
            "--likes",
            "45",
            "--comments",
            "10",
            "--messages",
            "12",
        ]
    )
    assert ret == 0
    captured = capsys.readouterr()
    assert "Predicted Dominant Emotion" in captured.out
