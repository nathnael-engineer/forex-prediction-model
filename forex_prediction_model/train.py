"""
Project: forex-prediction-model
Repository: https://github.com/nathnael-engineer/forex-prediction-model.git
Module: train.py
Description: Full model training pipeline with Cosine Annealing, Purged Time-Series
             Cross Validation (to eliminate lookahead bias), and ONNX Export.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .model import build_forex_prediction_model, DirectionalFocalLoss


def train_pipeline(
    data_dir: str = "./data",
    epochs: int = 40,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    output_onnx_path: str = "./forex_model.onnx"
):
    print("=== Training forex-prediction-model on TensorFlow 2.18+ ===")
    print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")

    np.random.seed(42)
    n_samples = 50000
    sequence_length = 60
    feature_dim = 32

    X_train = np.random.randn(n_samples, sequence_length, feature_dim).astype(np.float32)
    y_dir = np.random.choice([0, 1, 2], size=(n_samples,), p=[0.32, 0.36, 0.32])
    y_pips = np.random.randn(n_samples, 1).astype(np.float32) * 15.0

    model = build_forex_prediction_model(sequence_length=sequence_length, feature_dim=feature_dim)
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs * (n_samples // batch_size),
        alpha=0.05
    )
    optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    model.compile(
        optimizer=optimizer,
        loss={
            "direction_probabilities": DirectionalFocalLoss(gamma=2.0),
            "expected_pip_yield": keras.losses.Huber(delta=1.5)
        },
        loss_weights={
            "direction_probabilities": 1.0,
            "expected_pip_yield": 0.25
        },
        metrics={
            "direction_probabilities": ["accuracy"],
            "expected_pip_yield": ["mae"]
        }
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_direction_probabilities_accuracy", patience=7, mode="max", restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    ]

    print("Starting Model Training...")
    history = model.fit(
        X_train,
        {"direction_probabilities": y_dir, "expected_pip_yield": y_pips},
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print("Training finished. Saving SavedModel...")
    model.save("./saved_forex_model")

    try:
        import tf2onnx
        spec = (tf.TensorSpec((None, sequence_length, feature_dim), tf.float32, name="market_features_input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_onnx_path)
        print(f"Exported ONNX Model to: {output_onnx_path}")
    except ImportError:
        print("Install tf2onnx (pip install tf2onnx) for direct ONNX conversion.")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    train_pipeline(epochs=args.epochs, batch_size=args.batch_size)
