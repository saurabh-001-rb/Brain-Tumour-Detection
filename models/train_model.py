import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def build_model(num_classes: int) -> keras.Model:
    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")
    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="brain_tumor_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_dataset(dataset_dir: Path, subset_name: str) -> tf.data.Dataset:
    dataset = keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=(subset_name == "Training"),
        seed=42,
    )
    dataset = dataset.cache().prefetch(AUTOTUNE)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a brain tumor classifier on MRI images.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Path to the dataset root containing Training/ and Testing/ subdirectories.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("my_cnn_model.keras"),
        help="Path where the trained Keras model will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Number of training epochs.",
    )

    args = parser.parse_args()
    train_dir = args.dataset_dir / "Training"
    val_dir = args.dataset_dir / "Testing"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Dataset directories not found. Expected Training/ and Testing/ under {args.dataset_dir}."
        )

    raw_train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=42,
    )
    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(AUTOTUNE)

    raw_val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False,
    )
    val_ds = raw_val_ds.cache().prefetch(AUTOTUNE)

    print(f"Class names: {class_names}")
    print(f"Training batches: {len(train_ds)}, Validation batches: {len(val_ds)}")

    model = build_model(num_classes=len(class_names))
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.model_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("Training complete. Evaluating on validation dataset...")
    loss, accuracy = model.evaluate(val_ds)
    print(f"Validation loss: {loss:.4f}, Validation accuracy: {accuracy:.4f}")

    if not args.model_path.exists():
        model.save(args.model_path)
        print(f"Saved model to {args.model_path}")
    else:
        print(f"Best model saved to {args.model_path}")

    metrics_path = args.model_path.with_suffix(".metrics.txt")
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        metrics_file.write(f"validation_loss: {loss:.6f}\n")
        metrics_file.write(f"validation_accuracy: {accuracy:.6f}\n")
        metrics_file.write(f"class_names: {class_names}\n")

    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
