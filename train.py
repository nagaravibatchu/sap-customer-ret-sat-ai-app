# train.py
# Schema-safe training script for SAP tabular data
# - Reads Excel (.xlsx/.xls)
# - Parses POSTING_MONTH in YYYY-MM-DD (e.g., 2026-01-16)
# - Derives POSTING_YEAR and POSTING_MON (fed as float32 for deployment safety)
# - Trains a multi-output Keras model:
#     sat: regression (Customer Satisfaction)
#     ret: classification (Customer Retained probability)
# - Saves model as customer_model.keras
#
# Run:
#   python train.py --data your_file.xlsx --epochs 100

import argparse
import sys
from tkinter import Tk, filedialog

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from schema import SCHEMA, TF_DTYPES, assert_schema_consistency

MODEL_OUTPUT = "customer_model.keras"
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 100
RANDOM_SEED = 42


def open_file_dialog() -> str:
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Select SAP Excel Training File",
        filetypes=[("Excel files", "*.xlsx *.xls")],
    )
    if not path:
        print("❌ No file selected. Exiting.")
        sys.exit(1)
    return path


def yesno_to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.map({"yes": 1.0, "no": 0.0}).fillna(0.0).astype("float32")


def load_and_prepare_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Excel RAW columns only (derived columns created later)
    required_excel_cols = (
        list(SCHEMA.CAT_COLS)
        + list(SCHEMA.YESNO_COLS)
        + list(SCHEMA.NUM_COLS)
        + [SCHEMA.DATE_COL, SCHEMA.TARGET_REG, SCHEMA.TARGET_CLS]
    )
    missing = [c for c in required_excel_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Excel: {missing}")

    # Categorical inputs
    for c in SCHEMA.CAT_COLS:
        df[c] = df[c].astype(str).fillna("UNK")

    # Yes/No inputs -> float32 0/1
    for c in SCHEMA.YESNO_COLS:
        df[c] = yesno_to_float(df[c])

    # Targets
    df[SCHEMA.TARGET_CLS] = yesno_to_float(df[SCHEMA.TARGET_CLS])  # 0/1 float32
    df[SCHEMA.TARGET_REG] = pd.to_numeric(df[SCHEMA.TARGET_REG], errors="coerce").astype("float32")

    # Numeric inputs
    for c in SCHEMA.NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float32")

    # Parse POSTING_MONTH in YYYY-MM-DD
    pm_raw = df[SCHEMA.DATE_COL].astype(str).str.strip()
    parsed = pd.to_datetime(pm_raw, format="%Y-%m-%d", errors="coerce")

    invalid_mask = parsed.isna()
    if invalid_mask.any():
        print("⚠️ Invalid POSTING_MONTH examples (expected YYYY-MM-DD):")
        print(pm_raw[invalid_mask].head(10).tolist())

    df["POSTING_YEAR"] = parsed.dt.year
    df["POSTING_MON"] = parsed.dt.month

    # Drop rows with invalid dates
    df = df.dropna(subset=["POSTING_YEAR", "POSTING_MON"])

    # Feed derived date fields as float32 to avoid cast layers
    df["POSTING_YEAR"] = df["POSTING_YEAR"].astype("float32")
    df["POSTING_MON"] = df["POSTING_MON"].astype("float32")

    # Drop rows missing targets
    df = df.dropna(subset=[SCHEMA.TARGET_REG, SCHEMA.TARGET_CLS])

    if len(df) == 0:
        raise ValueError(
            "After parsing YYYY-MM-DD POSTING_MONTH, 0 rows remain. "
            "Please verify your Excel date format."
        )

    print(f"✅ Rows after cleaning: {len(df):,}")
    return df


def make_tf_dataset(df: pd.DataFrame, batch_size: int) -> tf.data.Dataset:
    x = {}
    for name in SCHEMA.model_input_names():
        dtype = TF_DTYPES[name]
        if dtype == "string":
            x[name] = df[name].astype(str).values
        elif dtype == "float32":
            x[name] = df[name].astype("float32").values
        else:
            raise ValueError(f"Unsupported dtype: {name} -> {dtype}")

    y = {
        SCHEMA.OUT_SAT: df[SCHEMA.TARGET_REG].astype("float32").values,
        SCHEMA.OUT_RET: df[SCHEMA.TARGET_CLS].astype("float32").values,
    }

    ds = tf.data.Dataset.from_tensor_slices((x, y))

    buffer_size = min(len(df), 200_000)
    if buffer_size >= 2:
        ds = ds.shuffle(buffer_size, seed=RANDOM_SEED)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(train_df: pd.DataFrame) -> tf.keras.Model:
    inputs = {}
    encoded_parts = []

    # Categorical: StringLookup + Embedding
    for c in SCHEMA.CAT_COLS:
        inputs[c] = layers.Input(shape=(1,), name=c, dtype=tf.string)
        vocab = train_df[c].astype(str).fillna("UNK").unique().tolist()
        lookup = layers.StringLookup(
            vocabulary=vocab, mask_token=None, oov_token="UNK", name=f"{c}_lookup"
        )
        ids = lookup(inputs[c])
        emb_dim = int(min(32, max(4, round(len(vocab) ** 0.25 * 4))))
        emb = layers.Embedding(
            input_dim=lookup.vocabulary_size(),
            output_dim=emb_dim,
            name=f"{c}_emb",
        )(ids)
        encoded_parts.append(layers.Reshape((emb_dim,))(emb))

    # Flags: float32 inputs directly (no cast layer needed)
    for c in SCHEMA.YESNO_COLS:
        inputs[c] = layers.Input(shape=(1,), name=c, dtype=tf.float32)
        encoded_parts.append(inputs[c])

    # Numeric: Normalization
    for c in SCHEMA.NUM_COLS:
        inputs[c] = layers.Input(shape=(1,), name=c, dtype=tf.float32)
        norm = layers.Normalization(name=f"{c}_norm")
        norm.adapt(train_df[c].astype("float32").values.reshape(-1, 1))
        encoded_parts.append(norm(inputs[c]))

    # Derived date fields: float32 inputs directly
    for c in SCHEMA.DERIVED_COLS:
        inputs[c] = layers.Input(shape=(1,), name=c, dtype=tf.float32)
        encoded_parts.append(inputs[c])

    x = layers.Concatenate(name="features_concat")(encoded_parts)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)

    sat_out = layers.Dense(1, name=SCHEMA.OUT_SAT)(x)
    ret_out = layers.Dense(1, activation="sigmoid", name=SCHEMA.OUT_RET)(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={SCHEMA.OUT_SAT: sat_out, SCHEMA.OUT_RET: ret_out},
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={SCHEMA.OUT_SAT: "mse", SCHEMA.OUT_RET: "binary_crossentropy"},
        metrics={
            SCHEMA.OUT_SAT: [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            SCHEMA.OUT_RET: [
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.BinaryAccuracy(name="acc"),
            ],
        },
    )
    return model


def main():
    assert_schema_consistency()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to Excel training file (.xlsx/.xls)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    path = args.data if args.data else open_file_dialog()
    print(f"📂 Loading: {path}")

    df = load_and_prepare_excel(path)

    # Shuffle + split
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[: int(0.8 * n)]
    val_df = df.iloc[int(0.8 * n): int(0.9 * n)]
    test_df = df.iloc[int(0.9 * n):]

    train_ds = make_tf_dataset(train_df, args.batch_size)
    val_ds = make_tf_dataset(val_df, args.batch_size)
    test_ds = make_tf_dataset(test_df, args.batch_size)

    model = build_model(train_df)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("\n📊 Test evaluation:")
    model.evaluate(test_ds)

    model.save(MODEL_OUTPUT)
    print(f"\n✅ Saved model: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()