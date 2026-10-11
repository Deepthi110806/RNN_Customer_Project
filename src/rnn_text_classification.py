"""
RNN-based text classification of short customer messages.

Pipeline:
  1. Load the customer-message dataset (text, category)
  2. Convert text -> integer sequences (TextVectorization)
  3. Embedding -> Bidirectional LSTM -> Dense softmax classifier
  4. Train / validate, evaluate on the held-out test set
  5. Predict categories for brand-new (unseen) messages

Usage (run from the project root):
    python src/rnn_text_classification.py            # train + evaluate + demo predictions
    python src/rnn_text_classification.py --predict "my card has not arrived"
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, callbacks

SEED = 42
MAX_TOKENS = 8000      # vocabulary size
SEQ_LEN = 40           # max words per message (messages are short)
EMBED_DIM = 128
LSTM_UNITS = 128
BATCH_SIZE = 64
EPOCHS = 25

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "dataset")
RESULTS_DIR = os.path.join(ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "rnn_classifier.keras")
LABELS_PATH = os.path.join(RESULTS_DIR, "labels.json")

UNSEEN_MESSAGES = [
    "My new card still has not shown up in the mail",
    "How do I change my PIN number?",
    "I was charged twice for the same purchase",
    "Why was my transfer declined?",
    "Can I top up my account with a debit card?",
    "I lost my phone and my card was in the case",
    "What is the exchange rate if I pay abroad?",
    "Someone I don't know withdrew cash from my account",
]


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "customer_train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "customer_test.csv"))
    return train, test


def build_model(vectorizer, num_classes):
    """Embedding -> BiLSTM -> Dense. The vectorizer is part of the model,
    so it accepts raw strings at prediction time."""
    inputs = layers.Input(shape=(1,), dtype=tf.string)
    x = vectorizer(inputs)
    x = layers.Embedding(MAX_TOKENS, EMBED_DIM, mask_zero=True)(x)
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.history["accuracy"], label="train")
    ax[0].plot(history.history["val_accuracy"], label="validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(history.history["loss"], label="train")
    ax[1].plot(history.history["val_loss"], label="validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150)
    plt.close()


def plot_per_class_f1(report, path):
    items = {k: v["f1-score"] for k, v in report.items()
             if isinstance(v, dict) and k not in ("macro avg", "weighted avg")}
    items = dict(sorted(items.items(), key=lambda kv: kv[1]))
    worst = list(items.items())[:20]
    plt.figure(figsize=(8, 6))
    plt.barh([k for k, _ in worst], [v for _, v in worst], color="#d1495b")
    plt.xlabel("F1-score"); plt.title("20 weakest categories (test set)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def predict(model, class_names, messages, top_k=3):
    probs = model.predict(tf.constant(list(messages)), verbose=0)
    rows = []
    for msg, p in zip(messages, probs):
        idx = np.argsort(p)[::-1][:top_k]
        rows.append((msg, [(class_names[i], float(p[i])) for i in idx]))
    return rows


def print_predictions(rows):
    for msg, top in rows:
        print(f'\nMessage : "{msg}"')
        for rank, (label, conf) in enumerate(top, 1):
            print(f"   {rank}. {label:<40s} {conf*100:5.1f}%")


def train():
    set_seed()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_df, test_df = load_data()
    print(f"Train: {len(train_df)} | Test: {len(test_df)} | "
          f"Classes: {train_df['category'].nunique()}")

    le = LabelEncoder().fit(train_df["category"])
    y_train = le.transform(train_df["category"])
    y_test = le.transform(test_df["category"])
    class_names = list(le.classes_)

    # validation split taken from the training data (test set stays unseen)
    idx = np.random.permutation(len(train_df))
    n_val = int(0.1 * len(idx))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    X = train_df["text"].astype(str).to_numpy()
    X_tr, y_tr = tf.constant(X[tr_idx]), y_train[tr_idx]
    X_val, y_val = tf.constant(X[val_idx]), y_train[val_idx]

    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS, output_mode="int",
        output_sequence_length=SEQ_LEN, standardize="lower_and_strip_punctuation")
    vectorizer.adapt(X_tr)

    model = build_model(vectorizer, len(class_names))
    model.summary()

    cbs = [callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                   restore_best_weights=True)]
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cbs, verbose=2)
    plot_history(history)

    # ---- evaluation on unseen test set ----
    X_test = tf.constant(test_df["text"].astype(str).to_numpy())
    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    print(f"\nTest accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    report = classification_report(y_test, pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_test, pred, target_names=class_names, zero_division=0))
    plot_per_class_f1(report, os.path.join(RESULTS_DIR, "per_class_f1.png"))
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": acc, "macro_f1": macro_f1,
                   "epochs_run": len(history.history["loss"])}, f, indent=2)

    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(class_names, f)

    # ---- predictions on previously unseen messages ----
    rows = predict(model, class_names, UNSEEN_MESSAGES)
    print_predictions(rows)
    with open(os.path.join(RESULTS_DIR, "unseen_predictions.txt"), "w") as f:
        for msg, top in rows:
            f.write(f'"{msg}" -> ' + ", ".join(f"{l} ({c*100:.1f}%)" for l, c in top) + "\n")


def predict_only(messages):
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_names = json.load(f)
    print_predictions(predict(model, class_names, messages))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", nargs="+", help="classify one or more messages with the saved model")
    args = ap.parse_args()
    if args.predict:
        predict_only(args.predict)
    else:
        train()
