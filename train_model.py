import os
import argparse
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def main(args):
    # ----------------------------
    # 1. Load embeddings and labels
    # ----------------------------
    if not os.path.exists(args.embeddings):
        print(" ❌  Embeddings file not found:", args.embeddings)
        return
    
    data = np.load(args.embeddings)
    embeddings = data["embeddings"]
    labels = data["labels"]
    print(f"Loaded {len(embeddings)} embeddings.")

    # Load label encoder (to map back names)
    label_encoder_path = "embeddings/label_encoder.pkl"
    if os.path.exists(label_encoder_path):
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        print("Label encoder loaded.")
    else:
        print(" ⚠️  Label encoder not found, creating a new one.")
        label_encoder = LabelEncoder()
        label_encoder.fit(labels) # Fit on all labels before splitting

    num_classes = len(label_encoder.classes_)
    print("Classes:", label_encoder.classes_)

    # ----------------------------
    # 2. Split data
    # ----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # ❗ FIX: Convert string labels to integer labels
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    # ----------------------------
    # 3. Build a simple classifier
    # ----------------------------
    model = Sequential([
        Dense(128, activation="relu", input_shape=(embeddings.shape[1],)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ----------------------------
    # 4. Train the classifier
    # ----------------------------
    # ❗ FIX: Use the new encoded integer labels for training
    model.fit(
        X_train, y_train_encoded,
        validation_data=(X_val, y_val_encoded),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    # ----------------------------
    # 5. Save model and report
    # ----------------------------
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    model.save(args.model)
    print("Model saved to:", args.model)
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to embeddings file (.npz)")
    parser.add_argument("--model", required=True, help="Path to save model (.keras)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)