import os
import argparse
import pickle
import numpy as np
from deepface import DeepFace  # Using DeepFace
from sklearn.preprocessing import LabelEncoder
import cv2
from PIL import Image

# Helper function to load and process an image
def load_and_prep_image(path, img_size=160):
    """Loads an image, converts to BGR, and resizes."""
    try:
        # Use PIL to open to handle various image formats and convert to RGB
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        
        # Convert RGB (PIL) to BGR (OpenCV)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Resize
        resized_img = cv2.resize(img_bgr, (img_size, img_size))
        return resized_img
        
    except Exception as e:
        print(f" Warning: Failed to load image {path}: {e}")
        return None

def main(args):
    # ----------------------------
    # 1. Initialize DeepFace model
    # ----------------------------
    print("Building FaceNet model via DeepFace...")
    try:
        DeepFace.build_model("Facenet")
        print("Model built successfully.")
    except Exception as e:
        print(f" ERROR: Could not build model: {e}")
        return
        
    dataset_dir = args.dataset
    output_path = args.out
    img_size = 160

    if not os.path.exists(dataset_dir):
        print(" Dataset folder not found:", dataset_dir)
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ----------------------------
    # 2. Manually find images and labels
    # ----------------------------
    image_paths = []
    labels = []
    
    print("Finding images...")
    
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(img_path)
                labels.append(class_name)
            
    if len(labels) == 0:
        print(" ERROR: No images found. Check your dataset folder.")
        return
        
    print(f"\nFound {len(labels)} images belonging to {len(set(labels))} classes.\n")

    # ----------------------------
    # 3. Extract embeddings
    # ----------------------------
    embeddings = []
    final_labels = []
    
    for i, (path, label) in enumerate(zip(image_paths, labels)):
        if (i+1) % 10 == 0 or i == 0:
            print(f"Processing image {i + 1}/{len(labels)} ({path}) ...")
            
        # img is BGR, (160, 160, 3)
        img = load_and_prep_image(path, img_size)
        
        if img is None:
            continue
            
        try:
            # DeepFace expects BGR image
            embedding_obj = DeepFace.represent(
                img_path=img,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend="skip"
            )
            
            emb = np.array(embedding_obj[0]["embedding"])
            
            embeddings.append(emb)
            final_labels.append(label)
            
        except Exception as e:
            print(f" ERROR: Skipping {path}. Failed to compute embedding: {e}")

    if len(embeddings) == 0:
        print(" ERROR: Failed to generate any embeddings.")
        return

    embeddings = np.asarray(embeddings)
    labels = np.asarray(final_labels)

    # ----------------------------
    # 4. Encode labels (usernames)
    # ----------------------------
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    
    print("\nLabel mapping:")
    for idx, name in enumerate(label_encoder.classes_):
        print(f"{idx}: {name}")

    # ----------------------------
    # 5. Save embeddings + label encoder
    # ----------------------------
    print(f"\nSaving {len(embeddings)} embeddings to {output_path}...")
    np.savez(output_path, embeddings=embeddings, labels=labels)
    
    with open("embeddings/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("\n Embeddings saved to:", output_path)
    print(" Label encoder saved to: embeddings/label_encoder.pkl")
    print(" Extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset folder (e.g. data/train)")
    parser.add_argument("--out", required=True, help="Output embeddings file (e.g. embeddings/embeddings.npz)")
    args = parser.parse_args()
    main(args)