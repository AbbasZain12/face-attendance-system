import os
import shutil
import tensorflow as tf
from deepface import DeepFace
import sys

# --- Configuration ---
MODEL_NAME_TFLITE = "facenet.tflite"
EMBEDDINGS_FOLDER = "embeddings"
EMBEDDINGS_FILE = "embeddings.npz"
LABELS_FILE = "label_encoder.pkl"
OUTPUT_FOLDER = "deployment_package"
# --------------------

def build_and_convert_model_to_tflite(output_tflite_path):
    """Builds the FaceNet model using DeepFace and converts it to .tflite."""
    print("--- Step 1: Building FaceNet model in memory ---")
    try:
        # This function correctly loads the model architecture and weights
        model = DeepFace.build_model("Facenet")
        print("SUCCESS: Model built successfully in memory.")

    except Exception as e:
        print(f"FATAL ERROR: Failed to build the Keras model: {e}")
        print("Please ensure your internet connection is active and try again.")
        return False

    print("\n--- Step 2: Converting Keras model to TFLite ---")
    try:
        # Initialize the TFLite converter from the in-memory Keras model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply optimizations for mobile deployment
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the TFLite model to a file
        with open(output_tflite_path, "wb") as f:
            f.write(tflite_model)
            
        print(f"SUCCESS: Saved TFLite model to {output_tflite_path}")
        return True
        
    except Exception as e:
        print(f"FATAL ERROR: Failed during TFLite conversion: {e}")
        print("This may be due to a TensorFlow version mismatch.")
        return False

def package_files():
    """Builds, converts, and packages all necessary files into a single folder."""
    # Create the output directory, cleaning it if it already exists
    if os.path.exists(OUTPUT_FOLDER):
        print(f"Removing existing '{OUTPUT_FOLDER}' directory...")
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created new '{OUTPUT_FOLDER}' directory.")

    # Step 1: Build and Convert the model
    tflite_output_path = os.path.join(OUTPUT_FOLDER, MODEL_NAME_TFLITE)
    if not build_and_convert_model_to_tflite(tflite_output_path):
        sys.exit(1) # Exit the script if conversion fails

    # Step 2: Copy the user data files
    print("\n--- Step 3: Copying user data ---")
    try:
        embedding_src = os.path.join(EMBEDDINGS_FOLDER, EMBEDDINGS_FILE)
        labels_src = os.path.join(EMBEDDINGS_FOLDER, LABELS_FILE)

        if not os.path.exists(embedding_src) or not os.path.exists(labels_src):
            print(f"FATAL ERROR: User data not found in '{EMBEDDINGS_FOLDER}'.")
            print("Please run 'python app.py', then use the web interface to 'Retrain Model' before running this script.")
            return

        shutil.copy(embedding_src, OUTPUT_FOLDER)
        shutil.copy(labels_src, OUTPUT_FOLDER)
        print(f"SUCCESS: Copied {EMBEDDINGS_FILE} and {LABELS_FILE}")
        
    except Exception as e:
        print(f"FATAL ERROR: Could not copy user data files: {e}")
        return
        
    print("\n--------------------------")
    print("--- PACKAGE COMPLETE ---")
    print("--------------------------")
    print(f"All necessary files are now in the '{OUTPUT_FOLDER}' folder.")
    print("You can now zip this folder and send it to your app developer.")

if __name__ == "__main__":
    package_files()

