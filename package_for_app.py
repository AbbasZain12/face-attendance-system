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
TEMP_MODEL_FOLDER = "temp_keras_model" # A temporary folder for conversion
# --------------------

def build_and_convert_model_to_tflite(output_tflite_path):
    """Builds the FaceNet model, saves it temporarily, and converts to .tflite."""
    print("--- Step 1: Building FaceNet model in memory ---")
    try:
        # This function returns a custom wrapper object
        wrapper_model = DeepFace.build_model("Facenet")
        print("SUCCESS: Wrapper model built successfully in memory.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to build the Keras model: {e}")
        return False

    print("\n--- Step 2: Saving Keras model to a temporary directory ---")
    try:
        # Extract the actual Keras model from the wrapper
        actual_keras_model = wrapper_model.model

        # Clean up any previous temporary folder
        if os.path.exists(TEMP_MODEL_FOLDER):
            shutil.rmtree(TEMP_MODEL_FOLDER)
        
        # --- FIX ---
        # Use model.export() to save in the SavedModel format for TFLite conversion.
        actual_keras_model.export(TEMP_MODEL_FOLDER)
        print(f"SUCCESS: Model temporarily saved to '{TEMP_MODEL_FOLDER}'")

    except Exception as e:
        print(f"FATAL ERROR: Failed to save the temporary Keras model: {e}")
        return False

    print("\n--- Step 3: Converting saved model to TFLite ---")
    try:
        # Initialize the TFLite converter from the saved model directory
        converter = tf.lite.TFLiteConverter.from_saved_model(TEMP_MODEL_FOLDER)
        
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
        return False
    finally:
        # Clean up the temporary directory
        if os.path.exists(TEMP_MODEL_FOLDER):
            print(f"Cleaning up temporary folder '{TEMP_MODEL_FOLDER}'...")
            shutil.rmtree(TEMP_MODEL_FOLDER)

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
    print("\n--- Step 4: Copying user data ---")
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

