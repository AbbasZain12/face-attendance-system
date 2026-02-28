# recompute_embeddings.py
"""
Recompute embeddings for all images under data/train and data/faces and overwrite embeddings/embeddings.pkl.
Usage:
 python recompute_embeddings.py --feature_dir model/feature_extractor --out embeddings/embeddings.pkl
"""

import os
import argparse
import pickle
from extract_embeddings import process_folder, load_image, compute_embedding
import tensorflow as tf

def main(args):
    extractor = tf.keras.models.load_model(args.feature_dir)
    img_size = args.img_size
    out = args.out
    all_embeddings = {}

    # process data/train
    roots = [args.train_dir, args.faces_dir]
    for root in roots:
        if not os.path.exists(root):
            continue
        for classname in os.listdir(root):
            classfolder = os.path.join(root, classname)
            if not os.path.isdir(classfolder):
                continue
            print(f"Processing {classfolder}")
            embs = process_folder(extractor, classfolder, img_size)
            if classname in all_embeddings:
                all_embeddings[classname].extend(embs)
            else:
                all_embeddings[classname] = embs

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(all_embeddings, f)
    print(f"Recomputed embeddings saved to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, default='model/feature_extractor')
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--faces_dir', type=str, default='data/faces')
    parser.add_argument('--out', type=str, default='embeddings/embeddings.pkl')
    parser.add_argument('--img_size', type=int, default=160)
    args = parser.parse_args()
    main(args)
