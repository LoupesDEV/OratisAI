import os
import torch
import numpy as np
import logging
from scripts.model import ASRModel

def run(config):
    model_path = config['paths']['models']
    feature_dir = config['paths']['features']
    output_dir = config['paths']['decoded_texts']
    vocabulary = [
        "_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " "
    ]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = ASRModel(input_dim=config['training']['input_dim'], hidden_dim=config['training']['hidden_dim'], output_dim=len(vocabulary))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for file_name in sorted(os.listdir(feature_dir)):
        if file_name.endswith(".npy"):
            feature_path = os.path.join(feature_dir, file_name)
            features = np.load(feature_path)
            
            if features.shape[1] != config['training']['input_dim']:
                logging.error(f"Feature shape mismatch in decoding for {file_name}: Expected {config['training']['input_dim']}, got {features.shape[1]}")
                continue

            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(features)
                predicted_indices = torch.argmax(logits, dim=-1)
                predicted_text = "".join([vocabulary[i] for i in predicted_indices[0] if i != 0])

            segment_output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.txt")
            with open(segment_output_path, "w") as f:
                f.write(predicted_text.strip())

            logging.info(f"Decoded file: {file_name}, saved to {segment_output_path}")