import os
import logging
import librosa
import numpy as np

def run(config):
    input_dir = config['paths']['processed_audio']
    output_dir = config['paths']['features']

    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            audio_path = os.path.join(input_dir, file_name)
            audio, sr = librosa.load(audio_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=config['training']['input_dim'])
            
            if mfccs.size == 0:
                logging.warning(f"MFCC extraction failed for {file_name}, empty features.")
                continue
            
            # Ensure correct input shape for the model
            mfccs = mfccs.T
            if mfccs.shape[1] != config['training']['input_dim']:
                logging.error(f"MFCC shape mismatch for {file_name}: Expected {config['training']['input_dim']}, got {mfccs.shape[1]}")
                continue

            logging.debug(f"MFCC Shape for {file_name}: {mfccs.shape}")
            feature_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.npy")
            np.save(feature_path, mfccs)