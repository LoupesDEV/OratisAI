import os
import librosa
import numpy as np

def extract_features(input_dir, output_dir, n_mfcc=13):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            audio_path = os.path.join(input_dir, file_name)
            audio, sr = librosa.load(audio_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            feature_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.npy")
            np.save(feature_path, mfccs)
            print(f"MFCCs shape for {file_name}: {mfccs.shape}")
            print(f"Extracted features for: {file_name}")

if __name__ == "__main__":
    extract_features("data/processed", "features")