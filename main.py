import os
import subprocess

def ensure_directories():
    required_dirs = [
        "data/raw", "data/processed", "data/transcripts",
        "features", "models", "results/decoded_texts"
    ]
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Vérification/Création du dossier : {directory}")

def preprocess_audio():
    print("\n[Étape 1/5] Prétraitement des fichiers audio...")
    subprocess.run(["python", "scripts/preprocess.py"])

def extract_features():
    print("\n[Étape 2/5] Extraction des caractéristiques (MFCCs)...")
    subprocess.run(["python", "scripts/extract_features.py"])

def train_model():
    print("\n[Étape 3/5] Entraînement du modèle...")
    subprocess.run(["python", "scripts/train_model.py"])

def decode_audio():
    print("\n[Étape 4/5] Décodage des fichiers audio...")
    subprocess.run(["python", "scripts/decode_audio.py"])

def evaluate_model():
    print("\n[Étape 5/5] Évaluation du modèle...")
    subprocess.run(["python", "scripts/evaluate.py"])

if __name__ == "__main__":
    print("=== Lancement du pipeline ASR ===")
    ensure_directories()
    preprocess_audio()
    extract_features()
    train_model()
    decode_audio()
    evaluate_model()
    print("\n=== Pipeline terminé ===")