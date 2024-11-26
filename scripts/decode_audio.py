import torch
import numpy as np
import os
from train_model import ASRModel

def decode_output(logits, vocabulary):
    """
    Décoder les sorties du modèle (logits) en texte lisible.
    On utilise une méthode simple de décodage greedy.
    Args:
        logits (torch.Tensor): Probabilités de sortie du modèle (logits).
        vocabulary (list): Liste des caractères/phonèmes utilisés pour l'encodage.

    Returns:
        str: Texte décodé.
    """
    probs = torch.softmax(logits, dim=-1)  # Convertir les logits en probabilités
    predicted_indices = torch.argmax(probs, dim=-1)  # Obtenir les indices des max probas
    predicted_text = ""
    prev_index = None
    for index in predicted_indices[0]:  # Parcourir la séquence
        if index != prev_index:  # Supprimer les doublons consécutifs (CTC decoding)
            if index != 0:  # Ignore le caractère spécial "blank" (index 0)
                predicted_text += vocabulary[index]
        prev_index = index
    return predicted_text


def decode_audio(model_path, feature_dir, output_dir, vocabulary):
    """
    Charger un modèle pré-entraîné et décoder les fichiers audio en texte.

    Args:
        model_path (str): Chemin vers le fichier du modèle sauvegardé (.pth).
        feature_dir (str): Répertoire contenant les caractéristiques extraites des fichiers audio.
        output_dir (str): Répertoire où sauvegarder les transcriptions décodées.
        vocabulary (list): Liste des caractères ou phonèmes utilisés pour le décodage.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_dim = 13  # Nombre de MFCCs (doit correspondre au modèle entraîné)
    hidden_dim = 128
    output_dim = 29  # Correspond à 26 lettres, espace, et blank

    model = ASRModel(input_dim=13, hidden_dim=128, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    print(f"Modèle chargé depuis {model_path}")

    for file_name in os.listdir(feature_dir):
        if file_name.endswith(".npy"):  # Vérifier si le fichier est une caractéristique audio
            feature_path = os.path.join(feature_dir, file_name)
            features = np.load(feature_path)  # Charger les MFCCs
            features = torch.tensor(features.T, dtype=torch.float32).unsqueeze(0)  # Transpose les MFCCs pour correspondre au modèle

            with torch.no_grad():
                logits = model(features)  # Obtenir les logits du modèle
                decoded_text = decode_output(logits, vocabulary)  # Décoder en texte

                # Sauvegarder la transcription
                output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.txt")
                with open(output_path, "w") as f:
                    f.write(decoded_text)
                print(f"Décodé et sauvegardé : {output_path}")


if __name__ == "__main__":
    vocabulary = [
        "_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " "
    ]

    decode_audio(
        model_path="models/trained_model.pth",
        feature_dir="features",
        output_dir="results/decoded_texts",
        vocabulary=vocabulary
    )