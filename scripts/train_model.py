import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence

class ASRDataset(Dataset):
    def __init__(self, feature_dir, transcript_dir):
        self.feature_files = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith(".npy")]
        self.transcripts = []
        for f in self.feature_files:
            base_name = os.path.basename(f).replace("_processed", "").replace(".npy", ".txt")
            with open(os.path.join(transcript_dir, base_name), "r") as file:
                self.transcripts.append(file.read().strip())

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        features = np.load(self.feature_files[idx])
        transcript = self.transcripts[idx]
        return torch.tensor(features, dtype=torch.float32), transcript

def collate_fn(batch):
    features, transcripts = zip(*batch)
    lengths = torch.tensor([f.shape[1] for f in features])  # Longueur de chaque séquence
    padded_features = pad_sequence([f.T for f in features], batch_first=True, padding_value=0.0)  # Transpose chaque MFCC
    return padded_features, lengths, transcripts


class ASRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ASRModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

def train_model(feature_dir, transcript_dir, save_path, epochs=10, batch_size=16):
    dataset = ASRDataset(feature_dir, transcript_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    input_dim = 13  # Nombre de MFCCs
    hidden_dim = 128
    output_dim = 29  # Nombre de caractères dans le vocabulaire

    model = ASRModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CTCLoss(blank=0)  # Blank index pour le décodage CTC
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for features, lengths, transcripts in dataloader:
            optimizer.zero_grad()

            # Convertir les transcriptions en indices numériques
            targets = []
            target_lengths = []
            for t in transcripts:
                encoded = [ord(c) - ord('a') + 1 for c in t.lower() if c.isalpha()]  # Exemple : 'a' -> 1, 'z' -> 26
                targets.extend(encoded)
                target_lengths.append(len(encoded))
            
            # Convertir les données en tenseurs
            targets = torch.tensor(targets, dtype=torch.long)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)
            output_lengths = torch.tensor(lengths, dtype=torch.long)

            # Passer les caractéristiques au modèle
            outputs = model(features)  # Sortie de taille [batch_size, max_seq_length, output_dim]
            outputs = outputs.log_softmax(2)  # LogSoftmax pour CTC

            # Calcul de la perte
            loss = criterion(outputs.permute(1, 0, 2), targets, output_lengths, target_lengths)  # Permuter pour CTC
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Sauvegarde du modèle
    torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_model("features", "data/transcripts", "models/trained_model.pth")