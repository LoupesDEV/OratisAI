import os
import logging
import torch
from torch import nn, optim
from scripts.model import ASRModel
import numpy as np

def run(config):
    input_dir = config['paths']['features']
    transcript_dir = config['paths']['transcripts']
    model_save_path = config['paths']['models']

    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']

    vocabulary = [
        "_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " "
    ]

    model = ASRModel(input_dim=config['training']['input_dim'], hidden_dim=config['training']['hidden_dim'], output_dim=len(vocabulary))
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        files_processed = 0

        for feature_file in os.listdir(input_dir):
            feature_path = os.path.join(input_dir, feature_file)
            transcript_file = feature_file.replace('.npy', '.txt')
            transcript_path = os.path.join(transcript_dir, transcript_file)

            if not os.path.exists(transcript_path):
                logging.warning(f"Missing transcription for {feature_file}, skipped.")
                continue

            features = torch.tensor(np.load(feature_path), dtype=torch.float32).unsqueeze(0)
            if features.size(-1) != config['training']['input_dim']:
                logging.error(f"Feature shape mismatch for {feature_file}: Expected {config['training']['input_dim']}, got {features.size(-1)}")
                continue

            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()

            labels = torch.tensor([vocabulary.index(c) for c in transcript if c in vocabulary], dtype=torch.long).unsqueeze(0)
            input_lengths = torch.tensor([features.size(1)], dtype=torch.long)
            target_lengths = torch.tensor([labels.size(1)], dtype=torch.long)

            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output.transpose(0, 1), labels, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            files_processed += 1

        avg_loss = epoch_loss / files_processed if files_processed > 0 else float("inf")
        logging.info(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved at {model_save_path}")
