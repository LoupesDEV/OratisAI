import argparse
import logging
import time
from scripts import preprocess, extract_features, train_model, decode_audio, evaluate
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main(segment_audio):
    while True:
        logging.info("=== Lancement du pipeline ASR ===")
        preprocess(config, segment_audio)
        extract_features(config)
        train_model(config)
        decode_audio(config)
        evaluate(config)
        logging.info("=== Cycle terminé. Redémarrage ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lancement du pipeline ASR ORATIS")
    parser.add_argument("--segment", choices=["yes", "no"], default="yes", help="Segmenter les fichiers audio en morceaux de 30 secondes")
    
    args = parser.parse_args()
    segment_audio = args.segment == "yes"

    main(segment_audio)
