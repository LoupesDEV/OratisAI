import os
import pytest
from scripts import decode_audio
import yaml

def test_decode():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    decode_audio.run(config)
    decoded_files = os.listdir(config['paths']['decoded_texts'])
    assert len(decoded_files) > 0, "Le décodage a échoué. Aucun fichier généré."