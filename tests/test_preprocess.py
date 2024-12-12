import os
import pytest
from scripts import preprocess
import yaml

def test_preprocess():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    preprocess.run(config)
    processed_files = os.listdir(config['paths']['processed_audio'])
    assert len(processed_files) > 0, "Le prétraitement a échoué. Aucun fichier traité."