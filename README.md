# ORATIS - Automatic Speech Recognition (ASR) Pipeline

## Description
ORATIS est un pipeline complet pour la reconnaissance vocale automatique (ASR). 
Le projet implémente les étapes clés nécessaires pour convertir des fichiers audio en texte, notamment :

- **Prétraitement** des fichiers audio.
- **Extraction des caractéristiques** (MFCCs).
- **Entraînement** d'un modèle basé sur un LSTM.
- **Décodage** pour générer des transcriptions.
- **Évaluation** des résultats avec des métriques WER (Word Error Rate) et CER (Character Error Rate).

## Structure du projet

```txt
ORATIS/
├── config/
│   └── config.yaml        # Configuration centralisée
├── data/
│   ├── raw/               # Fichiers audio bruts
│   ├── processed/         # Fichiers audio prétraités
│   ├── transcripts/       # Transcriptions de référence
├── features/              # Caractéristiques audio extraites (MFCCs)
├── models/
│   └── trained_model.pth  # Modèle entraîné
├── results/
│   └── decoded_texts/     # Transcriptions générées
├── scripts/
│   ├── __init__.py        # Rendre les scripts importables comme un module
│   ├── preprocess.py      # Prétraitement des fichiers audio
│   ├── extract_features.py # Extraction des caractéristiques (MFCCs)
│   ├── train_model.py     # Entraînement du modèle
│   ├── decode_audio.py    # Décodage des fichiers audio
│   └── evaluate.py        # Évaluation des performances du modèle
├── tests/
│   ├── __init__.py        # Tests unitaires
│   ├── test_preprocess.py # Tests pour le prétraitement
│   └── test_decode.py     # Tests pour le décodage
├── main.py                # Script principal pour exécuter le pipeline
├── README.md              # Documentation
├── requirements.txt       # Dépendances du projet
├── LICENSE                # License du projet
└── .gitignore             # Fichiers à ignorer dans Git
```

---

## Installation

1. **Cloner le projet :**
   ```bash
   git clone https://github.com/KucoDEV/Oratis.git
   cd Oratis
   ```

2. **Installer les dépendances :**
   Assurez-vous que Python 3.8 ou plus récent est installé.
   ```bash
   pip install -r requirements.txt
   ```

---

## Utilisation Directe
Placez vos fichiers audio bruts (`.wav` ou `.mp3`) dans le dossier `data/raw/`.
Lors du lancement de `main.py`, vous pouvez choisir si vous souhaitez **segmenter les fichiers audio** en segments de 30 secondes ou **utiliser le fichier complet**.

### Exemple d'exécution :
```bash
python main.py --segment yes
```
*Options disponibles :* 
- `--segment yes` : Segmente les fichiers audio en portions de 30 secondes.
- `--segment no`  : Utilise les fichiers audio dans leur intégralité.

---

## Utilisation Manuelle

### Étape 1 : Prétraitement des fichiers audio
Placez vos fichiers audio bruts dans `data/raw/`. Ensuite, exécutez le script de prétraitement :
```bash
python scripts/preprocess.py --segment yes
```
Les fichiers seront sauvegardés dans `data/processed/`.

### Étape 2 : Extraction des caractéristiques
Pour extraire les MFCCs des fichiers prétraités :
```bash
python scripts/extract_features.py
```
Les caractéristiques audio seront sauvegardées dans `features/`.

### Étape 3 : Entraînement du modèle
Pour entraîner le modèle LSTM :
```bash
python scripts/train_model.py
```
Le modèle entraîné sera sauvegardé dans `models/trained_model.pth`.

### Étape 4 : Décodage des fichiers audio
Pour générer des transcriptions :
```bash
python scripts/decode_audio.py
```
Les transcriptions seront sauvegardées dans `results/decoded_texts/`.

### Étape 5 : Évaluation des résultats
Pour évaluer la performance avec WER et CER :
```bash
python scripts/evaluate.py
```

---

## Résolution des problèmes

- **Problème :** Erreur `RuntimeError: input.size(-1) must be equal to input_size`.
  - **Solution :** Vérifiez que les MFCCs extraites ont une dimension correcte (13 coefficients). La transposition dans `decode_audio.py` est cruciale.

- **Problème :** WER ou CER élevés.
  - **Solution :**
    - Assurez-vous que les transcriptions dans `data/transcripts/` correspondent exactement aux fichiers audio.
    - Ajoutez plus de données d'entraînement.
    - Augmentez le nombre d'époques ou la complexité du modèle.

- **Problème :** `FileNotFoundError` lors de l'évaluation.
  - **Solution :** Vérifiez que les fichiers de référence existent dans `data/transcripts/`. Le nom du fichier de transcription doit correspondre au fichier audio.

---

## Licence
Ce projet est sous licence **MIT**. Consultez le fichier `LICENSE` pour plus de détails.

