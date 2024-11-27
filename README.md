
# ORATIS - Automatic Speech Recognition (ASR) Pipeline

## Description
ORATIS est un pipeline complet pour la reconnaissance vocale automatique (ASR). 
<br>Le projet implémente les étapes clés nécessaires pour convertir des fichiers audio en texte, notamment :
- **Prétraitement** des fichiers audio.
- **Extraction des caractéristiques** (MFCCs).
- **Entraînement** d'un modèle basé sur un LSTM.
- **Décodage** pour générer des transcriptions.
- **Évaluation** des résultats avec des métriques WER (Word Error Rate) et CER (Character Error Rate).

## Structure du projet
```
ORATIS/
├── data/
│   ├── raw/                # Fichiers audio bruts
│   ├── processed/          # Fichiers audio prétraités
│   ├── transcripts/        # Transcriptions de référence
│   └── features/           # Caractéristiques audio extraites (MFCCs)
├── models/
│   └── trained_model.pth   # Modèle entraîné
├── results/
│   └── decoded_texts/      # Transcriptions générées
├── scripts/
│   ├── preprocess.py       # Prétraitement des fichiers audio
│   ├── extract_features.py # Extraction des caractéristiques (MFCCs)
│   ├── train_model.py      # Entraînement du modèle
│   ├── decode_audio.py     # Décodage des fichiers audio
│   └── evaluate.py         # Évaluation des performances du modèle
├── main.py                 # Script principal pour exécuter le pipeline
├── README.md               # Documentation
├── requirements.txt        # Dépendances du projet
└── LICENSE                 # License du projet
```

## Installation

1. **Cloner le projet :**
   ```bash
   git clone https://github.com/KucoDEV/Oratis.git
   cd ORATIS
   ```

2. **Installer les dépendances :**
   Assurez-vous que Python 3.8 ou plus récent est installé.
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation Direct
Placez vos fichiers audio bruts (`.wav` ou `.mp3`) dans le dossier `data/raw/`.
```bash
python main.py
```

## Utilisation Différé

### Étape 1 : Prétraitement des fichiers audio
Placez vos fichiers audio bruts (`.wav` ou `.mp3`) dans le dossier `data/raw/`. Ensuite, exécutez le script de prétraitement :
```bash
python scripts/preprocess.py
```
Les fichiers seront normalisés et sauvegardés dans `data/processed/`.

---

### Étape 2 : Extraction des caractéristiques
Pour extraire les MFCCs des fichiers prétraités :
```bash
python scripts/extract_features.py
```
Les caractéristiques audio seront sauvegardées dans `features/`.

---

### Étape 3 : Entraînement du modèle
Pour entraîner le modèle LSTM :
```bash
python scripts/train_model.py
```
Le modèle entraîné sera sauvegardé dans `models/trained_model.pth`.

---

### Étape 4 : Décodage des fichiers audio
Pour générer des transcriptions des fichiers audio traités :
```bash
python scripts/decode_audio.py
```
Les transcriptions générées seront sauvegardées dans `results/decoded_texts/`.

---

### Étape 5 : Évaluation des résultats
Pour évaluer la performance du modèle avec des métriques WER et CER :
```bash
python scripts/evaluate.py
```

---

## Résolution des problèmes

- **Problème :** Erreur `RuntimeError: input.size(-1) must be equal to input_size`.
  - **Solution :** Vérifiez que les MFCCs extraites ont une dimension correcte (13 coefficients). La transposition des données dans `decode_audio.py` est cruciale.

- **Problème :** WER ou CER élevés.
  - **Solution :** 
    - Assurez-vous que les transcriptions dans `data/transcripts/` correspondent exactement aux fichiers audio.
    - Ajoutez plus de données d'entraînement.
    - Augmentez le nombre d'époques ou la complexité du modèle.

---

## Auteurs
Ce projet a été réalisé par **Mathéo PICHOT-MOÏSE** alias **Kuco**.

---

## Licence
Ce projet est sous licence **MIT**. Consultez le fichier `LICENSE` pour plus de détails.
