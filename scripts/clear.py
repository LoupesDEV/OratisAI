import os

def clear(processed_dir):
    if os.path.exists(processed_dir):
        for file_name in os.listdir(processed_dir):
            file_path = os.path.join(processed_dir, file_name)
            try:
                os.remove(file_path)
                print(f"Fichier supprimé : {file_path}")
            except Exception as e:
                print(f"Impossible de supprimer {file_path}: {e}")
    else:
        print(f"Le dossier {processed_dir} n'existe pas.")


if __name__ == "__main__":
    clear("data/processed")