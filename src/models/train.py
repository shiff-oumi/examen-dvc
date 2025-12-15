import os
import logging
import pickle
import pandas as pd
from sklearn.linear_model import Ridge
from src.config.config import load_config
from src.data.check_structure import check_existing_folder

logger = logging.getLogger(__name__)

def main():
    logger.info("Entraînement du modèle en cours...")

    # Charger la config
    config = load_config()
    processed_folder = config["gridsearch"]["input_folder"]
    models_folder = config["gridsearch"]["output_folder"]
    check_existing_folder(models_folder)

    # Charger les datasets
    X_train = pd.read_csv(os.path.join(processed_folder, config["gridsearch"]["x_train_file"]))
    y_train = pd.read_csv(os.path.join(processed_folder, config["gridsearch"]["y_train_file"]))

    # Charger les meilleurs paramètres
    best_params_file = os.path.join(models_folder, config["gridsearch"]["output_filename"])
    with open(best_params_file, "rb") as f:
        best_params = pickle.load(f)

    logger.info(f"Utilisation des paramètres: {best_params}")

    # Entraîner le modèle avec les meilleurs paramètres
    model = Ridge(**best_params)
    model.fit(X_train, y_train.values.ravel())

    # Sauvegarder le modèle entraîné
    trained_model_file = os.path.join(models_folder, "trained_model.pkl")
    with open(trained_model_file, "wb") as f:
        pickle.dump(model, f)

    logger.info(f" Modèle entraîné sauvegardé dans {trained_model_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
