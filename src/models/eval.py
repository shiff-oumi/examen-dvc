import os
import logging
import pickle
import json
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from src.config.config import load_config
from src.data.check_structure import check_existing_folder

logger = logging.getLogger(__name__)

def main():
    logger.info("Évaluation du modèle en cours...")

    # Charger la config
    config = load_config()
    processed_folder = config["gridsearch"]["input_folder"]
    models_folder = config["gridsearch"]["output_folder"]
    metrics_folder = "metrics"
    check_existing_folder(metrics_folder)

    # Charger le modèle entraîné
    trained_model_file = os.path.join(models_folder, "trained_model.pkl")
    with open(trained_model_file, "rb") as f:
        model = pickle.load(f)

    # Charger les datasets de test
    X_test = pd.read_csv(os.path.join(processed_folder, config["gridsearch"]["x_test_file"]))
    y_test = pd.read_csv(os.path.join(processed_folder, config["gridsearch"]["y_test_file"]))

    # Prédictions
    y_pred = model.predict(X_test)

    # Sauvegarder les prédictions dans data/predictions.csv
    predictions_file = os.path.join("data", "predictions.csv")
    df_pred = pd.DataFrame({"y_true": y_test.values.ravel(), "y_pred": y_pred})
    df_pred.to_csv(predictions_file, index=False)
    logger.info(f" Prédictions sauvegardées dans {predictions_file}")

    # Calcul des métriques
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    scores = {"mse": mse, "r2": r2}

    # Sauvegarder les scores dans metrics/scores.json
    scores_file = os.path.join(metrics_folder, "scores.json")
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=4)
    logger.info(f" Scores sauvegardés dans {scores_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
