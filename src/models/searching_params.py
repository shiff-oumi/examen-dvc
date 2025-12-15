import os
import logging
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from src.config.config import load_config
from src.data.check_structure import check_existing_folder

logger = logging.getLogger(__name__)

def main():
    logger.info("GridSearch en cours...")

    # Charger la config
    config = load_config()
    processed_folder = config["preprocess_data"]["output_folder"]
    models_folder = "models"
    check_existing_folder(models_folder)

    # Charger les datasets prétraités
    X_train = pd.read_csv(os.path.join(processed_folder, "X_train_scaled.csv"))
    X_test = pd.read_csv(os.path.join(processed_folder, "X_test_scaled.csv"))
    y_train = pd.read_csv(os.path.join(processed_folder, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(processed_folder, "y_test.csv"))

    # Choix du modèle : Ridge Regression
    model = Ridge()

    # Grille d’hyperparamètres
    param_grid = {
        "alpha": [0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "svd", "cholesky", "lsqr"]
    }

    # GridSearch
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train.values.ravel())

    # Meilleurs paramètres
    best_params = grid_search.best_params_
    logger.info(f"Meilleurs paramètres trouvés: {best_params}")

    # Évaluer sur le test set
    y_pred = grid_search.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"MSE sur le test set: {mse:.4f}")

    # Sauvegarder les meilleurs paramètres
    output_file = os.path.join(models_folder, "best_params.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(best_params, f)

    logger.info(f" Paramètres sauvegardés dans {output_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
