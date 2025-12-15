import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data.check_structure import check_existing_file, check_existing_folder
from src.config.config import load_config

logger = logging.getLogger(__name__)

def normalize_dataset(input_file,scaler):
    df = pd.read_csv(input_file)
    scaled_values = scaler.fit_transform(df.values)
    return pd.DataFrame(scaled_values, columns=df.columns)

def create_folder_if_necessary(output_folderpath):
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath, exist_ok=True)

def save_dataframes(X_train, X_test, output_folderpath):
    files = [X_train, X_test]
    names = [f"X_train_scaled.csv", f"X_test_scaled.csv"]
    for df, name in zip(files, names):
        output_filepath = os.path.join(output_folderpath, name)
        if check_existing_file(output_filepath):
            df.to_csv(output_filepath, index=False)

def main():
    logger.info("normalisation en cours")

    config = load_config()
    input_folder = config["normalize_data"]["input_folder"]
    output_folder = config["normalize_data"]["output_folder"]
    input_files = config["normalize_data"]["input_filenames"]

    create_folder_if_necessary(output_folder)


    scaler = StandardScaler()

    for filename in input_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".csv", "_scaled.csv"))

        logger.info(f"Normalisation de {input_path} -> {output_path}")
        X_train_scaled = normalize_dataset(os.path.join(input_folder, "X_train.csv"), scaler)
        X_test_scaled = normalize_dataset(os.path.join(input_folder, "X_test.csv"), scaler)
        
        save_dataframes(X_train_scaled, X_test_scaled, output_folder)


    logger.info("normalisation terminé")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
