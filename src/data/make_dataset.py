import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.check_structure import check_existing_file, check_existing_folder
from src.config.config import load_config

logger = logging.getLogger(__name__)

def import_dataset(file_path):
    return pd.read_csv(file_path)

def drop_columns(df):
    list_to_drop = ["date"]
    df.drop(list_to_drop, axis=1, inplace=True, errors="ignore")
    return df

def split_data(df):
    target = df["silica_concentrate"]
    feats = df.drop(["silica_concentrate"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        feats, target, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test

def create_folder_if_necessary(output_folderpath):
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath, exist_ok=True)

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    files = [X_train, X_test, y_train, y_test]
    names = [f"X_train.csv", f"X_test.csv",
             f"y_train.csv", f"y_test.csv"]
    for df, name in zip(files, names):
        output_filepath = os.path.join(output_folderpath, name)
        if check_existing_file(output_filepath):
            df.to_csv(output_filepath, index=False)

def main():
    logger.info("preprocessing en cours")

    config = load_config()
    input_file = os.path.join(config["preprocess_data"]["input_folder"],config["preprocess_data"]["input_filename"])
    output_folder = config["preprocess_data"]["output_folder"]

    df = import_dataset(input_file)

    # Pipeline
    df = drop_columns(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Sortie
    create_folder_if_necessary(output_folder)
    save_dataframes(X_train, X_test, y_train, y_test, output_folder)

    logger.info("preprocessing terminé")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
