import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.check_structure import check_existing_file, check_existing_folder
from src.config.config import load_config

logger = logging.getLogger(__name__)

def import_dataset(file_path):
    return pd.read_csv(file_path)

def drop_columns(df,columns_to_drop):
    df.drop(columns_to_drop, axis=1, inplace=True, errors="ignore")
    return df

def split_data(df, target_column, test_size, random_state):
    target = df[target_column]
    feats = df.drop([target_column], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        feats, target, test_size=test_size, random_state=random_state
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
    preprocess_config = config["preprocess_data"]

    input_file = os.path.join(preprocess_config["input_folder"],preprocess_config["input_filename"])
    output_folder = preprocess_config["output_folder"]
    columns_to_drop = preprocess_config["drop_col"]
    target_column = preprocess_config["target_col"]
    test_size = preprocess_config["test_size"]
    random_state = preprocess_config["random_state"]

    df = import_dataset(input_file)

    # Pipeline
    df = drop_columns(df, columns_to_drop)
    X_train, X_test, y_train, y_test = split_data(df, target_column, test_size, random_state)

    # Sortie
    create_folder_if_necessary(output_folder)
    save_dataframes(X_train, X_test, y_train, y_test, output_folder)

    logger.info("preprocessing terminé")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
