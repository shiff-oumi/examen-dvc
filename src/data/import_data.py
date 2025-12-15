import requests
import os
import logging
from src.data.check_structure import check_existing_file, check_existing_folder
from src.config.config import load_config

def import_raw_data(raw_data_relative_path,
                    file_url):
    """Telcharge un fichier depuis une URL et le sauvegarde dans le dossier raw_data_relative_path."""
    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path)

    filename = os.path.basename(file_url)
    output_file = os.path.join(raw_data_relative_path, filename)

    if check_existing_file(output_file):
        response = requests.get(file_url)
        if response.status_code == 200:
            # Process the response content as needed
            content = response.text
            text_file = open(output_file, "wb")
            text_file.write(content.encode('utf-8'))
            text_file.close()
        else:
            print(f'Error accessing the object {file_url}:', response.status_code)
                
def main():
    config = load_config()
    raw_url = config["raw_data"]["input_url"]
    raw_folder = config["raw_data"]["output_folder"]

    import_raw_data(raw_folder,raw_url)
    logger = logging.getLogger(__name__)
    logger.info('making raw data set')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()