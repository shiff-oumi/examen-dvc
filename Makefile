import_raw_data:
	python -m src.data.import_data

preprocess_data:
	python -m src.data.make_dataset

normalize_data:
	python -m src.data.normalize_data

grid_search:
	python -m src.models.seraching_params

train : 
	python -m src.models.train

evaluate:
	python -m src.models.eval
	

# Nettoyer les fichiers téléchargés
clean:
	rm -rf data/preprocessed_data/*.csv
	rm -rf data/raw_data/*.csv
	rm -rf data/*.csv