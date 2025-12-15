# Examen DVC et Dagshub

**Nom :** FABRE  
**Prénom :** Claudia  
**Email :** claudiafabre38@gmail.com  
**Lien DagHub :** https://dagshub.com/shiff-oumi/examen-dvc

## Description
Pipeline de Machine Learning avec DVC pour prédire la concentration de silice dans un processus de flottation.

## Structure du projet
```
examen-dvc/
├── src/
│   ├── config/
│   │   ├── config.py          # Chargement de la configuration
│   │   └── config.yaml        # Paramètres du pipeline
│   ├── data/
│   │   ├── import_data.py     # Téléchargement des données
│   │   ├── make_dataset.py    # Split train/test
│   │   └── normalize_data.py  # Normalisation
│   └── models/
│       ├── searching_params.py # GridSearch
│       ├── train.py           # Entraînement
│       └── eval.py            # Évaluation
├── data/                      # Données (géré par DVC)
├── models/                    # Modèles (géré par DVC)
├── metrics/                   # Métriques d'évaluation
├── dvc.yaml                   # Pipeline DVC
├── dvc.lock                   # Versions DVC
├── Makefile                 # Test des différentes étapes de scripts
├── requirements.txt           # Dépendances Python
└── README.md
```

## Installation
```bash
# Créer l'envrironnement virtuel
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows

# Installer les dépendances
pip install -r requirements.txt

# Configurer DVC remote (remplacer par vos credentials)
dvc remote modify origin --local access_key_id YOUR_USERNAME
dvc remote modify origin --local secret_access_key YOUR_TOKEN
```

## Utilisation
```bash
# Récupérer les données
dvc pull

# Exécuter le pipeline complet
dvc repro

# Voir le graphe du pipeline
dvc dag
```

## Pipeline

Le pipeline est composé de 6 étapes :

1. **import_data** : Téléchargement des données brutes
2. **make_dataset** : Séparation train/test
3. **normalize** : Normalisation des features
4. **gridsearch** : Recherche des meilleurs hyperparamètres
5. **training** : Entraînement du modèle 
6. **evaluate** : Évaluation et génération des prédictions

## Résultats

Les métriques sont disponibles dans `metrics/scores.json`.
Le modèle est disponible dans `models/trained_model.pkl`

N'hésitez pas à rajouter les dossiers ou les fichiers qui vous semblent pertinents.

Vous devez dans un premier temps *Fork* le repo et puis le cloner pour travailler dessus. Le rendu de cet examen sera le lien vers votre dépôt sur DagsHub. Faites attention à bien mettre https://dagshub.com/licence.pedago en tant que colaborateur avec des droits de lecture seulement pour que ce soit corrigé.