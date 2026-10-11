"""Nettoyage complet et pré-traitement des entrées, à partir de `welddb.load()`.

Les choix sont justifiés dans les notebooks 01 et 02 :
- valeurs en ppm inférieures à 1 : probablement saisies en % massique -> manquantes ;
- vanadium "<5" : limite de détection en ppm dans une colonne en % -> 0,0005 % ;
- variables renseignées sur moins de 20 % des lignes étiquetées : écartées ;
- éléments d'alliage non renseignés : non ajoutés -> 0 ;
- autres valeurs numériques manquantes : médiane (calculée sur l'entraînement) ;
- catégories manquantes : catégorie "inconnu", catégories rares regroupées ;
- standardisation de toutes les variables numériques.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import welddb

DROPPED = ["Co", "W", "Sn_ppm", "As_ppm", "Sb_ppm"]

# Éléments ajoutés volontairement : absents de la publication -> pas ajoutés
ALLOY = ["Ni", "Cr", "Mo", "V", "Cu", "Ti_ppm", "Al_ppm", "B_ppm", "Nb_ppm"]
# Toujours présents dans une soudure, ou paramètres toujours fixés : médiane
OTHER_NUM = (["C", "Si", "Mn", "S", "P", "O_ppm", "N_ppm"] + welddb.PROCESS_NUM)
CATEGORICAL = ["weld_type", "ac_dc", "polarity"]
FEATURES = ALLOY + OTHER_NUM + CATEGORICAL


def clean(df, extra=()):
    """Corrige les valeurs aberrantes connues et renvoie les colonnes d'entrée.

    `extra` : colonnes numériques supplémentaires (ex. température d'essai Charpy).
    """
    df = df.copy()
    # "<5" dans la colonne V (% massique, max ~0,3 % ailleurs) : limite exprimée en ppm
    df.loc[df["V"] > 1, "V"] /= 10_000
    for col in welddb.COMPOSITION_PPM:
        df.loc[df[col].between(0, 1, inclusive="neither"), col] = float("nan")
    df[CATEGORICAL] = df[CATEGORICAL].astype(object).fillna("inconnu")
    return df[FEATURES + list(extra)]


def make_preprocessor(extra=()):
    """Imputation, encodage et standardisation, à mettre en tête d'un Pipeline."""
    return ColumnTransformer([
        ("alloy", make_pipeline(SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True),
                                StandardScaler()), ALLOY),
        ("other", make_pipeline(SimpleImputer(strategy="median"),
                                StandardScaler()), OTHER_NUM + list(extra)),
        ("cat", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist",
                              sparse_output=False), CATEGORICAL),
    ])
