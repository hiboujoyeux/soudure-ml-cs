"""Chargement et nettoyage de format de la base welddb (Cool & Bhadeshia, Cambridge).

Ce module ne fait que rendre les données exploitables (types, unités, libellés).
Les choix de modélisation (imputation, normalisation, sélection des cibles) sont
faits et justifiés dans les notebooks.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "welddb" / "welddb.data"

# Noms courts des 44 colonnes, dans l'ordre de welddb.info
COLUMNS = [
    # Composition chimique en % massique
    "C", "Si", "Mn", "S", "P", "Ni", "Cr", "Mo", "V", "Cu", "Co", "W",
    # Composition chimique en ppm massique
    "O_ppm", "Ti_ppm", "N_ppm", "Al_ppm", "B_ppm", "Nb_ppm", "Sn_ppm", "As_ppm", "Sb_ppm",
    # Paramètres de soudage
    "current_A", "voltage_V", "ac_dc", "polarity", "heat_input_kJmm", "interpass_temp_C",
    "weld_type", "pwht_temp_C", "pwht_time_h",
    # Propriétés mécaniques
    "yield_strength_MPa", "uts_MPa", "elongation_pct", "reduction_area_pct",
    "charpy_temp_C", "charpy_toughness_J", "hardness", "fatt50_C",
    # Microstructure
    "primary_ferrite_pct", "ferrite_second_phase_pct", "acicular_ferrite_pct",
    "martensite_pct", "ferrite_carbide_pct",
    "weld_id",
]

COMPOSITION_WT = COLUMNS[0:12]
COMPOSITION_PPM = COLUMNS[12:21]
COMPOSITION = COMPOSITION_WT + COMPOSITION_PPM
PROCESS_NUM = ["current_A", "voltage_V", "heat_input_kJmm", "interpass_temp_C",
               "pwht_temp_C", "pwht_time_h"]
PROCESS_CAT = ["ac_dc", "polarity", "weld_type"]
MECHANICAL = COLUMNS[30:38]
MICROSTRUCTURE = COLUMNS[38:43]
CATEGORICAL = PROCESS_CAT + ["weld_id"]

# Libellés synonymes d'après welddb.info (ShMA = MMA, SMA = SA) ;
# SAA (submerged arc automatic) est rattaché à SA.
WELD_TYPE_ALIASES = {"ShMA": "MMA", "SAA": "SA"}


def _parse_number(value):
    """Convertit une cellule brute en (valeur, est_sous_seuil).

    - "<0.01" : sous la limite de détection -> on garde la limite et on lève le drapeau
    - "150-200" : plage -> milieu de la plage
    - "203(Hv30)", "224Hv10" : dureté Vickers annotée -> valeur numérique
    - "66tot33res" : azote total / résiduel -> azote total
    """
    if pd.isna(value):
        return np.nan, False
    s = str(value)
    if s.startswith("<"):
        return float(s[1:]), True
    m = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2, False
    m = re.match(r"(\d+(?:\.\d+)?)(?:\(?Hv\d+\)?|tot)", s)
    if m:
        return float(m.group(1)), False
    return float(s), False


def publication_of(weld_id):
    """Source bibliographique d'une soudure, déduite de son identifiant.

    Les identifiants commencent par l'auteur et souvent l'année
    ("Evans-Ni/CMn-1990/1991-0Aaw"). Plusieurs lignes d'une même publication
    (et souvent d'une même soudure) sont très corrélées : il faut en tenir compte
    lors de la validation croisée.
    """
    m = re.match(r"^(.*?\d{4})", weld_id)
    if m:
        return m.group(1)
    return re.split(r"[-\s]", weld_id)[0]


def load_raw(path=DATA_PATH):
    """Fichier brut, toutes colonnes en texte, "N" (non publié) -> NaN."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS,
                       na_values="N", dtype=str)


def load(path=DATA_PATH):
    """Données typées, avec une colonne booléenne `<col>_below_dl` par colonne
    contenant des valeurs sous la limite de détection."""
    raw = load_raw(path)
    df = pd.DataFrame(index=raw.index)

    for col in COLUMNS:
        if col in CATEGORICAL:
            continue
        parsed = raw[col].map(_parse_number)
        df[col] = parsed.str[0].astype(float)
        below_dl = parsed.str[1].astype(bool)
        if below_dl.any():
            df[f"{col}_below_dl"] = below_dl

    df["ac_dc"] = raw["ac_dc"]
    # Polarité "0" : uniquement en courant alternatif, où la polarité n'a pas de sens
    df["polarity"] = raw["polarity"].replace({"0": "AC"})
    df["weld_type"] = raw["weld_type"].replace(WELD_TYPE_ALIASES)
    df["weld_id"] = raw["weld_id"]
    df["publication"] = raw["weld_id"].map(publication_of)
    return df
