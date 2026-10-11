# soudure-ml-cs — Prédiction de la qualité de soudures sur aciers

Projet de Machine Learning du cours de CentraleSupélec : prédire la qualité d'une soudure à
partir de sa « recette » (composition chimique, paramètres de soudage, traitement thermique),
comparer plusieurs méthodes d'apprentissage dont une approche semi-supervisée, et en
déduire des recommandations pratiques pour bien souder.

## Contexte

La qualité des soudures est un enjeu industriel majeur, par exemple pour les mâts et
fondations d'éoliennes en mer ou les pipelines. Aujourd'hui, le savoir-faire sur ce qui fait
une bonne soudure se transmet surtout d'expert à expert, et les industriels en dépendent.
L'objectif est d'**acquérir cette connaissance à partir des données** : à la fois formaliser
le savoir expert et découvrir de nouvelles tendances.

## Les données

Base publique **welddb** de l'Université de Cambridge (Cool & Bhadeshia, 1996-1997),
fournie dans `welddb/` :

- **1 652 soudures** (métal fondu déposé), **44 colonnes**, issues de **46 publications** ;
- **entrées (la recette)** : 21 éléments chimiques (en % massique ou en ppm), courant,
  tension, type de courant et polarité, apport de chaleur, température entre passes, procédé
  de soudage, traitement thermique après soudage (température, durée) ;
- **sorties (propriétés mesurées)** : limite d'élasticité, résistance à la traction,
  allongement, striction, résilience Charpy (énergie et température d'essai), dureté, etc. ;
- **base très incomplète** : `N` signifie « non publié », et chaque propriété n'est mesurée
  que sur une partie des soudures (environ 50 %).

## La démarche en bref

| Étape | Choix principal | Détails |
|---|---|---|
| Définition de la qualité | Deux facettes, deux modèles de **régression** : la **résistance** (limite d'élasticité) et la **ténacité** (résilience Charpy, avec la température d'essai en entrée) | [Méthodologie §3](docs/methodologie.md#3-définir-la-qualité-dune-soudure) |
| Nettoyage | Conversion des valeurs textuelles (`<0.01`, `150-200`…), correction de 2 erreurs d'unité, variables presque vides écartées | [§2](docs/methodologie.md#2-nettoyage-des-données) |
| Pré-traitement | 0 pour les éléments d'alliage non renseignés, médiane sinon, encodage one-hot, standardisation, le tout dans un `Pipeline` | [§4](docs/methodologie.md#4-pré-traitement) |
| ACP | 10 composantes pour 80 % de la variance ; axes « alliage + traitement thermique », « énergie de soudage », « impuretés » ; groupes par publication | [§1.4](docs/methodologie.md#14-acp--développer-une-intuition) |
| Validation | **Validation croisée imbriquée, groupée par publication** : un découpage aléatoire sous-estime l'erreur de moitié | [§5](docs/methodologie.md#5-protocole-de-validation) |
| Métriques | RMSE (principale), MAE, R², part dans la tolérance, **taux de fausse acceptation** | [§6](docs/methodologie.md#6-métriques) |
| Modèles | Régression linéaire, k plus proches voisins, SVR, forêt aléatoire (vus en cours), plus deux références | [§7](docs/methodologie.md#7-modèles) |
| Semi-supervisé | Self-training avec forêt aléatoire, confiance mesurée par l'accord entre les arbres | [§8](docs/methodologie.md#8-apprentissage-semi-supervisé) |

## Résultats clés

Erreurs mesurées sur des **publications jamais vues** pendant l'entraînement :

| Modèle | Limite d'élasticité | Résilience Charpy |
|---|---|---|
| Moyenne (référence) | RMSE 93 MPa, R² −0,06 | RMSE 51 J, R² −0,18 |
| Température d'essai seule (référence) | — | RMSE 45 J, R² 0,03 |
| Régression linéaire | 105 MPa, R² −0,41 | 67 J, R² −1,35 |
| k plus proches voisins | 88 MPa, R² 0,05 | 49 J, R² −0,09 |
| **SVR (retenu)** | **67 MPa, R² 0,45** | **40 J, R² 0,28** |
| Forêt aléatoire | 67 MPa, R² 0,44 | 45 J, R² 0,07 |

- **Approche retenue : un SVR par propriété.** Il est au moins aussi bon que la forêt
  aléatoire, meilleur pour la ténacité et pour l'erreur la plus dangereuse (accepter une
  soudure trop faible). L'écart n'est cependant pas statistiquement significatif.
- **La résistance est bien mieux prédite que la ténacité.**
- **Le semi-supervisé n'apporte pas de gain** : il est 1 à 2 MPa moins bon que le supervisé.
- **Usage réaliste** : présélectionner des recettes avant les essais, pas remplacer les essais
  mécaniques.
- **Recommandations aux soudeurs** : un consommable à environ 1,0–1,4 % de Mn, bas carbone et
  peu oxydé (enrobages et flux basiques) ; un apport de chaleur modéré ; une température
  entre passes contrôlée ; les éléments d'alliage durcissants (Cr, Mo, V, Nb) seulement si
  nécessaire, car ils dégradent la ténacité.

Détails, figures et discussion : [docs/resultats.md](docs/resultats.md).

## Structure du dépôt

```
soudure-ml-cs/
├── README.md                  ce fichier
├── Sujet-soudures.pdf         sujet du projet
├── requirements.txt           librairies Python
├── welddb/                    données brutes et leur description (welddb.info)
├── src/
│   ├── welddb.py              chargement, noms de colonnes, conversion du format brut, publication de chaque soudure
│   └── preprocessing.py       nettoyage complet et pré-traitement (Pipeline scikit-learn)
├── notebooks/
│   ├── 01_analyse_descriptive.ipynb           format, valeurs manquantes, distributions, unités, corrélations, ACP
│   ├── 02_variables_qualite_strategie.ipynb   variables de qualité, variables influentes, stratégie de prédiction
│   ├── 03_modeles_limite_elasticite.ipynb     nettoyage, 4 modèles, validation imbriquée, self-training
│   ├── 04_charpy.ipynb                        même démarche pour la résilience Charpy
│   └── 05_comparaison_recommandations.ipynb   comparaison globale, choix final, effet des variables, recommandations
├── results/                   scores et prédictions hors pli (produits par 03 et 04, lus par 05)
└── docs/
    ├── methodologie.md        tous les choix et leurs justifications
    └── resultats.md           résultats détaillés, recommandations, limites
```

## Installation et exécution

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Exécuter les notebooks **dans l'ordre** (01 → 05) : le 05 lit les résultats enregistrés par
le 03 et le 04 dans `results/`. Temps d'exécution : environ 1 à 2 minutes pour chacun des
notebooks 03 et 04, quelques secondes à 1 minute pour les autres. Tous les résultats sont
reproductibles (graines aléatoires fixées).

## Références principales

- Cool, T., Bhadeshia, H. K. D. H., & MacKay, D. J. C. (1997). The yield and ultimate tensile strength of steel welds. *Materials Science and Engineering A*, 223.
- Données : [MAP_DATA_WELD, Phase Transformations Group, University of Cambridge](https://www.phase-trans.msm.cam.ac.uk/map/data/materials/welddb-b.html).
- Les autres références (semi-supervisé, métallurgie du soudage, normes) sont listées dans [docs/methodologie.md](docs/methodologie.md#références) et [docs/resultats.md](docs/resultats.md#références).
