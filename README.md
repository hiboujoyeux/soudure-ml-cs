# soudure-ml-cs — Prédiction de la qualité de soudures sur aciers

Projet de Machine Learning du cours de CentraleSupélec : prédire la qualité d'une soudure à
partir de sa « recette » (composition chimique, paramètres de soudage, traitement thermique),
comparer plusieurs méthodes d'apprentissage dont des approches semi-supervisées, et en
déduire des recommandations pratiques pour bien souder.

## Contexte

La qualité des soudures est un enjeu industriel majeur, par exemple pour les mâts et
fondations d'éoliennes en mer ou les pipelines. Aujourd'hui, le savoir-faire sur ce qui fait
une bonne soudure se transmet surtout d'expert à expert, et les industriels en dépendent.
L'objectif est d'**acquérir cette connaissance à partir des données** : à la fois formaliser
le savoir expert et découvrir de nouvelles tendances.

## Les données

Base publique **welddb** de l'Université de Cambridge (Cool & Bhadeshia, 1996-1997), à
placer dans `welddb/` (voir [Installation](#installation-et-exécution)) :

- **1 652 lignes** (métal fondu déposé ; une même soudure apparaît souvent sur plusieurs
  lignes, une par essai), **44 colonnes**, issues de **46 publications** ;
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
| Métriques | RMSE (principale), MAE, R², part dans la tolérance, **taux de fausse acceptation**, AUC ; marge de sécurité choisie en validation croisée | [§6](docs/methodologie.md#6-métriques) |
| Comparaison statistique | Test de Wilcoxon par publication avec correction de Holm, intervalles de confiance par bootstrap sur les publications | [§5.3](docs/methodologie.md#53-comparaison-statistique) |
| Modèles | Régression linéaire, k plus proches voisins, SVR, forêt aléatoire (vus en cours), gradient boosting, plus deux références | [§7](docs/methodologie.md#7-modèles) |
| Semi-supervisé | Self-training avec forêt aléatoire, puis régularisation de variété (Laplacian RLS), inductive et transductive | [§8](docs/methodologie.md#8-apprentissage-semi-supervisé) |

## Résultats clés

Erreurs mesurées sur des **publications jamais vues** pendant l'entraînement (moyenne des
5 plis) :

| Modèle | Limite d'élasticité | Résilience Charpy |
|---|---|---|
| Moyenne (référence) | RMSE 93 MPa, R² −0,06 | RMSE 51 J, R² −0,18 |
| Température d'essai seule (référence) | — | RMSE 45 J, R² 0,03 |
| Régression linéaire | 105 MPa, R² −0,41 | 67 J, R² −1,35 |
| k plus proches voisins | 88 MPa, R² 0,05 | 49 J, R² −0,09 |
| **SVR (retenu)** | 67 MPa, R² 0,45 | **40 J, R² 0,28** |
| Forêt aléatoire | 67 MPa, R² 0,44 | 45 J, R² 0,07 |
| Gradient boosting | **63 MPa, R² 0,50** | 47 J, R² 0,00 |

- **Approche retenue : un SVR par propriété.** C'est le seul modèle qui se comporte bien sur
  les deux cibles. Pour la ténacité, il bat la forêt aléatoire et le gradient boosting sur
  chacun des 5 plis, mais cet écart n'est pas statistiquement établi après correction pour
  comparaisons multiples (Wilcoxon et bootstrap sur les publications, corrigés par Holm) :
  environ la moitié de son avance vient d'une seule publication (Evans-StressRelief-1991).
  Il ne bat pas non plus significativement la référence « température d'essai seule ».
  Pour la limite d'élasticité, les trois meilleurs modèles ne se distinguent pas
  significativement.
- **Pour présélectionner des recettes sur la limite d'élasticité, le gradient boosting est
  préférable** : avec une marge d'environ 50 MPa, il accepte environ 10 % des soudures trop
  faibles et rejette 30 % des bonnes, contre 16 % et 52 % pour le SVR.
- **La résistance est bien mieux prédite que la ténacité.**
- **Le semi-supervisé n'apporte pas de gain**, ni le self-training (1 à 2 MPa moins bon que
  le supervisé), ni la régularisation de variété, même en voyant les recettes de test. La
  limite vient du manque de publications variées, pas du manque d'étiquettes.
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
├── requirements.txt           librairies Python
├── welddb/                    données brutes et leur description (welddb.info) — non versionné, à télécharger
├── src/
│   ├── welddb.py              chargement, noms de colonnes, conversion du format brut, publication de chaque soudure
│   └── preprocessing.py       nettoyage complet et pré-traitement (Pipeline scikit-learn)
├── notebooks/
│   ├── 01_analyse_descriptive.ipynb           format, valeurs manquantes, distributions, unités, corrélations, ACP
│   ├── 02_variables_qualite_strategie.ipynb   variables de qualité, variables influentes, effet publication, stratégie
│   ├── 03_modeles_limite_elasticite.ipynb     5 modèles, validation imbriquée, self-training, régularisation de variété
│   ├── 04_charpy.ipynb                        même démarche pour la résilience Charpy, courbes de transition
│   └── 05_comparaison_recommandations.ipynb   comparaison globale, tests statistiques, marges, choix final, recommandations
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

Les données ne sont pas versionnées. Télécharger les fichiers `welddb.data` et
`welddb.info` depuis la page
[MAP_DATA_WELD](https://www.phase-trans.msm.cam.ac.uk/map/data/materials/welddb-b.html) et
les placer dans `welddb/`, à la racine du dépôt.

Exécuter les notebooks **dans l'ordre** (01 → 05) : le 05 lit les résultats enregistrés par
le 03 et le 04 dans `results/`. Temps d'exécution : environ 2 à 3 minutes pour le notebook 03,
1 minute pour le 04, quelques secondes à 1 minute pour les autres. Tous les résultats sont
reproductibles (graines aléatoires fixées).

## Références principales

- Cool, T., Bhadeshia, H. K. D. H., & MacKay, D. J. C. (1997). The yield and ultimate tensile strength of steel welds. *Materials Science and Engineering A*, 223.
- Belkin, M., Niyogi, P., & Sindhwani, V. (2006). Manifold regularization: A geometric framework for learning from labeled and unlabeled examples. *Journal of Machine Learning Research*, 7.
- Données : [MAP_DATA_WELD, Phase Transformations Group, University of Cambridge](https://www.phase-trans.msm.cam.ac.uk/map/data/materials/welddb-b.html).
- Les autres références (semi-supervisé, métallurgie du soudage, normes) sont listées dans [docs/methodologie.md](docs/methodologie.md#références) et [docs/resultats.md](docs/resultats.md#références).
