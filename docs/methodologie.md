# Méthodologie

Ce document décrit et justifie tous les choix faits dans le projet, dans l'ordre de la
démarche. Les notebooks correspondants sont indiqués entre crochets.

Sommaire :
1. [Analyse descriptive](#1-analyse-descriptive)
2. [Nettoyage des données](#2-nettoyage-des-données)
3. [Définir la qualité d'une soudure](#3-définir-la-qualité-dune-soudure)
4. [Pré-traitement](#4-pré-traitement)
5. [Protocole de validation](#5-protocole-de-validation)
6. [Métriques](#6-métriques)
7. [Modèles](#7-modèles)
8. [Apprentissage semi-supervisé](#8-apprentissage-semi-supervisé)
9. [Interprétation et recommandations](#9-interprétation-et-recommandations)
10. [Choix envisagés puis écartés](#10-choix-envisagés-puis-écartés)

---

## 1. Analyse descriptive

[`01_analyse_descriptive.ipynb`]

### 1.1 Structure de la base

- **1 652 lignes, 44 colonnes**, issues de **46 publications** scientifiques.
- **Une soudure apparaît souvent sur plusieurs lignes.** Par exemple, dans les données
  d'Evans, le suffixe de l'identifiant indique la mesure : `aw` (brut de soudage, essai de
  traction), `awch` (essai Charpy), `ht` (après traitement thermique). **887 lignes** ont
  exactement les mêmes entrées qu'au moins une autre ligne.
- **La publication d'origine** a été déduite de l'identifiant de chaque soudure
  (`src/welddb.py`, fonction `publication_of`), avec quelques corrections manuelles : toutes
  les variantes `EvansLetter…` forment une seule source ; les identifiants `pNN-RR8201`
  (23 pages, 46 lignes) viennent d'un même rapport ; `KocakPEv` est une faute de frappe pour `KocakPREv` ;
  `Ga&K-1975II` est la partie II de `Gar&K-1975`. Dans le doute, on regroupe : c'est le choix
  le plus prudent pour éviter les fuites d'information à la validation (§5).

### 1.2 Valeurs manquantes

Le taux de remplissage est très variable :

| Groupe | Variables | Remplissage |
|---|---|---|
| Composition, toujours mesurée | C, Si, Mn, S, P | ~100 % |
| Procédé | apport de chaleur, température entre passes, type de soudure, traitement thermique | ~100 % |
| Procédé | courant, tension, AC/DC, polarité | 85 à 90 % |
| Éléments d'alliage | Ni, Cr, Mo, V, Cu | 35 à 56 % |
| Éléments en ppm | O, N | ~75 % ; Ti, Al, Nb, B : 30 à 57 % |
| Presque vides | Co, W, Sn, As, Sb | < 20 % |
| Propriétés mécaniques | limite d'élasticité, résistance à la traction, allongement, Charpy | 42 à 53 % |
| Presque vides | dureté, FATT, microstructure | < 10 % |

Point clé : **les éléments d'alliage manquent par blocs de publications**. Un auteur qui
étudie des soudures au carbone-manganèse ne mentionne pas le nickel ou le chrome, parce qu'il
n'en a pas ajouté. Cette observation justifie l'imputation par 0 (§4).

### 1.3 Unités et échelles

- Composition en **% massique** (C, Si, Mn, S, P, Ni, Cr, Mo, V, Cu, Co, W) et en **ppm
  massique** (O, Ti, N, Al, B, Nb, Sn, As, Sb).
- Procédé en unités physiques : A, V, kJ/mm, °C, heures.
- Les valeurs vont de 10⁻³ à 10³ : **une standardisation est indispensable** pour l'ACP et
  pour les modèles fondés sur des distances ou des produits scalaires (k-NN, SVR). Elle est
  inutile, mais sans effet néfaste, pour les modèles d'arbres (forêt aléatoire, gradient
  boosting).
- Les éléments d'alliage ont des distributions **très asymétriques** : la plupart des
  soudures sont au carbone-manganèse, quelques-unes sont fortement alliées (jusqu'à 10 % de
  Cr).

### 1.4 ACP : développer une intuition

L'ACP porte sur 20 variables d'entrée assez renseignées, après correction des erreurs d'unité
(§2), imputation (0 pour les éléments d'alliage, médiane sinon) et standardisation.

| Composante | Variables dominantes | Interprétation |
|---|---|---|
| PC1 | Cr, Mo, V, N, température du traitement thermique | **Niveau d'alliage et traitement thermique** : aciers alliés traités thermiquement d'un côté, soudures C-Mn brutes de l'autre |
| PC2 | courant, tension, apport de chaleur | **Énergie de soudage** : sépare l'arc submergé (SA, TSA), très énergétique, de l'arc manuel (MMA) |
| PC3 | S, P (opposés à C) | **Impuretés** |

Ce que l'ACP apprend :
- **10 composantes sont nécessaires pour 80 % de la variance** : les entrées sont peu
  redondantes, l'ACP ne réduira pas utilement la dimension. On ne l'utilise donc pas comme
  pré-traitement des modèles.
- **La limite d'élasticité augmente le long de PC1** : les éléments d'alliage seront des
  variables importantes.
- **Les publications forment des groupes compacts** (ex. Pat-1981, Gar&K-1975,
  Evans-AlTi-1994) : chaque étude explore une petite région de l'espace des recettes. C'est
  la première indication qu'il faudra une validation croisée groupée par publication (§5).

---

## 2. Nettoyage des données

[`src/welddb.py` pour le format, `src/preprocessing.py` pour le nettoyage]

### 2.1 Conversion du format brut

| Cas | Exemple | Traitement | Justification |
|---|---|---|---|
| Valeur non publiée | `N` | manquante | Indiqué dans `welddb.info` : ce n'est **pas** zéro |
| Sous la limite de détection | `<0.01`, `<5` | valeur de la limite, plus une colonne indicatrice `<col>_below_dl` | La vraie valeur est entre 0 et la limite ; la limite est une borne supérieure prudente et très proche de 0 |
| Plage | `150-200` (température entre passes) | milieu de la plage | Meilleure estimation ponctuelle |
| Dureté annotée | `203(Hv30)`, `224Hv10` | valeur numérique | Toutes en Vickers ; la charge d'essai change peu la valeur |
| Azote total / résiduel | `66tot33res` | azote total | Cohérent avec les autres lignes, qui donnent l'azote total |
| Polarité `0` | seulement en courant alternatif | `AC` | En AC, la polarité n'a pas de sens |
| Types de soudure synonymes | `ShMA`, `SAA` | `MMA`, `SA` | `welddb.info` : ShMA = MMA ; SAA = arc submergé automatique |

### 2.2 Erreurs d'unité corrigées

| Problème | Lignes | Traitement | Comment on l'a repéré |
|---|---|---|---|
| Valeurs inférieures à 1 dans les colonnes en **ppm** (ex. Al = 0,004) | 313 valeurs | → manquantes | Une teneur de 0,004 ppm n'a pas de sens physique ; ce sont très probablement des % massiques (0,004 % = 40 ppm), mais on ne peut pas en être sûr, donc on les traite comme manquantes |
| **Vanadium `<5`** dans une colonne en % massique | 9 lignes (EvansLetter) | → 0,0005 % | Ailleurs, V ne dépasse pas 0,32 % ; il s'agit d'une limite de 5 **ppm**. Non corrigée, cette valeur faisait prédire 4 699 MPa à la régression linéaire |

### 2.3 Variables écartées

- **Co, W, Sn, As, Sb** : renseignés sur moins de 20 % des lignes étiquetées. Les imputer
  reviendrait à inventer l'essentiel de la colonne.
- **Dureté, FATT, microstructure** : trop rares (moins de 140 lignes) pour servir d'entrées
  ou de cibles. La microstructure aurait été très utile pour la ténacité (§3.2), mais elle
  n'est presque jamais renseignée.
- **Résistance à la traction, allongement, striction** : non utilisés comme entrées, car ce
  sont des propriétés mesurées après soudage, pas des éléments de la recette.

### 2.4 Pas de fusion des lignes identiques

Fusionner les lignes de même recette aurait évité de compter plusieurs fois une même soudure.
On a choisi de **ne pas le faire**, pour garder une méthodologie simple à expliquer : la
validation groupée par publication (§5) place de toute façon toutes les lignes d'une même
soudure du même côté du découpage, ce qui supprime le risque de fuite.

---

## 3. Définir la qualité d'une soudure

[`02_variables_qualite_strategie.ipynb`]

### 3.1 Les variables candidates

| Variable | Ce qu'elle mesure | Lignes |
|---|---|---|
| Limite d'élasticité (MPa) | contrainte à partir de laquelle la déformation devient permanente | 780 |
| Résistance à la traction (MPa) | contrainte maximale avant rupture | 738 |
| Allongement, striction (%) | ductilité | ~700 |
| Résilience Charpy (J) | énergie absorbée lors d'un choc, à une température donnée | 879 |

Une bonne soudure doit être **résistante** (ne pas se déformer sous la charge) **et tenace**
(ne pas casser brutalement sous un choc, en particulier au froid).

### 3.2 Le choix : deux facettes, deux modèles de régression

1. **Résistance → limite d'élasticité.** C'est la grandeur utilisée pour dimensionner les
   structures. La résistance à la traction lui est corrélée à **0,92** : la prédire en plus
   serait redondant.
2. **Ténacité → résilience Charpy, avec la température d'essai en entrée.** L'énergie
   absorbée dépend énormément de la température (transition ductile-fragile) : 50 J à 20 °C
   et 50 J à −60 °C ne décrivent pas du tout la même soudure. Avec la température en entrée,
   le modèle apprend toute la courbe de transition, et on peut lui demander « combien de
   joules à −40 °C ? ».

**Pourquoi deux modèles séparés plutôt qu'une étiquette « bonne / mauvaise »** : seules
**144 lignes** ont à la fois une limite d'élasticité et un essai Charpy ; une étiquette
combinée réduirait fortement les données. Elle demanderait aussi des seuils arbitraires et
ferait perdre de l'information (401 MPa et 900 MPa deviendraient « bon » tous les deux).

**Pourquoi la régression** : on garde l'information fine, et on peut toujours en déduire une
décision conforme / non conforme en comparant la prédiction à un seuil (§6).

### 3.3 Particularité des données Charpy

[`04_charpy.ipynb`]

**474 des 879 lignes Charpy** valent exactement **100 J** ou **28 J**, presque toutes issues
des travaux d'Evans. Dans ces lignes, l'énergie est fixée et c'est la **température** qui a
été mesurée : « à quelle température la soudure absorbe-t-elle 100 J ? ». Ce sont des points
valides de la courbe de transition, donc on les garde. On verra qu'elles biaisent la forêt
aléatoire (voir [résultats](resultats.md#2-résilience-charpy)).

### 3.4 Variables qui influencent la limite d'élasticité

Deux méthodes complémentaires :
- **corrélation de Spearman** : simple, mais une variable à la fois et seulement pour les
  relations monotones ;
- **importance par permutation** d'un gradient boosting, calculée sur des publications de
  test : on mélange une colonne et on mesure la hausse de l'erreur. Cette méthode tient
  compte des interactions.

Résultat : **Cr, Mo et Mn** dominent, puis V, Si et la température du traitement thermique.
C'est cohérent avec la métallurgie (durcissement par les éléments d'alliage). On garde
néanmoins toutes les variables assez renseignées : les variables corrélées se partagent
l'importance, et une partie de l'effet peut être un effet publication.

### 3.5 Le traitement thermique recouvre deux opérations

Les températures de traitement de 200–250 °C correspondent à un **dégazage de l'hydrogène**
(les lignes « brutes de soudage » d'Evans sont à 250 °C pendant 14 h). Au-dessus de 400 °C
(en pratique 550–760 °C), il s'agit d'un **revenu de détensionnement**, qui modifie la
microstructure. La température est gardée comme variable numérique : les modèles non
linéaires peuvent apprendre cet effet de seuil.

---

## 4. Pré-traitement

[`src/preprocessing.py`]

| Étape | Traitement | Justification |
|---|---|---|
| Ni, Cr, Mo, V, Cu, Ti, Al, B, Nb manquants | **→ 0** | Éléments ajoutés volontairement : non mentionné = non ajouté (§1.2). La teneur résiduelle réelle est faible |
| C, Si, Mn, S, P, O, N, courant, tension, apport de chaleur, températures, durée manquants | **→ médiane** | Ces éléments sont toujours présents dans une soudure (O, N) ou ces paramètres toujours fixés : 0 n'aurait pas de sens. La médiane est robuste aux valeurs extrêmes |
| Type de soudure, AC/DC, polarité | **one-hot** ; manquant → catégorie « inconnu » ; catégories de moins de 10 lignes regroupées | Les modèles demandent des nombres. Trop peu de lignes pour apprendre une catégorie rare (GTAA, GMAA : 4 lignes chacune) |
| Toutes les variables numériques | **standardisation** (moyenne 0, écart-type 1) | Échelles de 10⁻³ à 10³ (§1.3). Sans elle, la distance du k-NN et le noyau du SVR seraient dominés par les variables en ppm ou en ampères |
| Cible | standardisée pendant l'entraînement (`TransformedTargetRegressor`), prédictions remises dans l'unité d'origine | Nécessaire pour le SVR, dont les hyperparamètres C et ε dépendent de l'échelle de la cible ; sans effet sur les autres modèles |

**Pas de fuite d'information** : le pré-traitement est la première étape d'un `Pipeline`
scikit-learn. La médiane et la standardisation sont donc recalculées sur les seules données
d'entraînement de chaque pli.

---

## 5. Protocole de validation

[`02_variables_qualite_strategie.ipynb` pour la justification, `03` et `04` pour la mise en œuvre]

### 5.1 Pourquoi grouper par publication

La question posée est : *quelle sera la qualité d'une **nouvelle** soudure ?* Or les lignes
d'une même publication se ressemblent beaucoup (même laboratoire, même protocole, souvent la
même soudure mesurée plusieurs fois). Deux constats :

- prédire simplement la moyenne de la publication explique déjà **47 %** de la variance de
  la limite d'élasticité (**44 %** en R² ajusté, qui tient compte des 42 moyennes
  estimées) ;
- avec le même modèle (gradient boosting) :

| Découpage de la validation croisée | RMSE |
|---|---|
| Aléatoire (`KFold`) | **36 MPa** |
| Groupé par publication (`GroupKFold`) | **70 MPa** |
| Référence : prédire la moyenne | 93 MPa |

Avec un découpage aléatoire, le modèle voit en entraînement des soudures jumelles de celles
du test : son erreur est **sous-estimée de moitié**. On utilise donc partout `GroupKFold`
par publication : toutes les lignes d'une publication sont soit dans l'entraînement, soit
dans le test.

### 5.2 Validation croisée imbriquée

Les hyperparamètres (nombre de voisins, C du SVR…) sont choisis **sans jamais regarder le
pli de test** :

```
Pour chaque pli externe (5 plis groupés par publication) :
    entraînement (4/5 des publications) ──► GridSearchCV avec GroupKFold(4) ──► meilleurs hyperparamètres
    test (1/5 des publications)          ──► évaluation du modèle retenu
```

Si on choisissait les hyperparamètres d'après le score de test, ce score serait optimiste.

### 5.3 Comparaison statistique

[`05_comparaison_recommandations.ipynb`, section 4]

Avec seulement 5 plis, comparer des moyennes n'est pas concluant : l'écart-type entre plis
est donné à titre indicatif. Les modèles sont comparés par deux méthodes complémentaires,
toutes deux fondées sur les **publications**, l'unité réellement indépendante :

- **Test de Wilcoxon apparié** sur le RMSE calculé publication par publication
  (42 publications pour la limite d'élasticité, 26 pour le Charpy) : non paramétrique,
  adapté aux petits échantillons, sans hypothèse de normalité. Chaque publication pèse
  autant, quelle que soit sa taille. Six comparaisons sont faites : les p-valeurs sont
  corrigées par la **méthode de Holm**, sinon le risque de trouver un écart « significatif »
  par hasard augmente avec le nombre de tests.
- **Intervalle de confiance à 95 % par bootstrap sur les publications** : on tire 2 000
  fois, avec remise, autant de publications qu'il y en a, et on recalcule le RMSE sur toutes
  leurs lignes. Tirer des publications plutôt que des lignes respecte la dépendance entre
  lignes d'une même publication. Pour comparer deux modèles, on calcule l'écart de RMSE sur
  les mêmes tirages ; il est significatif si son intervalle ne contient pas 0. Ici, chaque
  ligne pèse autant.

Les deux méthodes peuvent diverger quand une grande publication pèse sur le résultat : on
les présente donc ensemble.

---

## 6. Métriques

Notations : yᵢ valeur mesurée, ŷᵢ valeur prédite, ȳ moyenne, n nombre de soudures de test.

| Métrique | Formule | Pourquoi |
|---|---|---|
| **RMSE** (principale) | √( (1/n) Σ (yᵢ − ŷᵢ)² ) | Dans l'unité physique (MPa, J) ; pénalise fortement les grosses erreurs, les plus dangereuses pour dimensionner une structure |
| **MAE** | (1/n) Σ \|yᵢ − ŷᵢ\| | Erreur « typique », moins sensible aux quelques soudures très mal prédites |
| **R²** | 1 − Σ (yᵢ − ŷᵢ)² / Σ (yᵢ − ȳ)² | Sans unité : permet de comparer les deux cibles. 0 = pas mieux que la moyenne, négatif = pire |
| **Part dans la tolérance** | part des soudures avec \|yᵢ − ŷᵢ\| ≤ 50 MPa (ou 20 J) | Lecture « métier » simple |
| **Taux de fausse acceptation** | non conformes prédites conformes / non conformes | **L'erreur dangereuse** : accepter une soudure trop faible |
| **Taux de faux rejet** | conformes prédites non conformes / conformes | L'erreur coûteuse : refaire ou retester une bonne soudure |
| **AUC** (aire sous la courbe ROC) | probabilité qu'une soudure non conforme tirée au hasard ait une prédiction plus basse qu'une soudure conforme tirée au hasard | Capacité à **classer** les soudures, indépendamment du seuil et de la marge : 0,5 = hasard, 1 = parfait |

RMSE, MAE et R² sont calculés pli par pli puis moyennés ; les autres métriques portent sur
l'ensemble des prédictions hors pli.

**Seuils de conformité** utilisés pour les trois dernières métriques : 420 MPa de limite
d'élasticité (classe « 42 » de la norme EN ISO 2560-A) et 47 J à la température d'essai. Ils
servent à illustrer ; un industriel utiliserait ceux de son cahier des charges.

**Marge de sécurité.** Les modèles tirent vers la moyenne et acceptent trop de soudures
faibles. On n'accepte donc une soudure que si sa prédiction dépasse le seuil d'une certaine
marge. Cette marge est un réglage : la choisir en regardant les taux mesurés reviendrait à
la régler sur les données d'évaluation, et donnerait des taux optimistes (pour le SVR sur
la limite d'élasticité : 10 % de fausse acceptation annoncés, 16 % en réalité). Elle est
donc choisie **en validation croisée groupée par publication** : pour chaque groupe de
publications, on retient la plus petite marge donnant au plus 10 % de fausse acceptation sur
les autres publications, puis on mesure les taux sur ce groupe.

**Pourquoi plusieurs métriques** : elles ne racontent pas la même histoire. Par exemple, pour
le Charpy, la simple moyenne a la meilleure « part dans la tolérance », parce que beaucoup
de lignes valent 100 J et que la moyenne (≈ 88 J) tombe près d'elles. Seul le croisement des
métriques permet de conclure.

**Références** : en plus de la moyenne, une forêt aléatoire qui ne connaît **que la
température d'essai** sert de référence pour le Charpy. Elle mesure ce que la recette apporte
réellement.

---

## 7. Modèles

On compare les quatre méthodes vues en cours et le gradient boosting d'arbres, la méthode
d'ensemble de référence sur les données tabulaires (déjà utilisée au notebook 02 pour
l'importance des variables).

| Modèle | Principe | Hyperparamètres testés | Attendu |
|---|---|---|---|
| **Régression linéaire** | ŷ = w₀ + Σ wⱼ xⱼ, moindres carrés | aucun | Référence interprétable ; suppose des effets additifs et constants |
| **k plus proches voisins** | moyenne de la cible des k soudures les plus proches | k ∈ {5, 10, 20, 40, 80} ; pondération uniforme ou par la distance | Non linéaire, mais sensible aux variables inutiles |
| **SVR, noyau gaussien** | régression à vaste marge ; ignore les erreurs inférieures à ε ; non linéaire grâce au noyau | C ∈ {0,1 ; 1 ; 10 ; 100}, γ ∈ {`scale` ; 0,01 ; 0,1}, ε ∈ {0,05 ; 0,2} | Fonction lisse, robuste au bruit |
| **Forêt aléatoire** | moyenne de 300 arbres appris sur des échantillons bootstrap, avec un tirage aléatoire de variables à chaque nœud | part des variables par nœud ∈ {0,33 ; 0,66 ; 1}, taille minimale des feuilles ∈ {1, 3, 5} | Capte les interactions, ne prédit jamais hors de la plage vue |
| **Gradient boosting** (GBRT) | somme de 300 petits arbres appris l'un après l'autre, chacun corrigeant les résidus des précédents ; contribution réduite par le taux d'apprentissage ; chaque arbre appris sur 80 % des lignes | taux d'apprentissage ∈ {0,03 ; 0,1}, profondeur maximale ∈ {2, 3, 5} | Capte les interactions ; plus précis que la forêt en général, mais peut sortir de la plage vue (somme de corrections) |

Pour la limite d'élasticité, la validation retient toujours des arbres de profondeur 2 avec
un taux de 0,1, deux valeurs au bord de la grille : une grille plus large pourrait encore
améliorer légèrement le gradient boosting.

---

## 8. Apprentissage semi-supervisé

[`03_modeles_limite_elasticite.ipynb`, sections 5 et 6]

### 8.1 Motivation et démarche

872 lignes sur 1 652 n'ont pas de limite d'élasticité, mais leur recette est connue. Le
semi-supervisé cherche à exploiter ces entrées non étiquetées. La démarche se fait en deux
temps : une méthode simple d'abord (self-training), puis, après avoir compris son échec,
une méthode plus avancée choisie pour répondre à cet échec (régularisation de variété).

### 8.2 Méthodes envisagées

| Famille | Principe | Adaptée ici ? |
|---|---|---|
| **Self-training** [Yarowsky, 1995 ; Triguero et al., 2015] | Le modèle prédit des pseudo-étiquettes, on ajoute les plus fiables à l'entraînement, on recommence | **Oui** : fonctionne avec n'importe quel régresseur ; retenue en premier |
| **Régularisation de variété** (Laplacian RLS) [Belkin et al., 2006] | Régression à noyau pénalisée pour varier doucement le long d'un graphe reliant toutes les recettes | **Oui** : retenue en second (§8.4) |
| Co-training / COREG [Zhou & Li, 2005] | Deux k-NN s'étiquettent mutuellement | Possible, mais repose sur le k-NN, notre modèle le plus faible |
| Co-training à deux vues [Blum & Mitchell, 1998] | Deux modèles sur deux groupes de variables (composition / procédé) | Les deux vues ne sont ni suffisantes ni indépendantes : le procédé seul prédit mal |
| Propagation de labels [Zhu & Ghahramani, 2002] | Diffusion des étiquettes dans un graphe de similarité | Cas particulier transductif de la régularisation de variété |
| Méthodes profondes (Mean Teacher, MixMatch…) | Régularisation par cohérence de réseaux de neurones | Hors du cadre du cours, peu de données tabulaires |

### 8.3 Premier choix : self-training avec forêt aléatoire

- **La forêt aléatoire est, avec le SVR, l'un des meilleurs modèles du cours.** Le gradient
  boosting, un peu meilleur, ne convient pas ici : ses arbres sont appris les uns à partir
  des autres, et l'écart entre leurs prédictions ne mesure pas une confiance.
- **Elle fournit naturellement une mesure de confiance** : l'écart-type des prédictions de
  ses 300 arbres. Des arbres d'accord indiquent une pseudo-étiquette probablement fiable.
- **Elle est simple à expliquer et à reproduire.**

Algorithme, sur 5 itérations : entraîner la forêt, prédire les lignes non étiquetées, ajouter
à l'entraînement les 20 % de prédictions où les arbres sont le plus d'accord, recommencer.

**Évaluation honnête** : même validation groupée par publication ; les lignes non étiquetées
des publications de test sont exclues. Pour vérifier si le semi-supervisé aide quand les
étiquettes sont rares, l'expérience est répétée avec 100 %, 50 %, 25 % et 10 % des
étiquettes d'entraînement. Les étiquettes retirées rejoignent alors les données non
étiquetées.

**Reproductibilité** : les pseudo-étiquettes et la confiance sont calculées arbre par arbre
dans un ordre fixe, et le tri utilise un algorithme stable. Sinon, les calculs en parallèle
introduisent d'infimes écarts numériques qui changent les résultats d'une exécution à l'autre.

### 8.4 Second choix : régularisation de variété

**Ce que l'échec du self-training a appris.** Le self-training ajoute des pseudo-étiquettes
là où le modèle est déjà sûr de lui. Or, sur les 872 lignes non étiquetées, seules 262
recettes sont absentes des lignes étiquetées, et elles en sont nettement plus loin que les
lignes étiquetées entre elles (distance médiane au plus proche voisin étiqueté : 2,1 contre
0,8, entrées standardisées). La difficulté principale est le **changement de domaine** entre
publications.

**Pourquoi cette méthode.** On cherche f dans l'espace du noyau gaussien K qui minimise :

  (1/l) Σ_{i étiquetés} (yᵢ − f(xᵢ))² + λ_A ‖f‖²_K + (λ_I / n) fᵀ L f

où L est le laplacien normalisé du graphe des 10 plus proches voisins construit sur les
n recettes, étiquetées ou non. Le dernier terme impose que des recettes voisines aient des
prédictions proches (**hypothèse de variété**). La solution est un simple système linéaire.
- elle est **conçue pour la régression** et appartient à la même famille que le SVR ;
- elle utilise la **géométrie** des recettes non étiquetées, sans pseudo-étiquettes qui
  renforcent les biais du modèle ;
- avec **λ_I = 0**, elle redevient une **régression ridge à noyau supervisée** : ce témoin
  isole exactement l'effet des données non étiquetées ;
- elle existe en **version transductive** : on ajoute au graphe les recettes des
  publications de test, sans leurs mesures. C'est réaliste, car les recettes d'une nouvelle
  campagne sont connues avant les essais, et cette variante vise directement le changement de
  domaine.

**Évaluation.** Même validation croisée imbriquée groupée par publication. Hyperparamètres :
γ ∈ {0,01 ; 0,03}, λ_A ∈ {10⁻⁵ ; 10⁻⁴ ; 10⁻³}, λ_I ∈ {0,01 ; 0,1 ; 1}. Les lignes non
étiquetées des publications de validation et de test sont exclues à tous les niveaux (ou,
en transductif, seules leurs recettes sont utilisées, jamais leurs mesures). L'expérience
est répétée avec 50 %, 25 % et 10 % des étiquettes, avec γ et λ_A fixés et λ_I variable.
Enfin, on vérifie directement l'hypothèse de variété : écart de limite d'élasticité entre
recettes voisines dans le graphe, comparé à deux soudures tirées au hasard.

### 8.5 Pourquoi ne pas l'appliquer au Charpy

La température d'essai est une entrée du modèle Charpy, et les lignes sans essai Charpy n'en
ont pas. Il faudrait leur inventer une température : les pseudo-étiquettes (ou les points du
graphe) n'auraient alors aucun lien avec une mesure réelle.

---

## 9. Interprétation et recommandations

[`05_comparaison_recommandations.ipynb`, section 6]

- **Modèles finaux** : SVR entraîné sur toutes les données étiquetées, hyperparamètres choisis
  par validation croisée groupée ; forêt aléatoire comme modèle de contrôle. La forêt est
  préférée au gradient boosting pour ce rôle : elle ne prédit jamais hors de la plage vue,
  ce qui en fait un contrôle prudent.
- **Dépendance partielle** : on fixe une variable à une valeur pour toutes les soudures, on
  prédit, et on fait la moyenne. On fait varier la valeur entre les **percentiles 5 et 95**
  observés, pour ne pas extrapoler.
- Pour le Charpy, la température d'essai est fixée à **−40 °C**, une température de service
  courante pour des structures en extérieur.
- **Une recommandation est jugée solide** si le SVR et la forêt aléatoire trouvent un effet
  dans le même sens et si cet effet est cohérent avec la métallurgie du soudage.

---

## 10. Choix envisagés puis écartés

| Option | Raison de l'écarter |
|---|---|
| Étiquette « bonne / mauvaise soudure » (classification) | 144 lignes seulement, seuils arbitraires, perte d'information (§3.2) |
| Prédire aussi la résistance à la traction | Redondante avec la limite d'élasticité (r = 0,92) |
| Fusionner les lignes de même recette | Complexifie la méthodologie ; le risque de fuite est déjà traité par la validation groupée (§2.4) |
| Validation croisée aléatoire | Sous-estime l'erreur de moitié (§5.1) |
| ACP comme pré-traitement | 10 composantes pour 80 % de variance : peu de gain, perte d'interprétabilité (§1.4) |
| Prédire la température de transition (100 J / 28 J) au lieu de l'énergie | N'utiliserait que 474 lignes et 18 publications. La forêt aléatoire y fait un peu mieux (R² ≈ 0,36), mais le SVR y échoue (R² négatif). Ces chiffres viennent d'un essai exploratoire hors notebook |
| Retirer les lignes Charpy à « énergie fixée » | Ne reste que 13 publications ; pas de meilleur résultat (R² ≈ 0,1) [`04_charpy.ipynb`] |
| COREG (co-training de deux k-NN) | Testé hors notebook : gain inférieur à 1 MPa sur une base k-NN déjà faible, et 5 minutes de calcul. Le test date d'avant la correction du regroupement des publications (§1.1) |
| Lire la marge de sécurité directement sur les taux mesurés | Règle la marge sur les données d'évaluation : taux trop optimistes (§6) |
| Régression ridge à noyau dans la comparaison principale | Apparue comme témoin de la régularisation de variété (environ 64 MPa, aussi bien que le gradient boosting) ; non ajoutée pour garder la comparaison centrée sur les méthodes du cours et le gradient boosting |

---

## Références

- Cool, T., Bhadeshia, H. K. D. H., & MacKay, D. J. C. (1997). The yield and ultimate tensile strength of steel welds. *Materials Science and Engineering A*, 223.
- Yarowsky, D. (1995). Unsupervised word sense disambiguation rivaling supervised methods. *ACL*.
- Zhu, X., & Ghahramani, Z. (2002). Learning from labeled and unlabeled data with label propagation. Technical report CMU-CALD-02-107.
- Zhou, Z.-H., & Li, M. (2005). Semi-supervised regression with co-training. *IJCAI*.
- Blum, A., & Mitchell, T. (1998). Combining labeled and unlabeled data with co-training. *COLT*.
- Belkin, M., Niyogi, P., & Sindhwani, V. (2006). Manifold regularization: A geometric framework for learning from labeled and unlabeled examples. *Journal of Machine Learning Research*, 7.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2).
- Triguero, I., García, S., & Herrera, F. (2015). Self-labeled techniques for semi-supervised learning: taxonomy, software and empirical study. *Knowledge and Information Systems*, 42(2).
- van Engelen, J. E., & Hoos, H. H. (2020). A survey on semi-supervised learning. *Machine Learning*, 109(2).
- NF EN ISO 2560 : Produits consommables pour le soudage — Électrodes enrobées pour le soudage manuel à l'arc des aciers non alliés et à grains fins — Classification.
