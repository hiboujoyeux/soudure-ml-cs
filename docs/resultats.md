# Résultats, recommandations et limites

Tous les chiffres sont mesurés sur des **publications jamais vues** pendant l'entraînement
(validation croisée imbriquée groupée par publication, voir
[méthodologie §5](methodologie.md#5-protocole-de-validation)). Les écarts-types sont
calculés entre les 5 plis.

Sommaire :
1. [Limite d'élasticité](#1-limite-délasticité)
2. [Résilience Charpy](#2-résilience-charpy)
3. [Semi-supervisé](#3-semi-supervisé)
4. [Comparaison globale et choix de l'approche](#4-comparaison-globale-et-choix-de-lapproche)
5. [Ce que le modèle a appris](#5-ce-que-le-modèle-a-appris)
6. [Recommandations](#6-recommandations)
7. [Limites et pistes](#7-limites-et-pistes)

---

## 1. Limite d'élasticité

[`03_modeles_limite_elasticite.ipynb`] — 780 soudures, 42 publications.

| Modèle | RMSE (MPa) | MAE (MPa) | R² |
|---|---|---|---|
| Moyenne (référence) | 92,6 ± 16,1 | 72,1 | −0,06 |
| Régression linéaire | 105,3 ± 22,1 | 80,5 | −0,41 |
| k plus proches voisins | 88,1 ± 25,9 | 66,4 | 0,05 |
| **SVR** | **66,8 ± 11,8** | **48,8** | **0,45** |
| **Forêt aléatoire** | **67,5 ± 11,7** | 49,7 | 0,44 ± 0,05 |

- **SVR et forêt aléatoire sont à égalité** : la forêt gagne 3 plis sur 5, le SVR 2, avec des
  écarts de 1 à 9 MPa dans les deux sens. Le R² de la forêt varie le moins d'un pli à l'autre.
- **La régression linéaire fait pire que la moyenne.** Ses plus grosses erreurs portent sur des
  aciers au chrome (2 à 9 % de Cr), rares dans la base, avec des prédictions physiquement
  absurdes (61 MPa pour une soudure mesurée à 392 MPa). Elle suppose que chaque élément a un
  effet additif et constant, alors qu'en métallurgie l'effet d'un élément dépend du
  traitement thermique et des autres éléments. Elle extrapole alors hors du domaine vu.
- **Le k-NN fait à peine mieux que la moyenne** : toutes les variables standardisées comptent
  autant dans la distance, y compris celles qui n'influencent pas la cible (fléau de la
  dimension). La validation choisit d'ailleurs souvent k = 80, ce qui revient presque à une
  moyenne.
- **Les soudures très résistantes (plus de 700 MPa) sont sous-estimées** : elles sont rares
  et concentrées dans quelques publications.

## 2. Résilience Charpy

[`04_charpy.ipynb`] — 879 essais, 26 publications. Entrées : la recette et la température
d'essai.

| Modèle | RMSE (J) | MAE (J) | R² |
|---|---|---|---|
| Moyenne (référence) | 51,2 ± 4,3 | 39,9 | −0,18 |
| Température d'essai seule (référence) | 45,1 ± 7,6 | 33,2 | 0,03 |
| Régression linéaire | 67,2 ± 35,8 | 54,2 | −1,35 |
| k plus proches voisins | 49,1 ± 4,7 | 37,7 | −0,09 |
| **SVR** | **39,7 ± 8,1** | **30,5** | **0,28** |
| Forêt aléatoire | 45,1 ± 10,3 | 34,2 | 0,07 |

- **Le SVR est le seul modèle nettement meilleur que la référence « température seule ».**
- **La forêt aléatoire est biaisée par les lignes « énergie fixée »** (100 J ou 28 J, voir
  [méthodologie §3.3](methodologie.md#33-particularité-des-données-charpy)) : elle prédit un
  plateau à 100 J pour les soudures d'Evans, quelle que soit la température. Pour ces
  recettes, l'entraînement contient surtout des lignes « 100 J », chacune à la température
  de transition de sa soudure. Le SVR, plus lisse, résiste mieux à ce biais.
- **Les courbes de transition prédites sont trop plates** : les vraies courbes passent de
  ~10 J à plus de 200 J en quelques dizaines de degrés ; les prédictions restent entre 50 et
  150 J.
- **Retirer les lignes « énergie fixée » n'aide pas** : il ne reste que 13 publications, et
  le R² tombe à environ 0,1 pour le SVR comme pour la forêt.
- **Pourquoi la ténacité est plus difficile à prédire** : elle dépend fortement de la
  microstructure (ferrite aciculaire, inclusions), presque jamais renseignée ; la transition
  ductile-fragile est très abrupte ; et peu de publications fournissent des courbes
  complètes.

## 3. Semi-supervisé

[`03_modeles_limite_elasticite.ipynb`, section 5] — self-training avec forêt aléatoire,
limite d'élasticité.

| Part des étiquettes d'entraînement | Supervisé (RMSE, MPa) | Self-training (RMSE, MPa) | Écart |
|---|---|---|---|
| 100 % | 66,3 | 67,5 | −1,3 |
| 50 % | 68,7 | 70,1 | −1,4 |
| 25 % | 74,2 | 75,9 | −1,7 |
| 10 % | 77,7 | 79,8 | −2,1 |

**Le self-training est en moyenne légèrement moins bon que le supervisé**, quelle que soit
la quantité d'étiquettes. Trois raisons :

1. **Les données non étiquetées apportent peu d'information nouvelle** : sur les 872 lignes
   sans limite d'élasticité, 489 ont exactement la même recette qu'une ligne étiquetée (la
   même soudure, mesurée en Charpy).
2. **La difficulté principale n'est pas le manque d'étiquettes, mais le changement de domaine
   entre publications.** Le self-training ajoute des pseudo-étiquettes là où la forêt est déjà
   sûre d'elle, c'est-à-dire dans les zones bien couvertes, pas là où elle se trompe.
3. **Le modèle renforce ses propres biais** (*confirmation bias*) : réentraîné sur ses propres
   prédictions, il tend vers les mêmes prédictions.

La méthode est pertinente sur le papier, mais les conditions qui la font fonctionner (des
données non étiquetées nombreuses, nouvelles et proches des zones d'incertitude) ne sont pas
réunies ici.

## 4. Comparaison globale et choix de l'approche

[`05_comparaison_recommandations.ipynb`]

### 4.1 Part des prédictions dans la tolérance

| Modèle | Limite d'élasticité (± 50 MPa) | Charpy (± 20 J) |
|---|---|---|
| Moyenne | 43 % | 47 % |
| Régression linéaire | 44 % | 32 % |
| k-NN | 50 % | 42 % |
| **SVR** | **64 %** | 42 % |
| Forêt aléatoire | 61 % | 40 % |

Pour le Charpy, la moyenne obtient le meilleur score, simplement parce que beaucoup de lignes
valent 100 J et que la moyenne (≈ 88 J) tombe près d'elles. C'est un exemple de métrique
trompeuse prise isolément.

### 4.2 Métrique de décision : accepter ou refuser une soudure

Seuils : 420 MPa (16 % des soudures sont en dessous) et 47 J à la température d'essai (28 %
en dessous).

| Modèle | Fausse acceptation, Re | Faux rejet, Re | Fausse acceptation, Charpy | Faux rejet, Charpy |
|---|---|---|---|---|
| Régression linéaire | 63 % | 11 % | 77 % | 7 % |
| k-NN | 74 % | 2 % | 100 % | 0 % |
| **SVR** | **45 %** | 4 % | **64 %** | 3 % |
| Forêt aléatoire | 70 % | 1 % | 96 % | 2 % |

- **Les modèles tirent vers la moyenne** et prédisent rarement des valeurs basses : sans
  précaution, ils acceptent une grande partie des soudures trop faibles.
- **Avec une marge de sécurité**, en n'acceptant que si la prédiction dépasse le seuil d'une
  certaine marge, le SVR descend sous 10 % de fausse acceptation avec une marge d'environ
  **90 MPa** (au prix de 58 % de faux rejets) pour la limite d'élasticité, et d'environ
  **45 J** (46 % de faux rejets) pour le Charpy.
- **Conclusion pratique** : le modèle peut **présélectionner** des recettes, mais **pas
  remplacer les essais mécaniques**.

### 4.3 Les écarts sont-ils significatifs ?

Test de Wilcoxon apparié sur le RMSE calculé publication par publication :

| Comparaison | Publications | SVR meilleur | Écart médian | p-valeur |
|---|---|---|---|---|
| Re : SVR vs forêt aléatoire | 42 | 57 % | −3,2 MPa | 0,42 |
| Charpy : SVR vs forêt aléatoire | 26 | 58 % | −1,2 J | 0,45 |
| Charpy : SVR vs température seule | 26 | 73 % | −3,7 J | 0,14 |

**Aucun écart n'est significatif au seuil de 5 %.** Le SVR est meilleur sur un peu plus de la
moitié des publications, mais avec si peu de publications indépendantes, on ne peut pas
exclure le hasard.

### 4.4 Approche retenue

**Un SVR à noyau gaussien par propriété**, avec le pré-traitement décrit dans la
méthodologie et une validation croisée groupée par publication. Hyperparamètres finaux :
C = 10, γ = 0,01, ε = 0,2 pour la limite d'élasticité ; C = 1, γ = 0,01, ε = 0,05 pour le
Charpy.

Justification :
- au moins aussi bon que la forêt aléatoire pour la limite d'élasticité ;
- meilleur pour la ténacité ;
- le plus faible taux de fausse acceptation sur les deux cibles ;
- effets lisses et faciles à interpréter.

Comme l'écart n'est pas significatif, la forêt aléatoire est gardée comme **modèle de
contrôle** pour l'interprétation. Le semi-supervisé n'est pas retenu.

## 5. Ce que le modèle a appris

Effet de chaque variable quand elle passe de sa valeur basse (percentile 5) à sa valeur haute
(percentile 95), toutes choses égales par ailleurs (dépendance partielle). Énergie Charpy à
−40 °C.

| Variable | Plage | Effet sur Re (SVR / forêt) | Effet sur KV à −40 °C (SVR / forêt) |
|---|---|---|---|
| Mn | 0,6 → 1,8 % | +111 / +61 MPa | optimum vers 1,1–1,3 % dans les deux modèles |
| Cr | 0 → 8,9 % | +357 / +110 MPa | −62 / −7 J |
| Mo | 0 → 1,1 % | +118 / +71 MPa | −12 / −13 J |
| V | 0 → 0,2 % | +186 / +56 MPa | −23 / −12 J |
| Nb | 0 → 500 ppm | +104 / +45 MPa | −32 / 0 J |
| Ni | 0 → 2,3 % | +85 / +44 MPa | −10 / −1 J |
| C | 0,04 → 0,12 % | +35 / +16 MPa | baisse au-delà de ~0,08 % |
| Si | 0,18 → 0,55 % | +42 / +19 MPa | −15 / −4 J |
| O | 230 → 700 ppm | −38 / −5 MPa | **−36 / −13 J** |
| N | 50 → 370 ppm | +49 / +19 MPa | −4 / −7 J |
| P | 0,005 → 0,026 % | +21 / +10 MPa | −6 / −7 J |
| Apport de chaleur | 1 → 4,8 kJ/mm | −34 / −13 MPa | −29 / −1 J |
| Température entre passes | 150 → 250 °C (Charpy) | effet faible | **+17 / +10 J** |
| Traitement thermique | 0 → 750 °C | −33 / −6 MPa | +4 / 0 J |

La forêt aléatoire donne des effets plus faibles, car elle ne prédit jamais au-delà des
valeurs vues (plateaux). Le sens des effets compte plus que leur valeur exacte.

**Le compromis résistance / ténacité** : Cr, Mo, V et Nb augmentent fortement la résistance
mais dégradent la ténacité. À l'inverse, l'oxygène et l'apport de chaleur dégradent les deux
(« perdant-perdant »), et le manganèse augmente la résistance sans pénaliser la ténacité
jusqu'à environ 1,3 %.

## 6. Recommandations

### 6.1 Recommandations pratiques pour les soudeurs

Elles sont classées selon ce que le soudeur ou le responsable soudage contrôle réellement.

| Levier | Que faire concrètement | Ce que montrent les modèles | Confiance |
|---|---|---|---|
| **Consommable : manganèse** | Choisir une électrode, un fil ou un couple fil/flux donnant **environ 1,0 à 1,4 % de Mn** dans le métal fondu (voir la fiche technique du fabricant) | Résistance en hausse avec Mn ; ténacité maximale vers 1,1–1,3 % | **Forte** : deux modèles d'accord, cohérent avec les travaux d'Evans sur les soudures C-Mn [Evans & Bailey, 1997] |
| **Consommable : carbone** | Préférer un métal fondu à **bas carbone (≤ 0,08 %)** | Résistance en hausse avec C, ténacité en baisse au-delà de ~0,08 % | Moyenne |
| **Consommable : éléments d'alliage** | N'utiliser des consommables alliés au **Cr, Mo, V, Nb** que si le cahier des charges l'exige (haute résistance, tenue à chaud) ; dans ce cas, **prévoir des essais Charpy** | Forte hausse de résistance, perte de ténacité | **Forte** pour Cr, Mo et V ; moyenne pour Nb |
| **Limiter l'oxygène** | Préférer les **enrobages et flux basiques** aux rutiles ou acides, et un gaz de protection peu oxydant (moins de CO₂/O₂) ; garder une bonne protection gazeuse | L'oxygène dégrade nettement la ténacité | **Forte** : deux modèles d'accord ; les inclusions d'oxydes sont un facteur connu de fragilité [Grong, 1997] |
| **Apport de chaleur** | Le garder **modéré (environ 1 à 2 kJ/mm)**. Apport de chaleur = k × U × I / v (k : rendement du procédé, par exemple 0,8 en MMA et 1,0 en arc submergé selon EN 1011-1 ; U tension ; I intensité ; v vitesse d'avance). Pour le réduire : avancer plus vite, baisser l'intensité ou la tension, faire **plus de passes plus fines** | Résistance (deux modèles) et ténacité (SVR) en baisse quand l'apport de chaleur augmente | Moyenne à forte : effet connu (microstructure plus grossière) |
| **Température entre passes** | La **mesurer avant chaque passe** (thermomètre de contact, crayons thermosensibles) et la tenir dans une plage contrôlée ; dans les données, **200–250 °C** donne une meilleure ténacité que 150 °C | Ténacité en hausse de 150 à 250 °C (deux modèles), résistance peu affectée | Moyenne : peu de données hors de 150–250 °C, ne pas extrapoler |
| **Impuretés (P, S, N)** | Utiliser des consommables à faibles P et S ; limiter l'azote par une bonne protection (**arc court**, débit de gaz adapté, pas de courants d'air) | Petite perte de ténacité quand P ou N augmentent | Moyenne |
| **Traitement thermique après soudage** | Le réaliser quand il est exigé (détensionnement), mais **ne pas compter sur lui pour améliorer la ténacité**, et anticiper une baisse de résistance | Résistance en baisse (SVR, environ −30 MPa), pas de gain net de ténacité | Faible : effet dépendant de la composition |

**En résumé, une bonne soudure** : un consommable bas carbone, autour de 1,0–1,4 % de Mn, peu
d'oxygène et d'impuretés ; un apport de chaleur modéré ; une température entre passes
contrôlée ; des éléments d'alliage durcissants seulement si la résistance l'exige, en
vérifiant alors la ténacité.

### 6.2 Recommandations sur l'usage des modèles

- **Limite d'élasticité** : utilisable pour **présélectionner** des combinaisons consommable /
  paramètres avant les essais, avec une marge de sécurité d'environ 90 MPa sur le seuil visé.
- **Ténacité** : pas assez fiable pour se passer des essais Charpy. Ceux-ci restent
  indispensables, en particulier pour les soudures alliées.
- **Domaine de validité** : les modèles sont fiables surtout **à l'intérieur du domaine déjà
  exploré** (soudures C-Mn faiblement alliées). Sur des soudures proches de celles déjà vues,
  l'erreur est environ deux fois plus faible que sur une publication nouvelle.

## 7. Limites et pistes

- **Corrélation n'est pas causalité** : les effets sont appris sur des données publiées.
  Certains éléments varient ensemble (Cr et Mo), et une partie de l'effet peut venir des
  différences entre publications.
- **Domaine** : les recommandations ne valent qu'à l'intérieur des plages étudiées
  (percentiles 5–95).
- **Ténacité mal prédite** (R² ≈ 0,28) : les recommandations sur la ténacité sont des
  tendances à confirmer.
- **Ni** : les modèles ne montrent pas de gain de ténacité avec le nickel. Dans ces données, son
  effet dépend probablement de la teneur en manganèse, et il faut rester prudent.
- **Variables absentes de la base** : hydrogène, préparation des bords, position de soudage,
  épaisseur. Les bonnes pratiques habituelles sur ces points (étuvage des électrodes
  basiques, nettoyage, préchauffage) restent nécessaires.
- **Seuils et références normatives** : les seuils de 420 MPa et 47 J, ainsi que les
  rendements de procédé, sont donnés à titre d'illustration et sont à vérifier dans les
  normes applicables.
- **Pistes** :
  - des plans d'expériences ciblés, en faisant varier un paramètre à la fois (Mn, O, apport
    de chaleur, température entre passes), pour confirmer les effets ;
  - davantage de courbes Charpy complètes ;
  - la prise en compte de la microstructure quand elle est disponible.

## Références

- Evans, G. M., & Bailey, N. (1997). *Metallurgy of Basic Weld Metal*. Abington Publishing.
- Grong, Ø. (1997). *Metallurgical Modelling of Welding* (2ᵉ éd.). The Institute of Materials.
- Cool, T., Bhadeshia, H. K. D. H., & MacKay, D. J. C. (1997). The yield and ultimate tensile strength of steel welds. *Materials Science and Engineering A*, 223.
- NF EN ISO 2560 : Produits consommables pour le soudage — Électrodes enrobées pour le soudage manuel à l'arc des aciers non alliés et à grains fins — Classification.
- NF EN 1011-1 : Soudage — Recommandations pour le soudage des matériaux métalliques — Partie 1 : lignes directrices générales pour le soudage à l'arc.
