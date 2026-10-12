# Résultats, recommandations et limites

Tous les chiffres sont mesurés sur des **publications jamais vues** pendant l'entraînement
(validation croisée imbriquée groupée par publication, voir
[méthodologie §5](methodologie.md#5-protocole-de-validation)). Les écarts-types sont
calculés entre les 5 plis ; ils sont indicatifs, la significativité des écarts est traitée
en section 4.3.

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
| SVR | 66,8 ± 11,8 | 48,8 | 0,45 |
| Forêt aléatoire | 67,5 ± 11,7 | 49,7 | 0,44 |
| **Gradient boosting** | **62,9 ± 7,3** | **46,1** | **0,50** |

- **Le gradient boosting est le meilleur en moyenne**, avec le RMSE le plus régulier d'un
  pli à l'autre. Il gagne 3 plis sur 5 face au SVR comme face à la forêt. La validation
  retient toujours des arbres très simples (profondeur 2). L'écart n'est pas significatif
  (section 4.3).
- **SVR et forêt aléatoire sont à égalité** : la forêt gagne 3 plis sur 5, le SVR 2, avec des
  écarts de 1 à 9 MPa dans les deux sens.
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
| Gradient boosting | 46,6 ± 7,9 | 34,8 | 0,00 |

- **Le SVR est le seul modèle nettement meilleur que la référence « température seule ».**
- **La forêt aléatoire est biaisée par les lignes « énergie fixée »** (100 J ou 28 J, voir
  [méthodologie §3.3](methodologie.md#33-particularité-des-données-charpy)) : elle prédit un
  plateau à 100 J pour les soudures d'Evans, quelle que soit la température. Pour ces
  recettes, l'entraînement contient surtout des lignes « 100 J », chacune à la température
  de transition de sa soudure. Le SVR, plus lisse, résiste mieux à ce biais.
- **Le gradient boosting a le même défaut** (plateau vers 100–110 J), et prédit en plus des
  énergies **négatives** au froid (environ −20 J), physiquement impossibles : la somme des
  corrections de ses arbres peut sortir de la plage des valeurs vues.
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

[`03_modeles_limite_elasticite.ipynb`, sections 5 et 6] — limite d'élasticité.

### 3.1 Self-training avec forêt aléatoire

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

### 3.2 Régularisation de variété (Laplacian RLS)

Méthode choisie pour répondre à l'échec du self-training (voir
[méthodologie §8.4](methodologie.md#84-second-choix--régularisation-de-variété)) :

| Méthode | RMSE (MPa) |
|---|---|
| Témoin supervisé (ridge à noyau, λ_I = 0) | 63,6 ± 10,8 |
| Régularisation de variété, inductive | 63,8 ± 10,9 |
| Régularisation de variété, transductive (voit les recettes de test) | 63,9 ± 11,0 |

Avec des étiquettes rares (γ et λ_A fixés), plus le poids λ_I du graphe augmente, plus
l'erreur augmente : avec 25 % des étiquettes, 70,1 MPa sans graphe, 70,9 avec λ_I = 0,1,
76,3 avec λ_I = 1 et 83,0 avec λ_I = 10 en inductif ; la version transductive fait encore
moins bien. La validation croisée retient presque toujours le plus petit poids du graphe.

**Pourquoi la méthode n'apporte rien :**
1. **L'hypothèse de variété est vraie, mais déjà exploitée** : deux recettes voisines dans le
   graphe diffèrent de 40 MPa en médiane, contre 83 MPa pour deux soudures au hasard. Mais
   le noyau gaussien encode déjà « recettes proches, prédictions proches ».
2. **Les recettes non étiquetées sont peu nombreuses et loin des données étiquetées** : 262
   recettes nouvelles, à une distance médiane de 2,1 de la plus proche soudure mesurée,
   contre 0,8 entre soudures mesurées.
3. **Le décalage propre à chaque publication ne se voit pas dans les recettes** : la moyenne
   par publication explique 44 % de la variance (R² ajusté). Connaître les recettes de test
   ne permet pas de corriger ce décalage, qui n'apparaît que dans les mesures.

**Conclusion sur le semi-supervisé** : deux méthodes de principes différents (pseudo-
étiquettes, géométrie des entrées), y compris une version transductive qui vise le
changement de domaine, aboutissent au même constat : **la limite n'est pas le manque
d'étiquettes, mais le manque de publications variées**. Ce qui améliorerait le modèle, ce
sont de nouvelles mesures sur des recettes peu explorées.

Le témoin supervisé (ridge à noyau, environ 64 MPa) fait aussi bien que le gradient
boosting et mieux que le SVR, avec le même noyau : la perte quadratique semble mieux adaptée
ici que la perte insensible à ε du SVR.

## 4. Comparaison globale et choix de l'approche

[`05_comparaison_recommandations.ipynb`]

### 4.1 Part des prédictions dans la tolérance

| Modèle | Limite d'élasticité (± 50 MPa) | Charpy (± 20 J) |
|---|---|---|
| Moyenne | 43 % | 47 % |
| Régression linéaire | 44 % | 32 % |
| k-NN | 50 % | 42 % |
| SVR | 64 % | 42 % |
| Forêt aléatoire | 61 % | 40 % |
| **Gradient boosting** | **66 %** | 42 % |

Pour le Charpy, la moyenne obtient le meilleur score, simplement parce que beaucoup de lignes
valent 100 J et que la moyenne (≈ 88 J) tombe près d'elles. C'est un exemple de métrique
trompeuse prise isolément.

### 4.2 Métrique de décision : accepter ou refuser une soudure

Seuils : 420 MPa (16 % des soudures sont en dessous) et 47 J à la température d'essai (28 %
en dessous).

Sans marge :

| Modèle | Fausse acceptation, Re | Faux rejet, Re | AUC, Re | Fausse acceptation, Charpy | Faux rejet, Charpy | AUC, Charpy |
|---|---|---|---|---|---|---|
| Température seule | — | — | — | 63 % | 6 % | 0,70 |
| Régression linéaire | 63 % | 11 % | 0,74 | 77 % | 7 % | 0,63 |
| k-NN | 74 % | 2 % | 0,68 | 100 % | 0 % | 0,66 |
| SVR | **45 %** | 4 % | 0,85 | **64 %** | 3 % | **0,82** |
| Forêt aléatoire | 70 % | 1 % | 0,88 | 96 % | 2 % | 0,75 |
| Gradient boosting | **45 %** | 3 % | **0,92** | 77 % | 4 % | 0,73 |

- **Les modèles tirent vers la moyenne** et prédisent rarement des valeurs basses : sans
  précaution, ils acceptent une grande partie des soudures trop faibles.
- **L'AUC** mesure la capacité à classer les soudures, quelle que soit la marge : le
  gradient boosting est le meilleur pour la limite d'élasticité, le SVR pour le Charpy.

Avec une **marge de sécurité** choisie en validation croisée groupée par publication (voir
[méthodologie §6](methodologie.md#6-métriques)), pour viser au plus 10 % de fausse
acceptation :

| Modèle | Limite d'élasticité | Charpy |
|---|---|---|
| SVR | marge 50 à 100 MPa : 16 % de fausse acceptation, 52 % de faux rejets | **marge 40 à 50 J : 9 %, 48 %** |
| Forêt aléatoire | marge 60 à 80 MPa : 10 %, 44 % | marge 50 à 55 J : 10 %, 73 % |
| Gradient boosting | **marge 40 à 60 MPa : 11 %, 30 %** | marge 55 à 80 J : 11 %, 77 % |

- Pour la **limite d'élasticité**, le gradient boosting offre de loin le meilleur
  compromis. Pour le SVR, la marge nécessaire varie beaucoup selon les publications, et
  l'objectif de 10 % n'est pas tenu sur des publications nouvelles.
- Pour le **Charpy**, le SVR est le seul modèle utilisable à un coût raisonnable.
- Lire la marge directement sur les taux mesurés aurait donné des chiffres trop optimistes
  (pour le SVR sur la limite d'élasticité : 90 MPa et 10 % de fausse acceptation annoncés).
- **Conclusion pratique** : le modèle peut **présélectionner** des recettes, mais **pas
  remplacer les essais mécaniques**.

### 4.3 Les écarts sont-ils significatifs ?

**Test de Wilcoxon apparié** sur le RMSE calculé publication par publication (chaque
publication pèse autant), avec correction de Holm pour les six comparaisons :

| Comparaison (A vs B) | Publications | A meilleur | Écart médian | p-valeur | p corrigée (Holm) |
|---|---|---|---|---|---|
| Re : SVR vs forêt aléatoire | 42 | 57 % | −3,2 MPa | 0,42 | 1,00 |
| Re : gradient boosting vs SVR | 42 | 57 % | −1,5 MPa | 0,81 | 1,00 |
| Re : gradient boosting vs forêt aléatoire | 42 | 67 % | −4,0 MPa | 0,04 | 0,23 |
| Charpy : SVR vs forêt aléatoire | 26 | 58 % | −1,2 J | 0,45 | 1,00 |
| Charpy : SVR vs gradient boosting | 26 | 65 % | −4,3 J | 0,10 | 0,50 |
| Charpy : SVR vs température seule | 26 | 73 % | −3,7 J | 0,14 | 0,54 |

**Aucun écart ne résiste à la correction pour comparaisons multiples.**

**Bootstrap sur les publications** (intervalle de confiance à 95 % ; chaque ligne pèse
autant) :

| | RMSE (toutes les lignes) | IC 95 % |
|---|---|---|
| Re : SVR | 67,5 MPa | 56,5 – 78,3 |
| Re : forêt aléatoire | 68,3 MPa | 58,7 – 79,5 |
| Re : gradient boosting | 63,3 MPa | 54,1 – 73,3 |
| Charpy : température seule | 45,5 J | 37,4 – 53,8 |
| Charpy : SVR | 40,4 J | 32,1 – 48,8 |
| Charpy : forêt aléatoire | 46,0 J | 35,7 – 55,9 |
| Charpy : gradient boosting | 47,1 J | 37,3 – 57,2 |

| Écart de RMSE (A − B) | Écart | IC 95 % | Significatif |
|---|---|---|---|
| Re : SVR − forêt | −0,7 MPa | −9,4 – 6,8 | non |
| Re : gradient boosting − SVR | −4,3 MPa | −12,7 – 5,7 | non |
| Re : gradient boosting − forêt | −5,0 MPa | −10,9 – 0,4 | non |
| Charpy : SVR − forêt | −5,7 J | −11,4 – −0,3 | **oui** |
| Charpy : SVR − gradient boosting | −6,8 J | −13,5 – −0,8 | **oui** |
| Charpy : SVR − température seule | −5,2 J | −12,8 – 1,9 | non |

- **Limite d'élasticité** : aucun écart établi entre les trois meilleurs modèles.
- **Charpy** : le SVR est significativement meilleur que les deux modèles d'arbres d'après
  le bootstrap, mais pas d'après le Wilcoxon. Les deux méthodes ne pèsent pas les
  publications de la même façon : une seule publication, Evans-StressRelief-1991
  (94 lignes), fait environ la moitié de l'avance du SVR. Ses courbes de transition
  complètes sont mal prédites par les modèles d'arbres, qui y reproduisent le plateau à
  100 J appris sur les autres publications d'Evans.
- Le SVR ne bat pas significativement la référence « température seule » : la recette
  n'apporte qu'un gain modeste et incertain pour la ténacité.

### 4.4 Approche retenue

**Un SVR à noyau gaussien par propriété**, avec le pré-traitement décrit dans la
méthodologie et une validation croisée groupée par publication. Hyperparamètres finaux :
C = 10, γ = 0,01, ε = 0,2 pour la limite d'élasticité ; C = 1, γ = 0,01, ε = 0,05 pour le
Charpy.

Justification :
- c'est le seul modèle qui se comporte bien sur les deux cibles ;
- pour la ténacité, il est le meilleur, significativement devant la forêt et le gradient
  boosting (bootstrap), et moins sensible qu'eux aux lignes « énergie fixée » ;
- pour la limite d'élasticité, il ne se distingue pas significativement du gradient boosting
  et de la forêt ;
- ses effets sont lisses et faciles à interpréter.

**Le gradient boosting est l'alternative à privilégier pour présélectionner des recettes sur
la limite d'élasticité** : meilleur RMSE et meilleure AUC, et surtout le meilleur compromis
avec une marge de sécurité (30 % de faux rejets, contre 52 % pour le SVR).

La forêt aléatoire est gardée comme **modèle de contrôle** pour l'interprétation : elle ne
prédit jamais hors de la plage vue. Le semi-supervisé n'est pas retenu (section 3).

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
  paramètres avant les essais, de préférence avec le gradient boosting et une marge de
  sécurité d'environ 50 MPa sur le seuil visé. On peut alors attendre environ 10 % de
  fausse acceptation et 30 % de faux rejets sur des recettes nouvelles.
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
- **Incertitude sur les erreurs** : avec 26 à 42 publications indépendantes, les intervalles
  de confiance des RMSE sont larges (environ ± 11 MPa et ± 8 J), et la plupart des écarts
  entre les meilleurs modèles ne sont pas significatifs.
- **Les effets de la section 5 viennent du SVR et de la forêt** : le gradient boosting,
  meilleur pour la présélection sur la limite d'élasticité, n'a pas été utilisé pour
  l'interprétation.
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
