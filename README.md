# Classification des changements urbains à partir d'images satellites

Classification de polygones repérés sur des séries d'images satellites en 6 types de changement urbain : démolition, route, résidentiel, commercial, industriel et méga-projet.

Projet du cours de Machine Learning de CentraleSupélec, réalisé dans le cadre d'une compétition Kaggle.

## Le problème

Chaque exemple est un polygone tracé sur une zone photographiée à 5 dates différentes. Pour chacun, on connaît :
- le type de zone urbaine et le type de géographie ;
- la moyenne et l'écart-type des couleurs rouge, verte et bleue à chaque date ;
- le statut du chantier à chaque date (terrain naturel, terrain dégagé, construction commencée, terminée…) ;
- la forme du polygone.

Il faut prédire le type de changement. Les classes sont très déséquilibrées : les changements résidentiels et commerciaux dominent, tandis que les méga-projets sont rares. La métrique retenue est donc le **F1 macro**, qui donne le même poids à chaque classe.

## Démarche

1. **Construction des variables** (`src/features.py`), identique pour l'entraînement et le test :
   - types de zone encodés en indicateurs (une zone peut en cumuler plusieurs) ;
   - géométrie **en mètres** : surface, périmètre, compacité, allongement, position ;
   - les 5 prises de vue sont **remises dans l'ordre chronologique**, absent des fichiers bruts ;
   - avancement du chantier encodé selon l'ordre logique des étapes, avec progression et nombre de changements ;
   - évolution des couleurs, de la luminosité et de la texture entre les dates.
2. **Validation croisée par zone géographique** : les polygones d'une même zone partagent les mêmes images. Une validation aléatoire laisse le modèle « reconnaître » une zone déjà vue et surestime sa performance ; la validation par zone mesure la capacité à traiter une région nouvelle.
3. **Comparaison de modèles** : référence naïve, régression logistique, Random Forest et gradient boosting, avec et sans pondération des classes rares.
4. **Réglage des hyperparamètres** du gradient boosting par recherche aléatoire.
5. **Analyse** : matrice de confusion, performance par classe, importance des familles de variables par permutation.

## Résultats

Scores en validation croisée par zone (5 plis), la zone entière étant absente de l'entraînement :

| Modèle | F1 macro | Accuracy |
|---|---|---|
| Référence naïve (classe majoritaire) | 0,111 | 0,501 |
| Régression logistique | 0,436 | 0,581 |
| Gradient boosting | 0,474 | 0,708 |
| Gradient boosting équilibré | 0,475 | 0,658 |
| Random Forest | 0,477 | 0,726 |
| Random Forest équilibrée | 0,490 | 0,715 |
| **Gradient boosting réglé** | **0,494** | **0,737** |

**Ce qu'on en retient :**

- Les modèles d'arbres dépassent nettement le modèle linéaire : les relations entre variables et classes ne sont pas linéaires.
- La validation aléatoire donne 0,543 pour le meilleur modèle, soit **0,05 de plus** que la validation par zone : sans regroupement par zone, on surestime la performance.
- Quatre classes sur six sont bien reconnues, avec un F1 de 0,84 pour les démolitions, 0,78 pour le résidentiel, 0,67 pour les routes et 0,65 pour le commercial. Les classes industriel (802 exemples) et méga-projets (40 exemples) ne sont presque jamais prédites : ce sont elles qui limitent le F1 macro.


## Structure du dépôt

```
.
├── src/
│   └── features.py                              # chargement et construction des variables
├── classification_changements_urbains.ipynb     # exploration, modèles, analyse
├── train.py                                     # entraînement final et soumission
├── data/                                        # données (non incluses, voir ci-dessous)
├── requirements.txt
└── README.md
```

