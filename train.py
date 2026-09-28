"""Entraîne le modèle final sur tout le jeu d'entraînement et écrit la soumission.

Usage :
    python train.py [--train data/train.geojson] [--test data/test.geojson] [--out submissions/submission.csv]

Utilise les hyperparamètres trouvés dans le notebook (models/best_params.json)
s'ils existent, sinon un réglage par défaut raisonnable.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import CLASSES, load_dataset

DEFAULT_PARAMS = {'learning_rate': 0.1, 'max_leaf_nodes': 63, 'min_samples_leaf': 20,
                  'l2_regularization': 1.0, 'class_weight': 'balanced'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', default='data/train.geojson')
    parser.add_argument('--test', default='data/test.geojson')
    parser.add_argument('--params', default='models/best_params.json')
    parser.add_argument('--out', default='submissions/submission.csv')
    args = parser.parse_args()

    params = DEFAULT_PARAMS
    if Path(args.params).exists():
        params = json.loads(Path(args.params).read_text())
        print(f'Hyperparamètres chargés depuis {args.params}')
    print('Paramètres :', params)

    t = time.time()
    X, y, _, _ = load_dataset(args.train, with_target=True)
    X_test, _, _, test_ids = load_dataset(args.test, with_target=False)
    print(f'Variables construites en {time.time() - t:.0f} s : {X.shape[0]:,} polygones, {X.shape[1]} variables')

    t = time.time()
    model = HistGradientBoostingClassifier(max_iter=1000, early_stopping=True, random_state=42, **params)
    model.fit(X, y)
    print(f'Modèle entraîné en {time.time() - t:.0f} s ({model.n_iter_} itérations)')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pred = model.predict(X_test)
    pd.DataFrame({'Id': test_ids, 'change_type': pred}).to_csv(args.out, index=False)
    print(f'Soumission écrite dans {args.out}')
    print(pd.Series(pred).map(dict(enumerate(CLASSES))).value_counts().to_string())


if __name__ == '__main__':
    main()
