"""Chargement des données et construction des variables.

Toutes les transformations sont appliquées ligne par ligne, sans statistique
calculée sur le jeu de données : on peut donc appeler `build_features` sur le
jeu d'entraînement et sur le jeu de test séparément, sans fuite d'information.
"""
import json
import warnings

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape

CLASSES = ['Demolition', 'Road', 'Residential', 'Commercial', 'Industrial', 'Mega Projects']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

URBAN_TYPES = ['Dense Urban', 'Sparse Urban', 'Industrial', 'Rural', 'Urban Slum']
GEOGRAPHY_TYPES = ['Barren Land', 'Coastal', 'Dense Forest', 'Desert', 'Farms', 'Grass Land',
                   'Hills', 'Lakes', 'River', 'Snow', 'Sparse Forest']

# Ordre d'avancement d'un chantier, du terrain naturel au bâtiment en service.
# « Prior Construction » (construction déjà présente) est placé au même niveau
# qu'un bâtiment terminé.
STATUS_ORDER = {
    'Greenland': 0, 'Land Cleared': 1, 'Excavation': 2, 'Materials Dumped': 3,
    'Materials Introduced': 4, 'Construction Started': 5, 'Construction Midway': 6,
    'Construction Done': 7, 'Prior Construction': 7, 'Operational': 8,
}
STATUSES = sorted(set(STATUS_ORDER) - {'Prior Construction'}) + ['Prior Construction']

N_DATES = 5
COLORS = ['red', 'green', 'blue']


# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------
def _iter_features(path):
    """Parcourt les entités d'un GeoJSON une à une.

    Les fichiers de la compétition contiennent une entité par ligne : les lire
    ligne à ligne consomme peu de mémoire et tolère un fichier tronqué. Si le
    fichier n'est pas organisé ainsi, on le lit d'un bloc.
    """
    found = 0
    with open(path, errors='ignore') as f:
        for line in f:
            line = line.strip().rstrip(',')
            if line.startswith('{ "type": "Feature"') or line.startswith('{"type": "Feature"'):
                try:
                    feature = json.loads(line, strict=False)
                except json.JSONDecodeError:
                    print(f"ATTENTION : {path} semble tronqué, lecture arrêtée après {found:,} entités. "
                          "Vérifiez que le fichier est complet.")
                    return
                found += 1
                yield feature
    if found == 0:
        with open(path) as f:
            yield from json.load(f, strict=False)['features']


def load_geojson(path):
    """Renvoie un DataFrame des attributs, avec la géométrie (shapely) en colonne `geometry`."""
    properties, geometries = [], []
    for feature in _iter_features(path):
        properties.append(feature['properties'])
        geom = feature.get('geometry')
        geometries.append(shape(geom) if geom else None)
    df = pd.DataFrame(properties)
    df['geometry'] = geometries
    return df


def zone_id(df):
    """Identifiant de zone : les polygones d'une même image partagent les mêmes dates."""
    dates = df[[f'date{i}' for i in range(N_DATES)]].fillna('NA').astype(str)
    return pd.util.hash_pandas_object(dates, index=False).values


# --------------------------------------------------------------------------
# Construction des variables
# --------------------------------------------------------------------------
def _multi_hot(series, categories, prefix):
    values = series.fillna('').str.split(',')
    out = pd.DataFrame(0, index=series.index, columns=[f'{prefix}_{c}' for c in categories], dtype='int8')
    for i, items in zip(series.index, values):
        for item in items:
            item = item.strip()
            if item in categories:
                out.at[i, f'{prefix}_{item}'] = 1
    return out


def _snapshots(df):
    """Remet les 5 prises de vue dans l'ordre chronologique.

    Dans les fichiers, les dates ne sont pas triées : date0 peut être postérieure
    à date3. La date `date{i}` et son statut sont associés aux statistiques
    d'image `*_date{i+1}` (les images sont numérotées de 1 à 5).
    Renvoie des tableaux (n, 5) triés par date : jours depuis 2000, statut, et
    un tableau (n, 5, 6) des moyennes et écarts-types RVB.
    """
    n = len(df)
    ref = pd.Timestamp('2000-01-01')
    days = np.full((n, N_DATES), np.nan)
    status = np.full((n, N_DATES), np.nan)
    img = np.full((n, N_DATES, 6), np.nan)
    for i in range(N_DATES):
        d = pd.to_datetime(df[f'date{i}'], format='%d-%m-%Y', errors='coerce')
        days[:, i] = (d - ref).dt.days.values
        status[:, i] = df[f'change_status_date{i}'].map(STATUS_ORDER).values
        cols = [f'img_{c}_{s}_date{i + 1}' for s in ('mean', 'std') for c in COLORS]
        img[:, i, :] = df[cols].values

    order = np.argsort(np.where(np.isnan(days), np.inf, days), axis=1)
    rows = np.arange(n)[:, None]
    return days[rows, order], status[rows, order], img[rows, order], order


def _geometry_features(geoms):
    geoms = np.asarray(geoms, dtype=object)
    centroid = shapely.centroid(geoms)
    lon, lat = shapely.get_x(centroid), shapely.get_y(centroid)

    # Projection locale en mètres (équirectangulaire autour du centroïde) :
    # suffisante à l'échelle d'un bâtiment, sans dépendance supplémentaire.
    def to_metres(g, lon0, lat0):
        k = np.cos(np.radians(lat0))
        return shapely.transform(g, lambda xy: np.column_stack(
            [(xy[:, 0] - lon0) * 111_320 * k, (xy[:, 1] - lat0) * 110_540]))

    metric = np.array([to_metres(g, x, y) if g is not None else None
                       for g, x, y in zip(geoms, lon, lat)], dtype=object)
    area = shapely.area(metric)
    perimeter = shapely.length(metric)
    n_vertices = shapely.get_num_coordinates(metric)

    with warnings.catch_warnings():
        # Polygones dégénérés (tous les points alignés) : le rectangle est vide, on garde NaN
        warnings.simplefilter('ignore', RuntimeWarning)
        rect = shapely.oriented_envelope(metric)
    xy = [np.asarray(r.exterior.coords) if r is not None and r.geom_type == 'Polygon' else None for r in rect]
    sides = np.array([[np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[1])] if c is not None else [np.nan, np.nan]
                      for c in xy])
    long_side, short_side = sides.max(1), sides.min(1)

    with np.errstate(divide='ignore', invalid='ignore'):
        return pd.DataFrame({
            'lon': lon, 'lat': lat,
            'area_m2': area,
            'log_area': np.log1p(area),
            'perimeter_m': perimeter,
            'n_vertices': n_vertices,
            'compactness': 4 * np.pi * area / perimeter ** 2,   # 1 pour un disque
            'elongation': long_side / short_side,               # 1 pour un carré
            'rect_fill': area / (long_side * short_side),       # 1 pour un rectangle
        })


def build_features(df):
    """Construit la matrice des variables à partir du DataFrame brut."""
    out = [
        _multi_hot(df['urban_type'], URBAN_TYPES, 'urban'),
        _multi_hot(df['geography_type'], GEOGRAPHY_TYPES, 'geo'),
    ]

    days, status, img, _ = _snapshots(df)
    feats = {}

    # Temps : date de la première image, écarts entre images, durée totale
    feats['first_date_years'] = days[:, 0] / 365.25
    for i in range(N_DATES - 1):
        feats[f'gap_{i}_{i + 1}_days'] = days[:, i + 1] - days[:, i]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        feats['span_days'] = np.nanmax(days, 1) - np.nanmin(days, 1)

    # Statut du chantier à chaque date (ordre chronologique) et dynamique
    for i in range(N_DATES):
        feats[f'status_t{i}'] = status[:, i]
    with np.errstate(invalid='ignore'):
        feats['status_first'] = status[:, 0]
        feats['status_last'] = status[:, -1]
        feats['status_progress'] = status[:, -1] - status[:, 0]
        feats['status_max'] = np.nanmax(np.where(np.isnan(status), -1, status), 1)
        feats['status_n_changes'] = (np.diff(status, axis=1) != 0).sum(1)
    raw_status = df[[f'change_status_date{i}' for i in range(N_DATES)]]
    for s in STATUSES:
        feats[f'n_status_{s.replace(" ", "_")}'] = (raw_status == s).sum(1).values

    # Couleurs : valeurs à chaque date, puis évolution entre dates
    names = [f'{s}_{c}' for s in ('mean', 'std') for c in COLORS]
    for i in range(N_DATES):
        for j, name in enumerate(names):
            feats[f'img_{name}_t{i}'] = img[:, i, j]
    brightness = img[:, :, :3].mean(2)
    for i in range(N_DATES):
        feats[f'brightness_t{i}'] = brightness[:, i]
    for i in range(N_DATES - 1):
        feats[f'color_change_{i}_{i + 1}'] = np.linalg.norm(img[:, i + 1, :3] - img[:, i, :3], axis=1)
    feats['color_change_total'] = np.linalg.norm(img[:, -1, :3] - img[:, 0, :3], axis=1)
    feats['brightness_change'] = brightness[:, -1] - brightness[:, 0]
    feats['texture_change'] = img[:, -1, 3:].mean(1) - img[:, 0, 3:].mean(1)

    out.append(pd.DataFrame(feats, index=df.index))
    out.append(_geometry_features(df['geometry'].values).set_index(df.index))
    X = pd.concat(out, axis=1).astype('float32')
    return X.replace([np.inf, -np.inf], np.nan)


def load_dataset(path, with_target=True):
    """Charge un fichier et renvoie (X, y, zones, ids)."""
    raw = load_geojson(path)
    X = build_features(raw)
    y = raw['change_type'].map(CLASS_TO_ID).values if with_target else None
    return X, y, zone_id(raw), raw['index'].values
