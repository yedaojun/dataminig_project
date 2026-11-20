"""Shared data contract. Run python run_a.py before importing load_split."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
NUM = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
INPUTS = NUM + ['Type']
MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
TARGET = 'Machine failure'
ENGINEERED = ['temperature_difference', 'mechanical_power', 'wear_load']

def fingerprint():
    return hashlib.sha256((ROOT/'data/raw/ai4i2020.csv').read_bytes()).hexdigest()

def split_fingerprint():
    return hashlib.sha256((ROOT/'data/splits/indices.npz').read_bytes()).hexdigest()

def validate_config(cfg):
    if cfg['split_fractions'] != [0.7, 0.15, 0.15] or cfg['stratify'] != TARGET:
        raise ValueError('This implementation requires 70/15/15 stratified by Machine failure')

def features(frame, engineered=False):
    x = frame[INPUTS].copy()
    if engineered:
        x[ENGINEERED[0]] = x[NUM[1]] - x[NUM[0]]
        x[ENGINEERED[1]] = x[NUM[3]] * x[NUM[2]] * 2*np.pi/60
        x[ENGINEERED[2]] = x[NUM[4]] * x[NUM[3]]
    return x

def make_preprocessor(engineered=False, scale=False):
    return ColumnTransformer([
        ('numeric', StandardScaler() if scale else 'passthrough', NUM + (ENGINEERED if engineered else [])),
        ('type', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Type'])
    ], remainder='drop', verbose_feature_names_out=False)

def load_split(partition, engineered=False, allow_test=False):
    """Return X, binary y, multilabel Y in frozen positional-index order.
    Test access is an explicit protocol acknowledgement, not a security boundary.
    """
    if partition not in ('train', 'validation', 'test'):
        raise ValueError('Unknown partition')
    if partition == 'test' and not allow_test:
        raise ValueError('Freeze model choices and thresholds before allow_test=True')
    meta = json.loads((ROOT/'data/splits/manifest.json').read_text())
    if fingerprint() != meta['data_sha256']:
        raise ValueError('Raw dataset changed; do not reuse these indices')
    cfg = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    validate_config(cfg)
    if any(cfg[k] != meta[k] for k in ('seed', 'split_fractions', 'stratify')):
        raise ValueError('Configuration differs from frozen split')
    if split_fingerprint() != meta.get('indices_sha256'):
        raise ValueError('Frozen indices changed or checksum missing')
    df = pd.read_csv(ROOT/'data/raw/ai4i2020.csv')
    with np.load(ROOT/'data/splits/indices.npz') as archive:
        idx = archive[partition]
    subset = df.iloc[idx]
    return features(subset, engineered), subset[TARGET].copy(), subset[MODES].copy()
