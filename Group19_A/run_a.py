"""Audit, freeze split, and verify shared preprocessing; no model evaluation."""
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline import ROOT, NUM, INPUTS, MODES, TARGET, fingerprint, split_fingerprint, validate_config, features, make_preprocessor

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def main():
    cfg = json.loads((ROOT/'config.json').read_text())
    validate_config(cfg)
    manifest = {'data_sha256': fingerprint(), 'seed': cfg['seed'], 'split_fractions': cfg['split_fractions'],
        'stratify': cfg['stratify'], 'index_definition': 'zero-based row position in unchanged raw CSV'}
    frozen_path = ROOT/'data/splits/manifest.json'
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if any(frozen.get(k) != v for k, v in manifest.items()):
            raise RuntimeError('Frozen split/data configuration differs. Team review required.')
        if frozen.get('indices_sha256') != split_fingerprint():
            raise RuntimeError('Frozen indices changed or checksum missing')
        manifest['indices_sha256'] = frozen['indices_sha256']
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    splits = ROOT/'data/splits'; splits.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT/'data/raw/ai4i2020.csv')
    assert df[INPUTS+[TARGET]+MODES].notna().all().all()
    assert np.isfinite(df[NUM].to_numpy()).all()
    assert df[[TARGET]+MODES].isin([0, 1]).all().all()
    assert set(df.Type) <= {'L', 'M', 'H'}
    mismatch = df[TARGET].ne(df[MODES].max(axis=1))
    audit = {'rows': len(df), 'columns': len(df.columns), 'missing_cells': int(df.isna().sum().sum()),
        'duplicate_full_rows': int(df.duplicated().sum()), 'duplicate_input_rows': int(df.duplicated(INPUTS).sum()),
        'duplicate_rows_excluding_ids': int(df.drop(columns=['UDI','Product ID']).duplicated().sum()),
        'label_inconsistencies': int(mismatch.sum()),
        'failure_without_mode': int(((df[TARGET]==1)&(df[MODES].sum(axis=1)==0)).sum()),
        'mode_without_failure': int(((df[TARGET]==0)&(df[MODES].sum(axis=1)>0)).sum()),
        'multiple_mode_rows': int((df[MODES].sum(axis=1)>1).sum())}
    save_json(out/'audit.json', audit)
    df.loc[mismatch].to_csv(out/'label_inconsistencies.csv', index=False)
    df.isna().sum().rename('missing_count').to_csv(out/'missing_by_column.csv')
    df.Type.value_counts().rename('count').to_csv(out/'type_counts.csv')
    df[NUM].describe().to_csv(out/'numeric_summary.csv')
    df[[TARGET]+MODES].agg(['sum','mean']).T.to_csv(out/'label_counts_rates.csv')
    df[MODES].T.dot(df[MODES]).to_csv(out/'mode_overlap.csv')
    df.groupby(MODES).size().rename('count').to_csv(out/'mode_combinations.csv')
    if (splits/'manifest.json').exists():
        if json.loads((splits/'manifest.json').read_text()) != manifest:
            raise RuntimeError('Frozen split/data configuration differs. Team review required.')
        with np.load(splits/'indices.npz') as z:
            indices = {k:z[k] for k in ('train','validation','test')}
    else:
        train, held = train_test_split(np.arange(len(df)), test_size=0.30, stratify=df[TARGET], random_state=cfg['seed'])
        val, test = train_test_split(held, test_size=0.5, stratify=df.iloc[held][TARGET], random_state=cfg['seed'])
        assert cfg['split_fractions'] == [0.7,0.15,0.15]
        indices = {'train':train, 'validation':val, 'test':test}
        np.savez(splits/'indices.npz', **indices)
        manifest['indices_sha256'] = split_fingerprint()
        save_json(splits/'manifest.json', manifest)
    joined = np.concatenate(list(indices.values()))
    assert len(joined)==len(df) and np.array_equal(np.sort(joined), np.arange(len(df)))
    assert [len(indices[k]) for k in indices] == [7000,1500,1500]
    rows=[]
    for name, idx in indices.items():
        part = df.iloc[idx]
        rows.append({'split':name,'n':len(idx),'normal':int((part[TARGET]==0).sum()),
            **{k:int(part[k].sum()) for k in [TARGET]+MODES}})
        pd.DataFrame({'row_index':idx,'UDI':part.UDI.to_numpy()}).to_csv(splits/f'{name}.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'split_counts.csv', index=False)
    df.iloc[indices['train']][NUM].std(ddof=0).rename('training_std_ddof0').to_csv(out/'training_sensor_std.csv')
    # Integration checks use validation only. Test labels above are audit counts only.
    checks=[]
    for eng in (False,True):
        for scale in (False,True):
            pre = make_preprocessor(eng,scale)
            train_x=features(df.iloc[indices['train']],eng)
            a=pre.fit_transform(train_x)
            b=pre.transform(features(df.iloc[indices['validation']],eng))
            assert a.shape == (7000,11 if eng else 8)
            assert b.shape == (1500,a.shape[1]) and np.isfinite(b).all()
            assert not set(pre.get_feature_names_out()) & set([TARGET]+MODES+['UDI','Product ID'])
            if scale:
                np.testing.assert_allclose(pre.named_transformers_['numeric'].mean_, train_x.select_dtypes('number').mean())
            checks.append({'engineered':eng,'scaled':scale,'train_shape':list(a.shape),'validation_shape':list(b.shape),'features':pre.get_feature_names_out().tolist()})
    save_json(out/'preprocessing_checks.json', checks)
    save_json(out/'environment.json', {'python':platform.python_version(), **{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','joblib']}})
    print(json.dumps(audit,indent=2)); print(pd.DataFrame(rows).to_string(index=False))
    print('PASS: disjoint coverage, fixed split, label exclusion, train-only fit, four preprocessing variants. No model test evaluation.')

if __name__ == '__main__':
    main()
