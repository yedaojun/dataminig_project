"""Independent regression checks in a temporary copy; leaves delivered data intact."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from pipeline import ROOT, NUM, INPUTS, MODES, features, make_preprocessor

def run(folder, ok=True):
    result = subprocess.run([sys.executable, str(folder/'run_a.py')], capture_output=True, text=True)
    if (result.returncode == 0) != ok:
        raise AssertionError(result.stdout + result.stderr)

def api(folder):
    result = subprocess.run([sys.executable, '-c', "from pipeline import load_split; load_split('train')"], cwd=folder, capture_output=True)
    assert result.returncode != 0

scratch = ROOT/'work'
scratch.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='group19_check_', dir=scratch) as tmp:
    folder=Path(tmp)/'project'
    shutil.copytree(ROOT,folder,ignore=shutil.ignore_patterns('__pycache__','work'))
    original=np.load(ROOT/'data/splits/indices.npz')
    # Rebuild from raw CSV and configuration; compare each index, not just counts.
    target=(folder/'data/splits').resolve()
    assert target.is_relative_to(scratch.resolve())
    shutil.rmtree(target)
    run(folder)
    with np.load(folder/'data/splits/indices.npz') as fresh:
        for part in ('train','validation','test'):
            np.testing.assert_array_equal(original[part], fresh[part])
    original.close()
    def hashes():
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/'results').iterdir()}
    before=hashes(); run(folder); assert hashes()==before
    # Reject altered configuration before overwriting prior outputs.
    config=folder/'config.json'; raw=config.read_bytes()
    for key,value in [('stratify','TWF'),('split_fractions',[0.6,0.2,0.2]),('seed',43)]:
        cfg=json.loads(raw); cfg[key]=value; config.write_text(json.dumps(cfg))
        run(folder,False); api(folder); assert hashes()==before
    config.write_bytes(raw)
    indices=folder/'data/splits/indices.npz'; backup=indices.read_bytes()
    with np.load(indices) as archive:
        changed={k:archive[k].copy() for k in archive.files}
    changed['train'][0]=changed['validation'][0]
    np.savez(indices,**changed)
    run(folder,False); api(folder); assert hashes()==before
    indices.write_bytes(backup)
    csv=folder/'data/raw/ai4i2020.csv'; csv.write_bytes(csv.read_bytes()+b'\n')
    run(folder,False); api(folder); assert hashes()==before

df=pd.read_csv(ROOT/'data/raw/ai4i2020.csv')
x=features(df,True)
np.testing.assert_allclose(x.mechanical_power,df['Torque [Nm]']*df['Rotational speed [rpm]']*np.pi/30)
np.testing.assert_allclose(x.temperature_difference,df[NUM[1]]-df[NUM[0]])
np.testing.assert_allclose(x.wear_load,df[NUM[4]]*df[NUM[3]])
audit=json.loads((ROOT/'results/audit.json').read_text())
assert audit['label_inconsistencies']==int((df['Machine failure']!=df[MODES].any(axis=1)).sum())==27
from pipeline import load_split
try:
    load_split('test')
except ValueError:
    pass
else:
    raise AssertionError('Test access must require acknowledgement')
print('PASS: clean rebuild; identical repeated outputs; config, index and raw-file mutation rejection; formulas; independent label audit; test guard.')
