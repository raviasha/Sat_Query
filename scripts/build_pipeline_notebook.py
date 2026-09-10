"""Build the portable, readable Colab notebook from the separate stage scripts."""
import base64
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temporary:
    subprocess.run(['uv','build','--wheel','--out-dir',temporary],cwd=ROOT,check=True)
    wheel=next(Path(temporary).glob('*.whl'))
    data=wheel.read_bytes();filename=wheel.name
encoded=base64.b64encode(data).decode()
wrapped='\n'.join('    '+repr(encoded[i:i+100]) for i in range(0,len(encoded),100))
install=f'''import base64, hashlib, subprocess, sys
from pathlib import Path
wheel_path = Path('/content/{filename}')
wheel_bytes = base64.b64decode(\n{wrapped}\n)
assert hashlib.sha256(wheel_bytes).hexdigest() == '{hashlib.sha256(data).hexdigest()}'
wheel_path.write_bytes(wheel_bytes)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade', '--force-reinstall', '--no-deps', str(wheel_path)])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', str(wheel_path)+'[downloads,features]'])
print('Verified SatQuery package installed')
'''
cells=[]
def cell(kind,source):
    value={'id':f'pipeline-cell-{len(cells):02d}','cell_type':kind,'metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code':value.update(execution_count=None,outputs=[])
    cells.append(value)
cell('markdown','''# SatQuery: 1,000-area CROMA pipeline
Run the cells **one at a time** and inspect each completion report before continuing.
Use a free T4 GPU runtime (Runtime → Change runtime type). This notebook uses the previously downloaded `MyDrive/SatQuery/bigearthnet-v2-1000` ZIPs and saves separate stages under `MyDrive/SatQuery/pipeline-1000`.

The official area splits remain 600 train / 200 validation / 200 test. All 225 tokens from an area stay together. Existing completed exports are verified and reused. An existing model is reused; choose a new output folder for a new experiment.

Coverage percentages are **not confidence**. Final inference attaches held-out class errors and support. Label matching verifies CLC reference-map geometry, not independent ground truth at 80 metres.
''')
cell('markdown','## Install the included, checksum-verified package')
cell('code',install)
cell('markdown','## Connect the previously approved Google Drive')
cell('code',"from google.colab import drive\ndrive.mount('/content/drive')\n")
for name,title in [
 ('colab_stage1.py','1. Convert images into verified CROMA tensors'),
 ('colab_stage2.py','2. Extract frozen CROMA features'),
 ('colab_stage3_4.py','3–4. Generate class percentages and independently audit every 80 m block'),
 ('colab_stage5.py','5. Train on training areas and choose the epoch using validation'),
 ('colab_stage6_7.py','6–7. Test the frozen model, calculate class metrics and verify inference'),
]:
    cell('markdown','## '+title)
    cell('code',(ROOT/'scripts'/name).read_text())
cell('markdown','''## Reading the results
`evaluation/per-class-metrics.csv` contains coverage MAE/RMSE/bias in percentage points, errors restricted to blocks where a class occurs, support, and dominant-class precision/recall/F1. Low overall error for an absent class does not establish reliability.

`evaluation/reliability.json` is bound to the exact saved model. The inference module copies it into its export. Display these historical errors separately from predicted cover percentage. No calibrated per-prediction confidence is claimed.

The test set has now been used for this baseline. Select later features and models on validation data; reserve a fresh test set for the eventual final comparison.
''')
notebook={'cells':cells,'metadata':{'accelerator':'GPU','kernelspec':{'display_name':'Python 3','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
path=ROOT/'notebooks'/'SatQuery_Pipeline_1000.ipynb'
path.write_text(json.dumps(notebook,indent=2)+'\n')
print(path)
