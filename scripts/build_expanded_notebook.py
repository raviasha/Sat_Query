"""Build the expanded experiment using unchanged model and audit stage logic."""
import base64
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as td:
    subprocess.run(['uv','build','--wheel','--out-dir',td],cwd=ROOT,check=True)
    wheel=next(Path(td).glob('*.whl'));data=wheel.read_bytes();name=wheel.name
install=f'''import base64, hashlib, subprocess, sys
from pathlib import Path
payload=base64.b64decode({base64.b64encode(data).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={hashlib.sha256(data).hexdigest()!r}
p=Path('/content')/{name!r};p.write_bytes(payload)
subprocess.check_call([sys.executable,'-m','pip','install','--quiet',str(p)+'[downloads,features]','zstandard'])
from google.colab import drive
drive.mount('/content/drive')
print('Package installed and Drive mounted',flush=True)
'''
cells=[]
def add(kind,text):
    cell={'id':f'expanded-{len(cells):02d}','cell_type':kind,'metadata':{},'source':text.splitlines(True)}
    if kind=='code':compile(text,'cell','exec');cell.update(outputs=[],execution_count=None)
    cells.append(cell)
add('markdown','# SatQuery: expanded linear baseline\nRun in the education account with a T4 GPU. Reads completed official archives, selects 4,600 training areas and preserves the original 200 validation and 200 test areas. The same seed, linear softmax head, normalization and training settings are retained. Select models on validation only. Results from the reused test set are comparative, not a new untouched final benchmark.\n\nRun cells in order. Each stage checks prior artifacts. Saves to `MyDrive/SatQuery/pipeline-5000`. Large archive scans may take hours, but selected modality caches can be reused after reconnection. Do not run duplicate copies.\n')
add('code',install)
add('markdown','## Extract the selected areas from official archive parts')
add('code',(ROOT/'scripts/colab_official_subset.py').read_text())
for script,title in [('colab_stage1.py','1. Prepare tensors'),('colab_stage2.py','2. Extract CROMA features'),('colab_stage3_4.py','3–4. Audit spatial labels and percentages'),('colab_stage5.py','5. Train the same linear model'),('colab_stage6_7.py','6–7. Evaluate class errors and verify inference')]:
    src=(ROOT/'scripts'/script).read_text()
    src=src.replace('pipeline-1000','pipeline-5000').replace('bigearthnet-v2-1000','bigearthnet-v2-5000')
    src=src.replace('1000','5000').replace('1,000','5,000').replace('225000','1125000')
    src=src.replace("'train':600","'train':4600").replace("'train': 600","'train': 4600").replace("'train_areas':600","'train_areas':4600")
    # Keep the original 1000 bootstrap repetitions; it is an evaluation setting.
    src=src.replace('bootstrap_repeats=5000','bootstrap_repeats=1000')
    if script=='colab_stage2.py':
        src=src.replace("print('CROMA device:', device, flush=True)","assert device == 'cuda', 'Choose a T4 GPU before feature extraction'\nprint('CROMA device:', device, flush=True)")
    add('markdown','## '+title);add('code',src)
add('markdown','## Compare with the original baseline\nOriginal held-out dominant accuracy: 53.8743%; coverage MAE: 6.1553 percentage points. Compare per-class errors and support as well. Class percentages are not confidence scores. A nonlinear model is a separate future comparison.\n')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':cells}
p=ROOT/'notebooks/SatQuery_Pipeline_5000.ipynb';p.write_text(json.dumps(nb,indent=2)+'\n');print(p)
