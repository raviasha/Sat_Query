"""Build a standalone annotation-matching notebook using the current packaged pipeline."""
import json
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(root/'scripts/build_pipeline_notebook.py')],check=True)
pipeline=json.loads((root/'notebooks/SatQuery_Pipeline_1000.ipynb').read_text())
cells=[]
def add(kind,text):
    c={'id':f'annotations-{len(cells):02d}','cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':c.update(execution_count=None,outputs=[])
    cells.append(c)
add('markdown','''# Match BigEarthNet.txt to the existing 1,000 image pairs
Run cells one at a time. CPU is sufficient. This notebook downloads the official 467 MB text-only table once to the SatQuery source cache and matches records to the existing image metadata. It does not download the imagery again or train a model.

Both Sentinel IDs are checked. Source text is preserved. Original text splits and image splits remain separate; a benchmark annotation holds out its entire image. Missing images and conflicts are reported. Annotations describe the original 1.2 km area or explicit referenced regions; they must not be copied onto every 80 m token as if independently labeled.
''')
add('markdown','## Install the checksum-verified SatQuery package')
add('code',''.join(next(c for c in pipeline['cells'] if c['cell_type']=='code')['source']))
add('markdown','## Connect the existing Drive')
add('code',"from google.colab import drive\ndrive.mount('/content/drive')\n")
add('markdown','## Download and verify the official text source')
add('code',(root/'scripts/colab_download_annotations.py').read_text())
add('markdown','## Match annotations and verify links to the saved features')
add('code',(root/'scripts/colab_match_annotations.py').read_text())
add('markdown','''## Outputs
Google Drive → MyDrive → SatQuery → pipeline-1000 → annotations:
- `annotations.parquet`: all matching source records plus image split, image index, permitted partition and training eligibility.
- `image-annotations.jsonl`: one entry per selected image pair, with its linked annotations; missing images remain visible.
- `feature-links.jsonl`: exact feature batch and position for every image.
- `report.json`: source hashes, record counts, missing IDs and split conflicts.

The parent folder contains `stage-8-annotations-report.json`, including verified feature links. No training occurs in this notebook.
''')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
p=root/'notebooks/SatQuery_BigEarthNet_Text_Matching.ipynb';p.write_text(json.dumps(nb,indent=2)+'\n');print(p)
