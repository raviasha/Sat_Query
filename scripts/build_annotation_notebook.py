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
add('markdown','''# BigEarthNet.txt: download, match and audit
Run cells one at a time. CPU is sufficient. This notebook reuses the official 467 MB text table and existing image metadata and features. Configure the folders below for 5,000 images, the older 1,000-image run, or another selection. Each stage is separate; the fixed six-image audit is optional.

Both Sentinel IDs are checked. Source text is preserved. Original text splits and image splits remain separate; a benchmark annotation holds out its entire image. Missing images and conflicts are reported. Annotations describe the original 1.2 km area or explicit referenced regions; they must not be copied onto every 80 m token as if independently labeled.
''')
add('markdown','## Install the checksum-verified SatQuery package')
add('code',''.join(next(c for c in pipeline['cells'] if c['cell_type']=='code')['source']))
add('markdown','## Connect the existing Drive')
add('code',"from google.colab import drive\ndrive.mount('/content/drive')\n")
add('markdown','''## Configure existing data folders
Use the actual mounted paths. When running from another account, `pipeline-5000` and the imagery folder may be EDU-owned shortcuts inside a personal wrapper such as `SatQuery (1)`. Verify ownership in Drive. The wrapper itself is not necessarily EDU-owned. This cell requires existing inputs and creates nothing there.

For the old run, set the pipeline and image folder to their `1000` equivalents and point `TEXT_SOURCE` to the existing `annotation-source-cache` file; set `OUTPUT=P/'annotations'` to verify/reuse its original matched export.
''')
add('code',"""from pathlib import Path
from satquery.match_annotations import SOURCE_REVISION
P = Path('/content/drive/MyDrive/SatQuery (1)/pipeline-5000')
IMAGE_ROOT = Path('/content/drive/MyDrive/SatQuery (1)/bigearthnet-v2-5000')
TEXT_SOURCE = P/'annotations'/'source'/SOURCE_REVISION/'BigEarthNet.txt.parquet'
OUTPUT = P/'annotations'/'matched'
assert P.is_dir() and (IMAGE_ROOT/'metadata.parquet').is_file()
assert (P/'features'/'manifest.json').is_file()
print('Pipeline:', P, '\\nImages:', IMAGE_ROOT, '\\nText source:', TEXT_SOURCE, '\\nMatched output:', OUTPUT)
""")
add('markdown','## Download and verify the official text source')
add('code',(root/'scripts/colab_download_annotations.py').read_text())
add('markdown','## Match annotations and verify links to the saved features')
add('code',(root/'scripts/colab_match_annotations.py').read_text())
add('markdown','''## Outputs
Under the configured `OUTPUT` directory:
- `annotations.parquet`: all matching source records plus image split, image index, permitted partition and training eligibility.
- `image-annotations.jsonl`: one entry per selected image pair, with its linked annotations; missing images remain visible.
- `feature-links.jsonl`: exact feature batch and position for every image.
- `report.json`: source hashes, record counts, missing IDs and split conflicts.

The configured pipeline folder `P` contains `stage-8-annotations-report.json`, including verified feature links. No training occurs in this notebook.
''')
add('markdown','''## Optional: reproduce the six-image reference-map audit
This is a fixed sample with manually interpreted claims, not a parser for arbitrary Q&A and not a model evaluation. Requires the original 5,000-image metadata and `Reference_Maps-selected.zip`. It checks 18 presence claims, 18 area answers, 13 named boxes and selected caption claims. Two caption areas disagree with the maps. Choose a new output directory for each recheck.
''')
add('code',"import subprocess, sys\nsubprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'scipy'])\n")
audit=(root/'scripts/audit_annotation_samples.py').read_text()
add('code',"AUDIT_OUTPUT = P/'annotation-map-audit'/'recheck-six'\n"+
    "audit_script = Path('/content/satquery_audit_annotation_samples.py')\n"+
    'audit_script.write_text('+repr(audit)+')\n'+
    "subprocess.check_call([sys.executable, str(audit_script), '--pipeline', str(P), '--images', str(IMAGE_ROOT), '--source', str(TEXT_SOURCE), '--output', str(AUDIT_OUTPUT)])\n")
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
p=root/'notebooks/SatQuery_BigEarthNet_Text_Matching.ipynb';p.write_text(json.dumps(nb,indent=2)+'\n');print(p)
