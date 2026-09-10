"""Embed the built project wheel in a portable Colab notebook."""
import base64
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
wheel = root / 'dist/satquery_preprocessing-0.1.0-py3-none-any.whl'
payload = base64.b64encode(wheel.read_bytes()).decode()
digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
cells = []

def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)})

def code(text):
    compile(text, '<notebook-cell>', 'exec')
    cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {},
                  'outputs': [], 'source': text.splitlines(True)})

markdown('''# SatQuery: BigEarthNet v2 subset → Google Drive

Download **1,000 matched 1.2 × 1.2 km areas**: **600 training, 200 validation, 200 test**, preserving the official split assignments. The selection is reproducible and balanced across countries within each split. The original three demo areas are excluded.

Each area contains **12 optical bands, 2 SAR bands, and 1 original land-cover reference map**. Files retain native band resolutions and are compatible with the existing SatQuery preprocessing, CROMA feature, target and prediction modules.

Expected persistent storage is approximately **0.5–1 GB** for this initial subset and reference-map cache (estimate; actual size is reported at the end). Allow additional storage for downstream tensors, features and checkpoints. This notebook fetches individual records by byte range; it does not download the full 155 GB source database or the full optical/SAR archives.

Use a standard CPU runtime; no GPU or paid plan is needed for downloading. Choose **Runtime → Run all**, then complete Google's Drive connection prompt yourself. Only this notebook and the `MyDrive/SatQuery/bigearthnet-v2-1000` output folder are needed.

If Colab disconnects, reconnect and run all cells again with the same settings. Completed ZIP batches are verified and skipped; an unfinished batch is fetched again. Keep the notebook open while downloading.

**Source transparency:** imagery comes from an unofficial LMDB pre-conversion of BigEarthNet v2. GeoTIFFs are reconstructed from the stored native band arrays using the original reference-map georeferencing; TIFF headers need not be byte-identical to the original archives. For one existing demo area, all 12 optical and both SAR arrays were verified pixel-for-pixel against original TIFFs. This is a sample check, not a guarantee for every mirrored record. Source revisions and hashes are recorded in each batch.

Sources: [BigEarthNet](https://bigearth.net/), [native-array mirror](https://huggingface.co/datasets/hackelle/BigEarthNetV2-LMDB), [original metadata/reference-map mirror](https://huggingface.co/datasets/torchgeo/bigearthnet).
''')
markdown('## 1. Install the included SatQuery package\nThe notebook includes the project package, so no repository upload or access token is required.')
code(f'''import base64, hashlib, subprocess, sys
from pathlib import Path

wheel_path = Path('/content/{wheel.name}')
wheel_bytes = base64.b64decode('{payload}')
assert hashlib.sha256(wheel_bytes).hexdigest() == '{digest}', 'Embedded package checksum mismatch'
wheel_path.write_bytes(wheel_bytes)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', str(wheel_path) + '[downloads,features]'])
print('SatQuery installed. Continue to connect Google Drive.')
''')
markdown('## 2. Connect your Google Drive\nComplete the Google authorization dialog. Colab requests access to Drive for the mounted filesystem. The download cells below write inside the chosen SatQuery folder.')
code('''from google.colab import drive

drive.mount('/content/drive')
''')
markdown('## 3. Choose the subset\nUse these defaults for the first run. To create a different-sized subset later, use a new destination folder so the existing selection remains reproducible.')
code('''from pathlib import Path

DESTINATION = Path('/content/drive/MyDrive/SatQuery/bigearthnet-v2-1000')
WORK = Path('/content/satquery-download-work')
COUNTS = {'train': 600, 'validation': 200, 'test': 200}
SEED = 17
SHARD_SIZE = 100
WORKERS = 4

assert Path('/content/drive/MyDrive').is_dir(), 'Connect Drive before downloading'
print('Saving to:', DESTINATION)
print('Matched areas:', sum(COUNTS.values()), COUNTS)
''')
markdown('## 4. Download and verify each batch\nThe original reference-map archive is about 327 MB and is downloaded once with resume support. Progress then shows each batch of 100 paired areas. A completed batch is saved as a ZIP file plus a checksum receipt.')
code('''from satquery.download_bigearthnet import download_subset

result = download_subset(
    DESTINATION, WORK, counts=COUNTS, seed=SEED,
    shard_size=SHARD_SIZE, workers=WORKERS,
)
''')
markdown('## 5. Check completion\nA complete run has ten ZIP batches and 15,000 TIFFs. The selected metadata records the optical/SAR pair, original split, country and area-level labels. The reference maps provide the spatial targets.')
code('''import json
from satquery.preprocessing import sha256

report = json.loads((DESTINATION / 'download.json').read_text())
assert report['complete'] and report['sample_count'] == sum(COUNTS.values())
assert sha256(DESTINATION / 'selection.json') == report['selection_sha256']
assert sha256(DESTINATION / 'metadata.parquet') == report['metadata_sha256']
for shard in report['shards']:
    assert sha256(DESTINATION / shard['archive']) == shard['sha256'], shard['archive']
size = sum(p.stat().st_size for p in DESTINATION.rglob('*') if p.is_file())
print('Verified matched areas:', report['sample_count'])
print('Official splits:', report['counts'])
print('Countries:', report['countries'])
print('Labels present:', len(report['classes_present']))
print('Saved batches:', len(report['shards']))
print(f'Total saved size, including source cache: {size / 10**9:.3f} GB')
print('Drive folder:', DESTINATION)
''')
markdown('''## Next: use the separate ML modules

The download is complete when the final verification cell succeeds. Each ZIP contains the normal `BigEarthNet-S2/`, `BigEarthNet-S1/`, `Reference_Maps/` folders, its selected `metadata.parquet`, and provenance hashes. Extract one batch to Colab's temporary disk to process it with the existing pipeline, or extract all batches and use the top-level combined metadata (each ZIP has only its own metadata).

The 225 CROMA spatial features per area correspond to 8 × 8 pixels on the aligned 10 m grid. Keep all patches from an area in its original split. Train on training areas, tune on validation areas, and report final accuracy on the held-out test areas. This small country-balanced subset is an initial experiment, not an official full-dataset benchmark.

To increase the dataset later, edit `COUNTS` and choose a **new** `DESTINATION`. The same downloader and the existing downstream modules can be reused.
''')
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
    'colab': {'name': 'SatQuery_BigEarthNet_Drive_Download.ipynb', 'provenance': []},
    'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
    'language_info': {'name': 'python'}}, 'cells': cells}
for i, cell in enumerate(cells): cell['id'] = f'satquery-{i:02d}'
output = root / 'notebooks/SatQuery_BigEarthNet_Drive_Download.ipynb'
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(notebook, indent=1) + '\n')
print(f'{output}: {output.stat().st_size:,} bytes; embedded wheel SHA256 {digest}')
