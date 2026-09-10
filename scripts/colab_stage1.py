# Stage 1 — verified BigEarthNet ZIPs to CROMA input tensors
import json
import shutil
import stat
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from satquery.prepare_croma import export_inputs
from satquery.preprocessing import DEFAULT_PROFILE, sha256

SOURCE = Path('/content/drive/MyDrive/SatQuery/bigearthnet-v2-1000')
PIPELINE = Path('/content/drive/MyDrive/SatQuery/pipeline-1000')
RAW = Path('/content/satquery-pipeline-raw')
PREPARED = PIPELINE / 'prepared'
assert SOURCE.is_dir(), 'Connect the Google Drive containing the completed download first'
PIPELINE.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
report = json.loads((SOURCE / 'download.json').read_text())
assert report['complete'] and report['sample_count'] == 1000
assert sha256(SOURCE / 'metadata.parquet') == report['metadata_sha256']
assert sha256(SOURCE / 'selection.json') == report['selection_sha256']
metadata = pd.read_parquet(SOURCE / 'metadata.parquet')
assert not metadata.patch_id.duplicated().any() and not metadata.s1_name.duplicated().any()
assert metadata.split.value_counts().to_dict() == {'train': 600, 'validation': 200, 'test': 200}

if not PREPARED.exists():
    RAW.mkdir(parents=True, exist_ok=True)
    for shard in report['shards']:
        archive = SOURCE / shard['archive']
        assert sha256(archive) == shard['sha256'], archive.name
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                path = (RAW / member.filename).resolve()
                assert RAW.resolve() in path.parents, 'Unsafe ZIP path'
                assert not stat.S_ISLNK(member.external_attr >> 16), 'Unexpected ZIP symlink'
            zipped.extractall(RAW)
        print('Verified and extracted', archive.name, flush=True)
    # Each ZIP has its own subset metadata. Restore the combined index last.
    shutil.copyfile(SOURCE / 'metadata.parquet', RAW / 'metadata.parquet')
    print('Converting 1,000 areas; validating native grids, band order and masks...', flush=True)
    export_inputs(RAW, PREPARED, batch_size=32, progress=lambda s: print(s, flush=True))

manifest = json.loads((PREPARED / 'manifest.json').read_text())
validation = json.loads((PREPARED / 'validation.json').read_text())
assert validation['passed'] and sha256(PREPARED / 'manifest.json') == validation['manifest_sha256']
assert manifest['sample_count'] == 1000 and manifest['metadata_sha256'] == report['metadata_sha256']
assert manifest['channel_profile']['optical'] == list(DEFAULT_PROFILE.optical)
assert manifest['channel_profile']['sar'] == list(DEFAULT_PROFILE.sar)
ids, splits = [], Counter()
for batch in manifest['batches']:
    folder = PREPARED / batch['directory']
    for name, digest in batch['sha256'].items():
        assert sha256(folder / name) == digest, str(folder / name)
    info = json.loads((folder / 'batch.json').read_text())
    n = batch['sample_count']
    tensors = torch.load(folder / 'croma_inputs.pt', map_location='cpu', weights_only=True)
    for key, channels in [('optical_images', 12), ('SAR_images', 2)]:
        value = tensors[key]
        assert value.shape == (n, channels, 120, 120) and value.dtype == torch.float32
        assert torch.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
    maps = torch.load(folder / 'reference_maps.pt', map_location='cpu', weights_only=True)
    assert maps.shape == (n,120,120) and maps.dtype == torch.int64
    ids.extend(s['patch_id'] for s in info['samples'])
    splits.update(s['split'] for s in info['samples'])
    del tensors, maps
assert ids == metadata.patch_id.tolist() and dict(splits) == report['counts']
stage_report = {
    'stage': 1, 'passed': True, 'sample_count': len(ids), 'splits': dict(splits),
    'optical_shape': [len(ids),12,120,120], 'sar_shape': [len(ids),2,120,120],
    'reference_map_shape': [len(ids),120,120], 'stored_batches': len(manifest['batches']),
    'prepared_manifest_sha256': sha256(PREPARED / 'manifest.json'),
    'download_report_sha256': sha256(SOURCE / 'download.json'),
    'output': str(PREPARED), 'excluded_areas': [],
}
(PIPELINE / 'stage-1-report.json').write_text(json.dumps(stage_report, indent=2) + '\n')
print('STAGE 1 COMPLETE:', json.dumps(stage_report), flush=True)
