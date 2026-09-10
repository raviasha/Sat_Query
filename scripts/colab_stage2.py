# Stage 2 — frozen CROMA feature extraction
import json
from pathlib import Path
import torch
from satquery.features import CromaFeatureExtractor, extract_features
from satquery.preprocessing import sha256

PIPELINE = Path('/content/drive/MyDrive/SatQuery/pipeline-1000')
PREPARED, FEATURES = PIPELINE / 'prepared', PIPELINE / 'features'
stage1 = json.loads((PIPELINE / 'stage-1-report.json').read_text())
assert stage1['passed'] and stage1['sample_count'] == 1000
assert sha256(PREPARED / 'manifest.json') == stage1['prepared_manifest_sha256']
torch.set_num_threads(2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('CROMA device:', device, flush=True)
if not FEATURES.exists():
    extractor = CromaFeatureExtractor.from_pretrained(cache_dir=PIPELINE / 'croma-checkpoint', device=device)
    extract_features(PREPARED, FEATURES, extractor, batch_size=8 if device == 'cuda' else 2,
                     progress=lambda s: print(s, flush=True))
    del extractor
    if device == 'cuda': torch.cuda.empty_cache()
manifest = json.loads((FEATURES / 'manifest.json').read_text())
validation = json.loads((FEATURES / 'validation.json').read_text())
assert validation['passed'] and manifest['sample_count'] == 1000
assert sha256(FEATURES / 'manifest.json') == validation['manifest_sha256']
assert manifest['source_manifest_sha256'] == stage1['prepared_manifest_sha256']
for batch in manifest['batches']:
    for name, digest in batch['sha256'].items():
        assert sha256(FEATURES / batch['directory'] / name) == digest
stage2 = {'stage':2,'passed':True,'sample_count':1000,'tokens_per_area':225,
          'feature_dimension':768,'spatial_features':['optical_encodings','SAR_encodings','joint_encodings'],
          'feature_manifest_sha256':sha256(FEATURES / 'manifest.json'),'device':manifest['device'],
          'checkpoint_sha256':manifest['model']['checkpoint_sha256'],'output':str(FEATURES)}
(PIPELINE / 'stage-2-report.json').write_text(json.dumps(stage2, indent=2)+'\n')
print('STAGE 2 COMPLETE:', json.dumps(stage2), flush=True)
