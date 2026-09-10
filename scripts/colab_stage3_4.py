# Stages 3–4: coverage targets and independent exhaustive spatial audit.
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.io import MemoryFile
from rasterio.windows import Window, bounds

from satquery.preprocessing import sha256
from satquery.targets import export_targets

P = Path('/content/drive/MyDrive/SatQuery/pipeline-1000')
SOURCE = P.parent / 'bigearthnet-v2-1000'
prepared, features, targets = P/'prepared', P/'features', P/'targets'
previous = json.loads((P/'stage-2-report.json').read_text())
assert previous['passed'] and sha256(features/'manifest.json') == previous['feature_manifest_sha256']
torch.set_num_threads(2)
if not targets.exists():
    export_targets(prepared, features, targets, progress=lambda s: print(s, flush=True))
pm, fm, tm = [json.loads((folder/'manifest.json').read_text()) for folder in (prepared,features,targets)]
validation = json.loads((targets/'validation.json').read_text())
assert validation['passed'] and validation['manifest_sha256'] == sha256(targets/'manifest.json')
assert tm['feature_manifest_sha256'] == sha256(features/'manifest.json')
assert tm['prepared_manifest_sha256'] == sha256(prepared/'manifest.json')

# Independent transcription of official v2 Table 1. Do not use the target module's lookup.
groups = [(111,112),(121,),(211,212,213),(221,222,223,241),(231,),(242,),(243,),
          (244,),(311,),(312,),(313,),(321,333),(322,323),(324,),(331,),(411,412),
          (421,422),(511,512),(521,522,523)]
excluded = {0,999,122,123,124,131,132,133,141,142,332,334,335,423}
lookup = {code:i for i,group in enumerate(groups) for code in group}
assert [c['clc_codes'] for c in tm['classes']] == [list(g) for g in groups]
original = {}
download = json.loads((SOURCE/'download.json').read_text())
for shard in download['shards']:
    archive = SOURCE/shard['archive']
    assert sha256(archive) == shard['sha256']
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith('_reference_map.tif'):
                patch = Path(name).name.removesuffix('_reference_map.tif')
                assert patch not in original
                with MemoryFile(z.read(name)) as memory, memory.open() as raster:
                    original[patch] = (raster.read(1), raster.transform, raster.crs, tuple(raster.bounds))
print('Read all original reference TIFFs:',len(original),flush=True)
assert len(original) == 1000
stats = {s:Counter() for s in ('train','validation','test')}
class_tokens = {s:np.zeros(19,dtype=np.int64) for s in stats}
class_pixels = {s:np.zeros(19,dtype=np.int64) for s in stats}
seen, sar_seen, checked = set(),set(),0
for pb,fb,tb in zip(pm['batches'],fm['batches'],tm['batches'],strict=True):
    for name,digest in tb['sha256'].items(): assert sha256(targets/tb['directory']/name)==digest
    info = json.loads((prepared/pb['directory']/'batch.json').read_text())
    fi = json.loads((features/fb['directory']/'batch.json').read_text())
    ti = json.loads((targets/tb['directory']/'batch.json').read_text())
    assert info['samples'] == fi['samples'] == ti['samples']
    maps = torch.load(prepared/pb['directory']/'reference_maps.pt',weights_only=True).numpy()
    values = {k:v.numpy() for k,v in torch.load(targets/tb['directory']/'targets.pt',weights_only=True).items()}
    for n,sample in enumerate(info['samples']):
        patch,split = sample['patch_id'],sample['split']
        assert patch not in seen and sample['s1_name'] not in sar_seen
        seen.add(patch); sar_seen.add(sample['s1_name'])
        source,transform,crs,area_bounds = original[patch]
        np.testing.assert_array_equal(source,maps[n])
        np.testing.assert_allclose(tuple(transform)[:6],sample['transform'],atol=0,rtol=0)
        np.testing.assert_allclose(area_bounds,sample['bounds'],atol=0,rtol=0)
        assert crs == rasterio.crs.CRS.from_user_input(sample['crs'])
        assert crs.is_projected and crs.linear_units_factor[1] == 1
        stats[split]['areas'] += 1
        for row in range(15):
            for col in range(15):
                token=row*15+col
                codes,counts=np.unique(source[row*8:row*8+8,col*8:col*8+8],return_counts=True)
                expected=np.zeros(19,dtype=np.int64); unknown=0
                for code,count in zip(codes,counts):
                    if int(code) in lookup: expected[lookup[int(code)]] += count
                    else:
                        assert int(code) in excluded
                        unknown += int(count)
                assert expected.sum()+unknown == 64
                np.testing.assert_array_equal(values['class_counts'][n,token],expected)
                np.testing.assert_array_equal(values['class_fractions'][n,token],expected/64)
                assert values['unlabeled_counts'][n,token] == unknown
                assert values['unlabeled_fraction'][n,token] == unknown/64
                assert values['fully_labeled_mask'][n,token] == (unknown==0)
                assert values['has_labels_mask'][n,token] == (unknown<64)
                ground=bounds(Window(col*8,row*8,8,8),transform)
                np.testing.assert_allclose(values['bounds'][n,token],ground,atol=0,rtol=0)
                assert ground[2]-ground[0] == ground[3]-ground[1] == 80
                stats[split]['fully_labeled' if unknown==0 else 'unlabeled' if unknown==64 else 'partially_labeled'] += 1
                if unknown==0:
                    class_tokens[split] += expected>0
                    class_pixels[split] += expected
                checked += 1
    print('Independently audited',len(seen),'areas /',checked,'tokens',flush=True)
assert len(seen)==1000 and checked==225000 and seen==set(original)
report={'stages':[3,4],'passed':True,'areas_checked':len(seen),'tokens_checked':checked,
        'class_counts_checked':checked*19,'original_reference_pixels_checked':1000*120*120,
        'ground_footprint_metres':[80,80],'token_order':'row-major',
        'fraction_denominator':64,'unlabeled_area_renormalized':False,
        'target_manifest_sha256':sha256(targets/'manifest.json'),
        'split_statistics':{s:dict(v) for s,v in stats.items()},
        'eligible_class_token_support':{s:v.tolist() for s,v in class_tokens.items()},
        'eligible_class_pixel_support':{s:v.tolist() for s,v in class_pixels.items()},
        'label_limitation':'Alignment verifies supplied CLC reference maps; it does not establish independent ground-truth accuracy at 80 metres.'}
(P/'stage-3-4-report.json').write_text(json.dumps(report,indent=2)+'\n')
del original,maps,values
print('STAGES 3–4 COMPLETE:',json.dumps(report),flush=True)
