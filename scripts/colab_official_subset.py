"""Select 5,000 areas from downloaded official parts, preserving baseline splits."""
import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import zstandard

from satquery.archive_http import HTTPArchiveReader
from satquery.download_bigearthnet import METADATA_SHA256, select_records
from satquery.official_parts import PartsReader, SelectedTIFFCheckpoints, extract_selected_tar
from satquery.preprocessing import DEFAULT_PROFILE, sha256

HOME = Path('/content/drive/MyDrive/SatQuery')
ARCHIVES = HOME/'bigearthnet-v2-full-official'
SOURCE = HOME/'bigearthnet-v2-5000'
RAW = Path('/content/satquery-official-selected')
SOURCE.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
COUNTS = {'train':4600,'validation':200,'test':200}
with PartsReader(ARCHIVES/'metadata.parquet.parts') as stream:
    raw_metadata = stream.read()
assert hashlib.sha256(raw_metadata).hexdigest() == METADATA_SHA256
metadata = pd.read_parquet(io.BytesIO(raw_metadata))
selected = select_records(metadata, COUNTS, seed=17)
old = select_records(metadata, {'train':600,'validation':200,'test':200}, seed=17)
for split in ('validation','test'):
    assert selected[selected.split==split].patch_id.tolist() == old[old.split==split].patch_id.tolist()
assert set(old[old.split=='train'].patch_id) <= set(selected[selected.split=='train'].patch_id)
config={'counts':COUNTS,'seed':17,'patch_ids':selected.patch_id.tolist(),
        'baseline_ids':{s:old[old.split==s].patch_id.tolist() for s in COUNTS},
        'source':'official downloaded archives','metadata_source_sha256':METADATA_SHA256}
selection=SOURCE/'selection.json'
if selection.exists(): assert json.loads(selection.read_text()) == config
else: selection.write_text(json.dumps(config,indent=2))
selected.to_parquet(SOURCE/'metadata.parquet',index=False)
print('Selection fixed: 4600 train / 200 validation / 200 test; original evaluation IDs unchanged.',flush=True)

# Cache each extracted modality as one small ZIP, so a new runtime can reuse it.
for sensor, archive_name, column, bands, trim in [
    ('Reference_Maps','Reference_Maps.tar.zst','patch_id',('reference_map',),2),
    ('BigEarthNet-S1','BigEarthNet-S1.tar.zst','s1_name',DEFAULT_PROFILE.sar,3),
    ('BigEarthNet-S2','BigEarthNet-S2.tar.zst','patch_id',DEFAULT_PROFILE.optical,2),
]:
    wanted={f'{identifier}_{band}.tif':f'{sensor}/{identifier.rsplit("_",trim)[0]}/{identifier}/{identifier}_{band}.tif'
            for identifier in selected[column] for band in bands}
    cached=SOURCE/(sensor+'-selected.zip')
    receipt=cached.with_suffix('.json')
    if cached.exists() and receipt.exists():
        info=json.loads(receipt.read_text())
        assert info['selection_sha256']==sha256(selection) and info['sha256']==sha256(cached)
        with zipfile.ZipFile(cached) as z:
            assert set(z.namelist())==set(wanted.values())
            z.extractall(RAW)
        print('Reused verified',sensor,'selection',flush=True)
        continue
    print('Streaming official archive:',archive_name,flush=True)
    # Optical Drive reads repeatedly disconnect. Stream identical official bytes
    # directly; completed modality caches still take precedence above.
    transport = 'official HTTP ranges' if sensor == 'BigEarthNet-S2' else 'verified Drive parts'
    checkpoints = None
    restored = set()
    if sensor == 'BigEarthNet-S2':
        checkpoints = SelectedTIFFCheckpoints(
            SOURCE/'optical-tiff-checkpoints', RAW, wanted,
            identity=sha256(selection), batch_size=1000,
        )
        restored = checkpoints.restore()
        recovery = SOURCE/'BigEarthNet-S2-partial-recovery.zip'
        recovery_receipt = recovery.with_suffix('.json')
        if recovery.exists() and recovery_receipt.exists():
            recovery_info = json.loads(recovery_receipt.read_text())
            assert recovery_info['sha256'] == sha256(recovery)
            with zipfile.ZipFile(recovery) as package:
                recovery_paths = set(package.namelist())
                assert recovery_paths <= set(wanted.values())
                package.extractall(RAW)
            recovery_names = {name for name, relative in wanted.items() if relative in recovery_paths}
            checkpoints.record(restored | recovery_names, final=True)
            restored |= recovery_names
            print('Imported recovery snapshot:',len(recovery_names),'TIFFs',flush=True)
        print('Restored optical TIFF checkpoints:',len(restored),'/',len(wanted),flush=True)
    reader = (HTTPArchiveReader(
        'https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1',
        63251710377, '2245ed2d1a93f6ce637d839bc856396e',
        chunk_size=64*1024*1024,
        cache_folder=SOURCE/'optical-archive-chunks',
        progress=lambda s: print(s, flush=True),
    ) if sensor == 'BigEarthNet-S2' else PartsReader(
        ARCHIVES/(archive_name+'.parts'), progress=lambda s: print(s, flush=True),
    ))
    print('Archive transport:', transport, flush=True)
    missing = {name: relative for name, relative in wanted.items() if name not in restored}
    with reader as source:
        with zstandard.ZstdDecompressor().stream_reader(source,closefd=False) as decoded:
            newly_seen = extract_selected_tar(
                decoded, missing, RAW, progress=lambda s:print(s,flush=True),
                on_selected=(lambda seen, cache=checkpoints, prior=frozenset(restored):
                             cache.record(prior | seen)) if checkpoints else None,
            )
            if checkpoints:
                checkpoints.record(restored | newly_seen, final=True)
            while decoded.read(8*1024*1024): pass
        while source.read(8*1024*1024): pass
        assert source.finished
    local=Path('/content')/(sensor+'-selected.zip')
    with zipfile.ZipFile(local,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for rel in sorted(wanted.values()): z.write(RAW/rel,rel)
    digest=sha256(local)
    pending=cached.with_suffix('.partial')
    shutil.copyfile(local,pending)
    assert sha256(pending)==digest
    pending.replace(cached)
    receipt.write_text(json.dumps({'sha256':digest,'selection_sha256':sha256(selection),'file_count':len(wanted),'transport':transport}))
    local.unlink()
    print('Saved verified selected modality:',sensor,flush=True)

# Existing pipeline stages consume the same paired ZIP contract as before.
shards=[]
for start in range(0,len(selected),100):
    part=selected.iloc[start:start+100]
    name=f'shard-{start//100:06d}.zip'
    dest=SOURCE/name
    receipt=dest.with_suffix('.json')
    if receipt.exists():
        info=json.loads(receipt.read_text())
        assert info['patch_ids']==part.patch_id.tolist() and sha256(dest)==info['sha256']
    else:
        local=Path('/content')/name
        with zipfile.ZipFile(local,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for row in part.itertuples():
                for sensor,identifier,trim in [('BigEarthNet-S2',row.patch_id,2),('BigEarthNet-S1',row.s1_name,3),('Reference_Maps',row.patch_id,2)]:
                    folder=RAW/sensor/identifier.rsplit('_',trim)[0]/identifier
                    for path in sorted(folder.glob('*.tif')): z.write(path,path.relative_to(RAW))
            with tempfile.TemporaryDirectory() as td:
                path=Path(td)/'metadata.parquet';part.to_parquet(path,index=False);z.write(path,'metadata.parquet')
        digest=sha256(local)
        pending=dest.with_suffix('.partial');shutil.copyfile(local,pending)
        assert sha256(pending)==digest
        pending.replace(dest);local.unlink()
        info={'archive':name,'sha256':digest,'sample_count':len(part),'patch_ids':part.patch_id.tolist()}
        receipt.write_text(json.dumps(info))
    shards.append(info)
    print('Prepared paired shard',len(shards),'/50',flush=True)
report={'complete':True,'sample_count':5000,'counts':COUNTS,'metadata_sha256':sha256(SOURCE/'metadata.parquet'),
        'selection_sha256':sha256(selection),'shards':shards,'source':'official archives'}
(SOURCE/'download.json').write_text(json.dumps(report,indent=2))
print('OFFICIAL SUBSET COMPLETE',flush=True)
