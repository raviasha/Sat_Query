# Match the official text records to existing images, then link to saved features.
import json
from pathlib import Path
import pandas as pd
from satquery.match_annotations import match_annotations, SOURCE_REVISION, SOURCE_SHA256
from satquery.preprocessing import sha256
from satquery.prediction_data import checked_file

P=Path(globals().get('P','/content/drive/MyDrive/SatQuery/pipeline-1000'))
IMAGE_ROOT=Path(globals().get('IMAGE_ROOT',P.parent/'bigearthnet-v2-1000'))
TEXT_SOURCE=Path(globals().get('TEXT_SOURCE',P.parent/'annotation-source-cache'/SOURCE_REVISION/'BigEarthNet.txt.parquet'))
OUTPUT=Path(globals().get('OUTPUT',P/'annotations'))
assert sha256(TEXT_SOURCE)==SOURCE_SHA256, 'Text source checksum mismatch'
download=json.loads((IMAGE_ROOT/'download.json').read_text())
assert sha256(IMAGE_ROOT/'metadata.parquet')==download['metadata_sha256']
metadata=pd.read_parquet(IMAGE_ROOT/'metadata.parquet')
assert len(metadata)>0
if not OUTPUT.exists():
    match_annotations(IMAGE_ROOT/'metadata.parquet',TEXT_SOURCE,OUTPUT,
                      source_revision=SOURCE_REVISION,expected_source_sha256=SOURCE_SHA256,
                      progress=lambda s:print(s,flush=True))
report=json.loads((OUTPUT/'report.json').read_text())
assert report['source_sha256']==SOURCE_SHA256
assert report['selected_metadata_sha256']==download['metadata_sha256']
for name,digest in report['files'].items(): assert sha256(OUTPUT/name)==digest
features=P/'features'
fm=json.loads((features/'manifest.json').read_text())
previous=json.loads((P/'stage-2-report.json').read_text())
assert sha256(features/'manifest.json')==previous['feature_manifest_sha256']
scenes=[json.loads(line) for line in (OUTPUT/'image-annotations.jsonl').read_text().splitlines()]
assert [s['patch_id'] for s in scenes]==metadata.patch_id.tolist()
links=[]
for batch in fm['batches']:
    info=json.loads(checked_file(features.resolve(),batch,'batch.json'))
    for offset,sample in enumerate(info['samples']):
        index=batch['start_index']+offset
        scene=scenes[index]
        assert scene['image_index']==index
        assert scene['patch_id']==sample['patch_id'] and scene['s1_name']==sample['s1_name']
        assert scene['image_split']==sample['split']
        links.append({k:scene[k] for k in ['patch_id','s1_name','image_index','image_split','annotation_splits','annotation_count','use_partition']} |
                     {'feature_batch':batch['directory'],'index_in_feature_batch':offset,
                      'feature_file_sha256':batch['sha256']['features.pt'],'feature_manifest_sha256':previous['feature_manifest_sha256']})
assert len(links)==len(metadata)
assert [link['image_index'] for link in links]==list(range(len(metadata)))
linkfile=OUTPUT/'feature-links.jsonl'
text=''.join(json.dumps(row)+'\n' for row in links)
if linkfile.exists(): assert linkfile.read_text()==text
else: linkfile.write_text(text)
receipt={**report,'feature_manifest_sha256':previous['feature_manifest_sha256'],
         'feature_link_count':len(links),'feature_links_sha256':sha256(linkfile),
         'matched_report_sha256':sha256(OUTPUT/'report.json'),
         'matched_output_bytes':sum(p.stat().st_size for p in OUTPUT.iterdir() if p.is_file()),
         'output':str(OUTPUT)}
(P/'stage-8-annotations-report.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('ANNOTATIONS MATCHED:',json.dumps(receipt),flush=True)
example=next((s for s in scenes if s['annotations']),None)
if example:
    print('EXAMPLE:',json.dumps({'patch_id':example['patch_id'],'s1_name':example['s1_name'],
                           'image_split':example['image_split'],'annotation_count':example['annotation_count'],
                           'annotations':[{k:r[k] for k in ['ID','input','output','type','category','split']} for r in example['annotations'][:3]]}),flush=True)
