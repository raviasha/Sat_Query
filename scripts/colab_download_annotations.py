# Fetch the official text-only source; no satellite imagery is downloaded again.
from pathlib import Path
import json
import pyarrow.parquet as pq
from satquery.download_bigearthnet import download_file
from satquery.match_annotations import SOURCE_REVISION, SOURCE_SHA256, SOURCE_URL, SOURCE_SIZE
TEXT_SOURCE=Path(globals().get('TEXT_SOURCE',Path('/content/drive/MyDrive/SatQuery/annotation-source-cache')/SOURCE_REVISION/'BigEarthNet.txt.parquet'))
download_file(SOURCE_URL,SOURCE_SIZE,SOURCE_SHA256,TEXT_SOURCE,progress=lambda s:print(s,flush=True))
parquet=pq.ParquetFile(TEXT_SOURCE)
assert parquet.metadata.num_rows==9553962
assert {'ID','patch_id','s1_name','input','output','type','split'}.issubset(parquet.schema_arrow.names)
receipt={'source_url':SOURCE_URL,'revision':SOURCE_REVISION,'sha256':SOURCE_SHA256,
         'bytes':TEXT_SOURCE.stat().st_size,'records':parquet.metadata.num_rows,
         'columns':parquet.schema_arrow.names,'path':str(TEXT_SOURCE)}
(TEXT_SOURCE.parent/'download-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('OFFICIAL TEXT SOURCE VERIFIED:',parquet.metadata.num_rows,'records',flush=True)
print(parquet.schema_arrow,flush=True)
