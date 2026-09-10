# Fetch the official text-only source; no satellite imagery is downloaded again.
from pathlib import Path
import pyarrow.parquet as pq
from satquery.download_bigearthnet import download_file
REVISION='72d865f2146f0a85b720f7f3ca1cdbaeafc3d316'
SOURCE_SHA256='d3b97f999456016bb13c2a8e94b8f47825654f07a0394a6b266a38b750ca1554'
SOURCE_URL=f'https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/resolve/{REVISION}/BigEarthNet.txt.parquet'
TEXT_SOURCE=Path('/content/drive/MyDrive/SatQuery/annotation-source-cache')/REVISION/'BigEarthNet.txt.parquet'
download_file(SOURCE_URL,466819745,SOURCE_SHA256,TEXT_SOURCE,progress=lambda s:print(s,flush=True))
parquet=pq.ParquetFile(TEXT_SOURCE)
print('OFFICIAL TEXT SOURCE VERIFIED:',parquet.metadata.num_rows,'records',flush=True)
print(parquet.schema_arrow,flush=True)
