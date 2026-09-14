from pathlib import Path
import json, zipfile, hashlib
from collections import Counter
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rasterio.io import MemoryFile
P = Path('/content/drive/MyDrive/SatQuery (1)/pipeline-5000')
D = P.parent/'bigearthnet-v2-5000'
A = P/'annotations/source/72d865f2146f0a85b720f7f3ca1cdbaeafc3d316/BigEarthNet.txt.parquet'
OUT = P/'annotation-map-audit/expanded-six'
OUT.mkdir(parents=True, exist_ok=True)
meta = pd.read_parquet(D/'metadata.parquet')
# Deterministic selection before inspecting any annotation answers.
candidates = meta[meta.split == 'test'].sort_values('patch_id')
chosen = pd.concat([g.sample(n=1,random_state=37) for _,g in candidates.groupby('country',sort=True)][:6])
ids = chosen.patch_id.tolist()
assert len(ids)==len(set(ids))==6
print('SELECTED', chosen[['patch_id','country','split','labels']].to_json(orient='records'),flush=True)
rows=[]
for batch in pq.ParquetFile(A).iter_batches(batch_size=131072):
    table=pa.Table.from_batches([batch])
    rows.extend(table.filter(pc.is_in(table['patch_id'],value_set=pa.array(ids))).to_pylist())
print('ANNOTATIONS',len(rows),dict(Counter(r['category'] for r in rows)),flush=True)
with zipfile.ZipFile(D/'Reference_Maps-selected.zip') as z:
    names={Path(n).name.removesuffix('_reference_map.tif'):n for n in z.namelist() if n.endswith('_reference_map.tif')}
    maps={}; map_info={}
    for pid in ids:
        raw=z.read(names[pid])
        with MemoryFile(raw) as mf, mf.open() as ds:
            maps[pid]=ds.read(1)
            codes,counts=np.unique(maps[pid],return_counts=True)
            map_info[pid]={'shape':list(maps[pid].shape),'pixel_area_m2':abs(ds.transform.a*ds.transform.e),'counts':dict(zip(map(str,codes),map(int,counts))),'sha256':hashlib.sha256(raw).hexdigest()}
np.savez_compressed(OUT/'reference-map-samples.npz',**maps)
(OUT/'source-records.json').write_text(json.dumps(rows,indent=2)+'\n')
(OUT/'map-counts.json').write_text(json.dumps(map_info,indent=2)+'\n')
for pid in ids:
    print('IMAGE',pid,'MAP',json.dumps(map_info[pid]),flush=True)
    for r in rows:
        if r['patch_id']==pid and (r['type']=='caption' or r['category'] in ('presence','referring_expression')):
            print('RECORD',json.dumps({k:r[k] for k in ['ID','type','category','input','output']}),flush=True)
print('SAMPLES_SAVED',str(OUT),flush=True)

import re, math
from scipy import ndimage
from satquery.targets import CLASSES
by_id={r['ID']:r for r in rows}
def class_mask(pid,name):
    return np.isin(maps[pid],dict(CLASSES)[name])
def metric(rid,name):
    r=by_id[rid];m=class_mask(r['patch_id'],name)
    return float(m.sum()*map_info[r['patch_id']]['pixel_area_m2']),float(m.mean()*100)
# Manually transcribed claims from the six displayed captions; compare independently below.
claims={17046:[('Arable land',1195000),('Coniferous forest',207000),(CLASSES[6][0],72000)],4618202:[('Broad-leaved forest',1196000),('Pastures',244000)],410073:[('Mixed forest',1013000),('Coniferous forest',295000),('Inland waters',132000)],695388:[('Pastures',1171000),(CLASSES[6][0],183000),('Arable land',86000)],2380958:[('Broad-leaved forest',1008000),('Transitional woodland, shrub',283000),(CLASSES[6][0],149000)],2970546:[('Mixed forest',1237000),('Pastures',141000),('Coniferous forest',72000)]}
caption_checks=[]
for rid, items in claims.items():
    for name, stated in items:
        area,pct=metric(rid,name)
        caption_checks.append({'id':rid,'class':name,'stated_m2':stated,'map_m2':area,'map_percent':pct,'rounded_match':round(area/1000)*1000==stated})
# Question semantics transcribed from source records; retain IDs for review.
area_specs=[(17036,CLASSES[6][0],'pct','<=',70,None),(17037,'Coniferous forest','pct','==',100,None),(17048,'Arable land','pct','between',80,90),(410063,'Inland waters','pct','>',0,None),(410064,'Inland waters','m2','between',288000,720000),(410075,'Mixed forest','m2','between',864000,1152000),(695375,'Arable land','pct','==',100,None),(695376,'Arable land','pct','>=',0,None),(695390,'Pastures','pct','between',70,90),(2380947,'Transitional woodland, shrub','pct','>',20,None),(2380948,'Transitional woodland, shrub','pct','<=',60,None),(2380960,'Transitional woodland, shrub','pct','between',10,40),(2970536,'Mixed forest','m2','<',1152000,None),(2970537,'Mixed forest','pct','>=',80,None),(2970548,'Pastures','pct','between',0,10),(4618194,'Broad-leaved forest','pct','between',0,20),(4618195,'Pastures','pct','<',20,None),(4618204,'Broad-leaved forest','pct','between',70,90)]
area_checks=[]
for rid,name,unit,op,lo,hi in area_specs:
    area,pct=metric(rid,name);v=area if unit=='m2' else pct
    truth={'<':lambda:v<lo,'<=':lambda:v<=lo,'>':lambda:v>lo,'>=':lambda:v>=lo,'==':lambda:v==lo,'between':lambda:lo<=v<=hi}[op]()
    expected=by_id[rid]['output']=='yes' if by_id[rid]['type']=='binary' else True
    area_checks.append({'id':rid,'class':name,'map_value':v,'unit':unit,'operator':op,'lower':lo,'upper':hi,'annotation_answer':by_id[rid]['output'],'matches':bool(truth==expected)})
presence_specs={17040:'Coniferous forest',17041:'Permanent crops',17052:'Arable land',4618198:'Broad-leaved forest',4618199:CLASSES[6][0],4618208:'Broad-leaved forest',410067:'Inland waters',410068:'Broad-leaved forest',410079:'Coniferous forest',695379:CLASSES[6][0],695380:'Permanent crops',695394:'Arable land',2380951:'Transitional woodland, shrub',2380952:'Complex cultivation patterns',2380964:'Transitional woodland, shrub',2970540:'Coniferous forest',2970541:'Broad-leaved forest',2970552:'Pastures'}
presence_checks=[]
for rid,name in presence_specs.items():
    area,_=metric(rid,name);r=by_id[rid];expected=r['output']=='yes' if r['type']=='binary' else True
    presence_checks.append({'id':rid,'class':name,'map_m2':area,'matches':bool((area>0)==expected)})
box_specs={17044:'Coniferous forest',17045:CLASSES[6][0],410071:'Coniferous forest',410072:'Inland waters',695384:'Arable land',695385:'Pastures',695386:CLASSES[6][0],695387:'Arable land',2380956:CLASSES[6][0],2380957:'Transitional woodland, shrub',2970544:'Coniferous forest',2970545:'Pastures',4618201:'Pastures'}
box_checks=[]
for rid,name in box_specs.items():
    r=by_id[rid];mask=class_mask(r['patch_id'],name);box=list(map(float,re.findall(r'[0-9]+(?:[.][0-9]+)?',r['output'])))
    assert len(box)==4
    xmin,ymin,xmax,ymax=np.array(box)*120
    wx=np.maximum(0,np.minimum(np.arange(120)+1,xmax)-np.maximum(np.arange(120),xmin));wy=np.maximum(0,np.minimum(np.arange(120)+1,ymax)-np.maximum(np.arange(120),ymin));w=wy[:,None]*wx[None,:]
    checks=[]
    for conn in (1,2):
        regions,n=ndimage.label(mask,structure=ndimage.generate_binary_structure(2,conn));found=[]
        for index,slices in enumerate(ndimage.find_objects(regions),1):
            if slices is None:continue
            yy,xx=slices;b=[xx.start,yy.start,xx.stop,yy.stop]
            norm=[math.floor(b[0]*100/120)/100,math.floor(b[1]*100/120)/100,math.ceil(b[2]*100/120)/100,math.ceil(b[3]*100/120)/100]
            size=int((regions==index).sum());found.append({'pixels':size,'box':norm,'error':float(np.max(np.abs(np.array(norm)-box)))})
        best=min(found,key=lambda f:f['error'])
        checks.append({'connectivity':4 if conn==1 else 8,'components':n,'best':best,'exact_box_match':best['error']<1e-9})
    box_checks.append({'id':rid,'class':name,'annotation_box':box,'class_percent_in_box':float((w*mask).sum()/w.sum()*100),'components':checks})
result={'scope':'6 deterministic test images, one per country; selection seed 37 before reading answers; not a dataset-wide validation','images':chosen[['patch_id','country','split']].to_dict(orient='records'),'source_records':len(rows),'caption_total_area_checks':caption_checks,'presence_checks':presence_checks,'area_checks':area_checks,'box_checks':box_checks,'not_map_only':'Country and season are metadata; climate zone uses an external climate map; caption wording is generated language.'}
(OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print('CHECK_COUNTS',json.dumps({'caption_total_areas':len(caption_checks),'caption_matches':sum(x['rounded_match'] for x in caption_checks),'presence':len(presence_checks),'presence_matches':sum(x['matches'] for x in presence_checks),'area_qa':len(area_checks),'area_qa_matches':sum(x['matches'] for x in area_checks),'boxes':len(box_checks),'exact_box_match_4':sum(x['components'][0]['exact_box_match'] for x in box_checks),'exact_box_match_8':sum(x['components'][1]['exact_box_match'] for x in box_checks)}))
print('CAPTIONS',json.dumps(caption_checks))
print('BOXES',json.dumps(box_checks))
print('QA_FAILURES',json.dumps([x for x in area_checks+presence_checks if not x['matches']]))

for c in caption_checks:
    c['python_ties_to_even_match']=c.pop('rounded_match')
    c['nearest_1000_consistent']=abs(c['stated_m2']-c['map_m2'])<=500
count_specs=[(410073,'Inland waters',3),(695388,'Pastures',2),(695388,'Arable land',2),(2380958,'Transitional woodland, shrub',6),(2970546,'Mixed forest',2)]
count_checks=[]
for rid,name,claimed in count_specs:
    mask=class_mask(by_id[rid]['patch_id'],name)
    n4=int(ndimage.label(mask)[1]);n8=int(ndimage.label(mask,structure=np.ones((3,3)))[1])
    count_checks.append({'id':rid,'class':name,'claimed_regions':claimed,'map_regions_4':n4,'map_regions_8':n8,'matches_4':claimed==n4})
result['caption_total_area_checks']=caption_checks
result['caption_component_count_checks']=count_checks
summary={'images':6,'countries':chosen.country.tolist(),'records_loaded':len(rows),'presence_matches':sum(x['matches'] for x in presence_checks),'presence_checks':len(presence_checks),'area_qa_matches':sum(x['matches'] for x in area_checks),'area_qa_checks':len(area_checks),'named_boxes_matching_components':sum(x['components'][0]['exact_box_match'] for x in box_checks),'named_boxes_checked':len(box_checks),'caption_area_claims_consistent_with_rounding':sum(x['nearest_1000_consistent'] for x in caption_checks),'caption_area_claims_checked':len(caption_checks),'caption_region_counts_matching':sum(x['matches_4'] for x in count_checks),'caption_region_counts_checked':len(count_checks),'box_class_percentage_min':min(x['class_percent_in_box'] for x in box_checks),'box_class_percentage_max':max(x['class_percent_in_box'] for x in box_checks),'caption_discrepancies':[x for x in caption_checks if not x['nearest_1000_consistent']]}
result['summary']=summary
result['rounding_policy']='Accept either direction for exact halfway cases when rounding to nearest 1000 square metres (absolute error <=500); Python round uses ties-to-even.'
result['scope_limits']='Presence/area answer claims and named-class box extents were checked. Point boxes, all adjacency/count/relative-position Q&A, and every caption phrase were not exhaustively checked. Climate metadata was not independently validated. This sample cannot prove all records are map-derived.'
(OUT/'comparison.json').write_text(json.dumps(result,indent=2))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
