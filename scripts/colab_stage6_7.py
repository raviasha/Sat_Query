# Stages 6–7: one final held-out test evaluation and per-class historical reliability.
import csv
import json
from pathlib import Path
import numpy as np
import torch
from satquery.evaluation import load_splits, predict_rows, coverage_metrics
from satquery.prediction import load_head
from satquery.preprocessing import sha256
from satquery.prediction_data import class_schema
from satquery.predict_cover import export_predictions

P=Path('/content/drive/MyDrive/SatQuery/pipeline-1000')
previous=json.loads((P/'stage-5-report.json').read_text())
assert previous['passed'] and sha256(P/'model'/'head.pt')==previous['checkpoint_sha256']
model,meta=load_head(P/'model'/'head.pt')
assert meta['feature_manifest_sha256']==sha256(P/'features'/'manifest.json')
assert meta['target_manifest_sha256']==sha256(P/'targets'/'manifest.json')
if 'splits' not in globals():
    splits,provenance=load_splits(P/'features',P/'targets',progress=lambda s:print(s,flush=True))
assert provenance['feature_manifest_sha256']==meta['feature_manifest_sha256']
assert provenance['target_manifest_sha256']==meta['target_manifest_sha256']
assert splits['test'].split=='test' and len(splits['test'].patch_ids)==200
output=P/'evaluation'
output.mkdir(exist_ok=True)
report_path=output/'test-report.json'
if not report_path.exists():
    test=splits['test']
    pred=predict_rows(model,test.x)
    metrics=coverage_metrics(pred.numpy(),test.y.numpy(),test.area_ids,bootstrap_repeats=1000,seed=17)
    train_report=json.loads((P/'model'/'training.json').read_text())
    baseline=np.broadcast_to(np.array(train_report['training_mean_class_fractions']),test.y.shape)
    baseline_metrics=coverage_metrics(baseline,test.y.numpy(),test.area_ids,bootstrap_repeats=1000,seed=17)
    report={'format_version':1,'checkpoint_sha256':sha256(P/'model'/'head.pt'),
            'feature_manifest_sha256':meta['feature_manifest_sha256'],'target_manifest_sha256':meta['target_manifest_sha256'],
            'evaluation_scope':'held_out_test_areas','official_test_areas':200,
            'eligible_tokens_only':True,'selected_epoch':meta['selected_epoch'],
            'metrics':metrics,'training_mean_baseline':baseline_metrics}
    prediction_path=output/'test-predictions.pt'
    pending=output/'.test-predictions.pt.tmp'
    torch.save({'predicted_fractions':pred,'true_fractions':test.y,'area_ids':test.area_ids.tolist()},pending)
    restored=torch.load(pending,weights_only=True)
    assert torch.equal(restored['predicted_fractions'],pred)
    assert torch.equal(restored['true_fractions'],test.y) and restored['area_ids']==test.area_ids.tolist()
    pending.replace(prediction_path)
    report['test_predictions_sha256']=sha256(prediction_path)
    pending_report=output/'.test-report.json.tmp'
    pending_report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    pending_report.replace(report_path)
else:
    report=json.loads(report_path.read_text())
    assert report['checkpoint_sha256']==sha256(P/'model'/'head.pt')
    metrics=report['metrics']
assert report['test_predictions_sha256']==sha256(output/'test-predictions.pt')
saved=torch.load(output/'test-predictions.pt',weights_only=True)
assert saved['area_ids']==splits['test'].area_ids.tolist()
assert torch.equal(saved['true_fractions'],splits['test'].y)
assert saved['predicted_fractions'].shape==splits['test'].y.shape and torch.isfinite(saved['predicted_fractions']).all()
del saved
reliability={'format_version':1,'checkpoint_sha256':report['checkpoint_sha256'],
             'feature_contract':meta['feature_contract'],'classes':class_schema(),
             'evaluation_scope':'held_out_test_areas','prediction_confidence_available':False,
             'test_report_sha256':sha256(report_path),'class_metrics':metrics['classes'],
             'note':'Show coverage prediction separately from historical test MAE, present-class MAE and area support. These are not calibrated probabilities or prediction intervals.'}
(output/'reliability.json').write_text(json.dumps(reliability,indent=2,allow_nan=False)+'\n')
fields=['index','name','mae_pp','rmse_pp','bias_pp','present_mae_pp','present_tokens','present_areas',
        'within_5pp_fraction','present_within_5pp_fraction','dominant_precision','dominant_recall','dominant_f1','support_status']
with (output/'per-class-metrics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(metrics['classes'])
print('HELD-OUT TEST:',json.dumps({k:v for k,v in metrics.items() if k not in ('classes','dominant_confusion_matrix')}),flush=True)
print('BASELINE MAE:',report['training_mean_baseline']['coverage_mae_pp'],flush=True)
for c in metrics['classes']:
    print(c['name'],': MAE',round(c['mae_pp'],2),'pp; present MAE',None if c['present_mae_pp'] is None else round(c['present_mae_pp'],2),'pp; areas',c['present_areas'],flush=True)
# Exercise the inference module and attach the matching reliability artifact.
if not (P/'predictions').exists():
    export_predictions(P/'features',P/'model'/'head.pt',P/'predictions',batch_size=2048,
                       reliability=output/'reliability.json',progress=lambda s:print(s,flush=True))
validation=json.loads((P/'predictions'/'validation.json').read_text())
assert validation['passed'] and validation['manifest_sha256']==sha256(P/'predictions'/'manifest.json')
pm=json.loads((P/'predictions'/'manifest.json').read_text())
assert pm['checkpoint_sha256']==report['checkpoint_sha256']
for batch in pm['batches']:
    for name,digest in batch['sha256'].items(): assert sha256(P/'predictions'/batch['directory']/name)==digest
assert pm['historical_class_reliability']['sha256']==sha256(P/'predictions'/'reliability.json')
stage={'stages':[6,7],'passed':True,'test_area_count':200,'eligible_test_tokens':metrics['token_count'],
       'test_coverage_mae_pp':metrics['coverage_mae_pp'],'baseline_coverage_mae_pp':report['training_mean_baseline']['coverage_mae_pp'],
       'test_dominant_accuracy':metrics['dominant_class_accuracy'],'class_count':19,
       'test_report_sha256':sha256(report_path),'per_class_csv':str(output/'per-class-metrics.csv'),
       'reliability_report':str(output/'reliability.json'),'inference_areas_verified':pm['sample_count']}
(P/'stage-6-7-report.json').write_text(json.dumps(stage,indent=2,allow_nan=False)+'\n')
print('STAGES 6–7 COMPLETE:',json.dumps(stage),flush=True)
