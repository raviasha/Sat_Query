# Stage 5: train on train areas and choose an epoch using validation areas.
import json
import shutil
import tempfile
from pathlib import Path

import torch

from satquery.evaluation import load_splits, predict_rows
from satquery.prediction import save_head, load_head
from satquery.preprocessing import sha256
from satquery.validation_training import fit_with_validation

P=Path('/content/drive/MyDrive/SatQuery/pipeline-1000')
audit=json.loads((P/'stage-3-4-report.json').read_text())
assert audit['passed'] and audit['tokens_checked']==225000
assert sha256(P/'targets'/'manifest.json')==audit['target_manifest_sha256']
torch.set_num_threads(2)
splits, provenance=load_splits(P/'features',P/'targets',progress=lambda s:print(s,flush=True))
assert provenance['input_sample_splits']=={'train':600,'validation':200,'test':200}
assert all(not(set(splits[a].patch_ids)&set(splits[b].patch_ids)) for a,b in [('train','validation'),('train','test'),('validation','test')])
print('Eligible token counts:',provenance['eligible_tokens_by_split'],flush=True)
output=P/'model'
if not output.exists():
    model,fit=fit_with_validation(splits['train'],splits['validation'],device='cuda' if torch.cuda.is_available() else 'cpu',
                                  max_epochs=60,patience=10,learning_rate=.001,batch_size=1024,seed=17,
                                  progress=lambda s:print(s,flush=True))
    metadata={**provenance,**{k:v for k,v in fit.items() if k!='history'},
              'training_mode':'train_split_only','evaluation_scope':'validation_selection_only',
              'loss':'unweighted_soft_target_cross_entropy','torch_version':str(torch.__version__)}
    temporary=Path(tempfile.mkdtemp(prefix='.model-',dir=P))
    try:
        save_head(temporary/'head.pt',model,metadata)
        restored,_=load_head(temporary/'head.pt')
        torch.testing.assert_close(predict_rows(restored,splits['validation'].x[:256]),
                                   predict_rows(model,splits['validation'].x[:256]),rtol=0,atol=0)
        report={**metadata,'history':fit['history'],'checkpoint_sha256':sha256(temporary/'head.pt'),
                'training_mean_class_fractions':splits['train'].y.mean(0).tolist()}
        (temporary/'training.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary);raise
else:
    report=json.loads((output/'training.json').read_text())
    assert report['feature_manifest_sha256']==provenance['feature_manifest_sha256']
    assert report['target_manifest_sha256']==provenance['target_manifest_sha256']
    assert report['checkpoint_sha256']==sha256(output/'head.pt')
stage5={'stage':5,'passed':True,'selected_epoch':report['selected_epoch'],
        'epochs_run':len(report['history']),'best_validation_loss':report['best_validation_loss'],
        'checkpoint_sha256':report['checkpoint_sha256'],'train_areas':600,'validation_areas':200,
        'test_used_for_selection':False,'eligible_tokens_by_split':provenance['eligible_tokens_by_split']}
(P/'stage-5-report.json').write_text(json.dumps(stage5,indent=2)+'\n')
if torch.cuda.is_available():torch.cuda.empty_cache()
print('STAGE 5 COMPLETE:',json.dumps(stage5),flush=True)
