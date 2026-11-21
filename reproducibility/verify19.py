from pathlib import Path
import json,hashlib,itertools,sys,argparse,zipfile
import numpy as np,pandas as pd
from scipy.special import softmax,expit,logsumexp
from sklearn.metrics import f1_score,roc_auc_score,balanced_accuracy_score,log_loss
parser=argparse.ArgumentParser(description='Independent numerical audit of completed PaddyDoctor notebook 19 outputs')
parser.add_argument('--results',default='results19_review/19_PaddyDoctor_RESULTADOS.zip')
parser.add_argument('--code',default='paddy19_fix_work/code')
parser.add_argument('--out',default='revision7_work')
args=parser.parse_args();W=Path(args.out);W.mkdir(exist_ok=True,parents=True);C=Path(args.code);I=C/'inputs'
source=Path(args.results)
if source.is_file():
 R=W/'audit_extracted'
 with zipfile.ZipFile(source) as z:
  for name in z.namelist():
   if '..' in Path(name).parts or Path(name).is_absolute():raise ValueError('Unsafe archive path')
  z.extractall(R)
else:R=source
CL=['blast','dead_heart','hispa'];SEEDS=[2026,2027,2028]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
checks=[];maxerr=0.;hashed=0;rec=[];models=sorted(read(R/'P6/experiment_plan.json')['models']);records=[]
meta=pd.read_csv(I/'nearclean.csv').set_index('image_id');designs={}
for protocol,domains in [('P6',['ADT45','OTHER_balanced']),('P7',['AndraPonni','REST'])]:
 d=pd.read_csv(R/'design'/f'{protocol}_assignments.csv');designs[protocol]=d
 assert sha(R/'design'/f'{protocol}_assignments.csv')==sha(I/'frozen_designs'/f'{protocol}_assignments.csv')
 assert d.image_id.nunique()==len(d)==(480 if protocol=='P6' else 300) and d.sha256.is_unique
 assert d.groupby('matched_pair_id').size().eq(2).all()
 assert d.groupby('matched_pair_id').pair_split.nunique().eq(1).all()
 assert d.groupby(['label','age','pair_split','domain']).size().unstack(fill_value=0).nunique(axis=1).eq(1).all()
 assert not d.dropna(subset=['visual_group_id']).groupby('visual_group_id').pair_split.nunique().gt(1).any()
 for col in ['sha256','label','age','variety']:assert np.array_equal(meta.loc[d.image_id,col].to_numpy(),d[col].to_numpy())
 if protocol=='P6':assert d[d.domain.eq('OTHER_balanced')].groupby(['pair_split','label','variety']).size().unstack().nunique(axis=1).eq(1).all()
 else:assert d[d.domain.eq('REST')].variety.ne('AndraPonni').all()
 plan=read(R/protocol/'experiment_plan.json');assert plan['source_sha256']==sha(C/'paddy19_cultivar_extensions.py') and plan['core_sha256']==sha(C/'paddy19_training_core.py')
 assert plan['assignments_sha256']==sha(R/'design'/f'{protocol}_assignments.csv')
 markers=list((R/protocol/'runs').rglob('COMPLETE.json'));assert len(markers)==24
 seen=set()
 for marker in markers:
  run=marker.parent;prov=read(run/'provenance.json');key=(prov['model_key'],prov['train_seed'],prov['train_domain']);assert key not in seen;seen.add(key)
  for item in read(marker)['files']:
   if item['area']=='results':assert sha(run/item['path'])==item['sha256'];hashed+=1
  history=pd.read_csv(run/'history.csv');summary=read(run/'training_summary.json')
  assert history.epoch.tolist()==list(range(1,31)) and summary['epochs_completed']==30
  best=history.loc[history.val_macro_f1.idxmax()];assert int(best.epoch)==summary['best_epoch'] and np.isclose(best.val_macro_f1,summary['best_val_macro_f1'],atol=1e-12)
  records.append(dict(protocol=protocol,model=key[0],seed=key[1],source=key[2],wall_seconds=summary['wall_seconds'],selected_epoch=summary['best_epoch']))
  for target in domains:
   evaldir=run/f'eval__{target}';df=pd.read_csv(evaldir/'predictions.csv');truth=d[d.domain.eq(target)&d.pair_split.eq('test')].sort_values('image_id');df=df.sort_values('image_id')
   assert df.image_id.tolist()==truth.image_id.tolist() and df.label.tolist()==truth.label.tolist()
   metrics=read(evaldir/'metrics.json');y=np.array([CL.index(c) for c in df.label]);z=df[[f'logit__{c}' for c in CL]].to_numpy();temp=read(run/'temperature.json')['temperature']
   for cal,T in [('raw',1.),('TS',temp)]:
    p=softmax(z/T,axis=1);assert np.array_equal([CL[i] for i in p.argmax(1)],df[f'y_pred_{cal}'].to_numpy())
    calculated={'macro_f1':f1_score(y,p.argmax(1),labels=range(3),average='macro',zero_division=0),'nll':np.mean(logsumexp(z/T,axis=1)-np.take_along_axis(z/T,y[:,None],axis=1).ravel()),'brier':np.mean(((p-np.eye(3)[y])**2).sum(1))}
    for metric,value in calculated.items():
     err=abs(metrics[cal][metric]-value);maxerr=max(maxerr,err);assert err<1e-10
   rec.append((protocol,target,key[0],key[1],key[2],df))
 assert seen==set(itertools.product(models,SEEDS,domains))
 checks.append(protocol+' frozen split, balanced support, pairs, group isolation, 24 runs and all 30 epochs')
checks.append(f'{hashed} result-file hashes match COMPLETE receipts; 96 evaluations and probability metrics agree')
# Independently rebuild mean seed losses and paired stratified percentile bounds.
summary=pd.read_csv(R/'19_paired_image_intervals.csv');new=[]
for row in summary.itertuples():
 selected=[r for r in rec if r[:3]==(row.protocol,row.target,row.model)]
 first=selected[0][-1];y=first.label.to_numpy();rng=np.random.default_rng(19100+(row.protocol=='P7'))
 ix=np.concatenate([rng.choice(np.where(y==c)[0],size=(5000,int((y==c).sum())),replace=True) for c in CL],axis=1)
 losses=[];points=[];controls=[];transfers=[]
 def batch_f1(y,pred):
  truth=np.array([CL.index(c) for c in y])[ix];predicted=np.array([CL.index(c) for c in pred])[ix]
  vals=[]
  for k in range(3):
   tp=((truth==k)&(predicted==k)).sum(1);den=(truth==k).sum(1)+(predicted==k).sum(1)
   vals.append(np.divide(2*tp,den,out=np.zeros(len(ix)),where=den>0))
  return np.mean(vals,axis=0)
 for seed in SEEDS:
  c=next(t[-1] for t in selected if t[3]==seed and t[4]==row.target);t=next(t[-1] for t in selected if t[3]==seed and t[4]!=row.target)
  assert c.image_id.tolist()==t.image_id.tolist()==first.image_id.tolist()
  cf=f1_score(y,c.y_pred_raw,labels=CL,average='macro',zero_division=0);tf=f1_score(y,t.y_pred_raw,labels=CL,average='macro',zero_division=0)
  controls.append(cf);transfers.append(tf);points.append(cf-tf);losses.append(batch_f1(y,c.y_pred_raw)-batch_f1(y,t.y_pred_raw))
 lo,hi=np.quantile(np.mean(losses,axis=0),[.025,.975]);vals=[np.mean(points),lo,hi];old=[row.mean_seed_loss,row.image_ci95_lower,row.image_ci95_upper]
 assert np.max(np.abs(np.array(vals)-old))<1e-12
 new.append(dict(protocol=row.protocol,target=row.target,model=row.model,control_f1=np.mean(controls),transfer_f1=np.mean(transfers),loss=vals[0],lower=lo,upper=hi,n_test=len(y)))
pd.DataFrame(new).to_csv(W/'19_verified_f1_summary.csv',index=False);checks.append('All 16 paired F1 losses and 5000-replicate intervals independently reproduced')
# Rebuild domain test probabilities and metrics from frozen features and coefficients.
featuresfile=R/'embeddings/frozen_resnet_features.npz';assert sha(featuresfile)==read(R/'embeddings/features_sha256.json')['sha256']
a=np.load(featuresfile,allow_pickle=False);features=a['features'];ids=a['image_ids'];assert features.shape==(len(ids),2048) and len(set(ids))==len(ids) and np.isfinite(features).all();loc={i:j for j,i in enumerate(ids)}
emb=pd.read_csv(R/'embeddings/19_domain_embedding_metrics.csv');pred=pd.read_csv(R/'embeddings/19_domain_embedding_predictions.csv');p4=pd.read_csv(I/'P4.csv').rename(columns={'domain_split':'pair_split'});p4=p4[p4.label.isin(['blast','dead_heart','downy_mildew','hispa','normal'])];p5=pd.read_csv(I/'P5.csv');specs={'P4_same5':p4,'P5':p5}
for row in emb.itertuples():
 d=specs[row.protocol].sort_values('image_id').reset_index(drop=True);x=features[[loc[i] for i in d.image_id]];tr=d.pair_split.eq('train');te=d.pair_split.eq('test');coeff=np.load(R/'embeddings'/f'{row.protocol}_domain_classifier.npz')
 assert np.allclose(x[tr].mean(0),coeff['scaler_mean'],atol=1e-12)
 # sklearn StandardScaler transforms float32 features in place, rounding each operation.
 xx=x[te].copy();xx-=coeff['scaler_mean'];xx/=coeff['scaler_scale']
 prob=expit(xx@coeff['coef'].ravel()+coeff['intercept'][0]);tp=pred[pred.protocol.eq(row.protocol)].sort_values('image_id')
 assert tp.image_id.tolist()==d.loc[te].image_id.tolist() and np.max(abs(prob-tp.prob_OTHER))<1e-12
 y=d.loc[te].domain.eq('OTHER').to_numpy();assert abs(roc_auc_score(y,prob)-row.test_auc)<1e-12 and abs(balanced_accuracy_score(y,prob>=.5)-row.test_balanced_accuracy)<1e-12
 td=d.loc[te].reset_index(drop=True);rng=np.random.default_rng(19200+(row.protocol=='P5'));strata=td.groupby(['domain','label']).indices;ix=np.concatenate([rng.choice(ii,size=(5000,len(ii)),replace=True) for ii in strata.values()],axis=1)
 auc=np.array([roc_auc_score(y[ii],prob[ii]) for ii in ix]);lo,hi=np.quantile(auc,[.025,.975]);assert np.max(np.abs([lo-row.auc_ci95_lower,hi-row.auc_ci95_upper]))<1e-12
checks.append(f'Frozen {len(ids)} x 2048 features hashed; training-only scaler, reserved-test probabilities, AUCs and 5000-replicate bounds verified')
pd.DataFrame(records).to_csv(W/'19_run_audit.csv',index=False)
report=dict(source_zip_sha256=sha(source) if source.is_file() else None,completed_fits=48,evaluations=96,result_files_hashed=hashed,max_metric_error=maxerr,checks=checks,all_16_loss_intervals_positive=bool((summary.image_ci95_lower>0).all()),cpu_environment=dict(python=sys.version),no_neural_refitting_in_audit=True)
(W/'19_verification_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));print(pd.DataFrame(new).to_string(index=False));print(pd.DataFrame(records).groupby('protocol').wall_seconds.sum()/3600)
