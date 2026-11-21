"""Class-specific analyses on frozen Paddy Doctor outputs; no training."""
from pathlib import Path
from zipfile import ZipFile
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import argparse
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project',type=Path,default=HERE if (HERE/'inputs').is_dir() else ROOT/'paddy14')
parser.add_argument('--output',type=Path,default=HERE/'reproduced_revision_results')
args=parser.parse_args()
PROJECT=args.project
OUT=args.output
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(PROJECT))
import paddy14_analysis as base

sources,entries,hist,audits,rules=base.inspect_sources(PROJECT,True)
cm= pd.read_csv((PROJECT/'analysis14_results/tables/class_metrics.csv' if (PROJECT/'analysis14_results').is_dir() else PROJECT/'results/tables/class_metrics.csv'))
cm=cm[(cm.method=='raw')&(cm.experiment!='P1')].copy()
cm['condition']=np.where(cm.source==cm.target,'control','transfer')
summary=cm.groupby(['experiment','target','model','class','condition']).agg(f1=('f1','mean'),f1_sd=('f1','std'),recall=('recall','mean'),precision=('precision','mean'),support=('support','first')).reset_index()
summary.to_csv(OUT/'class_performance_P4_P5.csv',index=False)
wide=summary.pivot(index=['experiment','target','model','class'],columns='condition',values=['f1','f1_sd','recall','precision','support'])
wide.columns=['_'.join(c) for c in wide.columns];wide=wide.reset_index();wide['f1_loss']=wide.f1_control-wide.f1_transfer
wide.to_csv(OUT/'class_control_transfer_differences.csv',index=False)

persistent=pd.read_csv((PROJECT/'analysis14_results/tables/persistent_errors_per_image.csv' if (PROJECT/'analysis14_results').is_dir() else PROJECT/'results/tables/persistent_errors_per_image.csv'))
persistent['always_wrong']=persistent.cross_errors_3seeds.eq(3)
per=persistent.groupby(['experiment','target','model','true_class']).agg(n=('image_id','size'),always_wrong=('always_wrong','sum'),control_correct_transfer_wrong=('correct_control_all_wrong_cross_all','sum'),highconf_raw=('wrong_highconf090_raw_all','sum'),highconf_ts=('wrong_highconf090_ts_all','sum')).reset_index()
per.to_csv(OUT/'persistent_errors_by_class.csv',index=False)

with ZipFile(PROJECT/'inputs/p5.zip') as z:
    assignment=pd.read_csv(z.open('data/splits/P5A_ADT45_to_OTHER_age_matched.csv'))
assignment.to_csv(OUT/'P5_frozen_pair_image_assignments.csv',index=False)
assert assignment.image_id.is_unique and len(assignment)==1000
paircheck=assignment.groupby('matched_pair_id').agg(n=('image_id','size'),domains=('domain','nunique'),ages=('age','nunique'),classes=('label','nunique'),splits=('pair_split','nunique'))
assert len(paircheck)==500 and paircheck.n.eq(2).all() and paircheck.domains.eq(2).all() and paircheck.ages.eq(1).all() and paircheck.classes.eq(1).all() and paircheck.splits.eq(1).all()
composition=assignment.groupby(['pair_split','label','domain','variety']).size().rename('n').reset_index()
composition.to_csv(OUT/'P5_cultivar_class_composition_by_split.csv',index=False)
ages=assignment.groupby(['pair_split','label','age','domain']).size().unstack(fill_value=0).reset_index()
assert np.array_equal(ages.ADT45,ages.OTHER)
ages.to_csv(OUT/'P5_exact_age_counts_by_class_and_split.csv',index=False)
metadata_path=PROJECT/'protocol_integrity/data/metadata/03_paddy_doctor_nearclean.csv'
if metadata_path.exists():
    metadata=pd.read_csv(metadata_path)
else:
    with ZipFile(ROOT/'manuscript_rewrite_work/protocol03.zip') as z:
        metadata=pd.read_csv(z.open('data/metadata/03_paddy_doctor_nearclean.csv'))
metadata['domain']=np.where(metadata.variety.eq('ADT45'),'ADT45','OTHER')
candidates=metadata.groupby(['label','age','domain']).size().unstack(fill_value=0)
candidates['available_pairs']=candidates[['ADT45','OTHER']].min(axis=1)
candidates=candidates.reset_index();candidates.to_csv(OUT/'P5_candidate_support_by_class_and_age.csv',index=False)
support=candidates.groupby('label').agg(ADT45_candidates=('ADT45','sum'),OTHER_candidates=('OTHER','sum'),exact_age_pairs=('available_pairs','sum')).reset_index()
support['selected_pairs']=np.where(support.label.isin(assignment.label.unique()),100,0)
support.to_csv(OUT/'P5_candidate_support_by_class.csv',index=False)
normal=assignment[(assignment.domain=='OTHER')&assignment.label.eq('normal')]
normal_summary=normal.groupby('pair_split').agg(n=('image_id','size'),ponni=('variety',lambda x:x.eq('Ponni').sum())).reset_index()
normal_summary.to_csv(OUT/'P5_normal_Ponni_by_split.csv',index=False)

# Raw-to-temperature-scaled gains; positive values mean a lower calibrated score.
# Reuse a target-specific class-stratified resampling matrix across models/seeds.
gains=[];confusions=[]
for experiment in ['P4','P5']:
 for target in ['ADT45','OTHER']:
    group=[e for e in entries if e['experiment']==experiment and e['target']==target and e['source']!=target]
    ref=group[0];y=ref['y'];rng=np.random.default_rng(20261121+['P4','P5'].index(experiment)*10+['ADT45','OTHER'].index(target))
    indices=np.concatenate([rng.choice(np.flatnonzero(y==c),(2000,(y==c).sum()),replace=True) for c in range(len(ref['classes']))],axis=1)
    for model in base.MODELS:
        runs=sorted([e for e in group if e['model']==model],key=lambda e:e['seed'])
        for e in runs:assert np.array_equal(e['df'].image_id,ref['df'].image_id)
        for metric in ['nll','brier']:
            differences=[];seed_gains=[]
            for e in runs:
                if metric=='nll':
                    raw=-np.log(np.clip(e['p'][np.arange(len(y)),y],base.EPS,1));scaled=-np.log(np.clip(e['q'][np.arange(len(y)),y],base.EPS,1))
                else:
                    onehot=np.eye(len(e['classes']))[y];raw=((e['p']-onehot)**2).sum(1);scaled=((e['q']-onehot)**2).sum(1)
                differences.append(raw-scaled);seed_gains.append(float((raw-scaled).mean()))
            mean_difference=np.mean(differences,axis=0);rep=mean_difference[indices].mean(axis=1);lo,hi=np.quantile(rep,[.025,.975])
            gains.append(dict(experiment=experiment,target=target,model=model,metric=metric,gain=float(mean_difference.mean()),ci_low=float(lo),ci_high=float(hi),improved_seeds=sum(g>0 for g in seed_gains),n=len(y),replicates=2000))
    # Full confusion counts per seed, then representative ConvNeXt figure.
    for e in [e for e in entries if e['experiment']==experiment and e['target']==target]:
        counts=confusion_matrix(e['y'],e['p'].argmax(1),labels=range(len(e['classes'])))
        for i,true in enumerate(e['classes']):
            for j,pred in enumerate(e['classes']):confusions.append(dict(experiment=experiment,target=target,source=e['source'],model=e['model'],seed=e['seed'],true_class=true,predicted_class=pred,count=int(counts[i,j]),row_fraction=float(counts[i,j]/counts[i].sum())))
pd.DataFrame(gains).to_csv(OUT/'paired_calibration_score_gains.csv',index=False)
pd.DataFrame(confusions).to_csv(OUT/'confusion_matrices_all_models_and_seeds.csv',index=False)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
for experiment in ['P4','P5']:
    fig,axes=plt.subplots(2,2,figsize=(9.4,8.2))
    for row,target in enumerate(['ADT45','OTHER']):
     for col,condition in enumerate(['control','transfer']):
        selected=[e for e in entries if e['experiment']==experiment and e['target']==target and e['model']=='convnext_tiny' and (e['source']==target)==(condition=='control')]
        probs=[]
        for e in selected:
            counts=confusion_matrix(e['y'],e['p'].argmax(1),labels=range(len(e['classes'])));probs.append(counts/counts.sum(1,keepdims=True))
        matrix=np.mean(probs,axis=0);ax=axes[row,col];im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues')
        labels=[x.replace('bacterial_','bact. ').replace('_',' ') for x in selected[0]['classes']]
        ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,title=f'Target {target} | {condition}',xlabel='Predicted label',ylabel='True label');ax.tick_params(axis='x',rotation=65,labelsize=7);ax.tick_params(axis='y',labelsize=7)
        for i in range(len(labels)):
         for j in range(len(labels)):
            if matrix[i,j]>=.045:ax.text(j,i,f'{100*matrix[i,j]:.0f}',ha='center',va='center',fontsize=6.8,color='white' if matrix[i,j]>.55 else 'black')
    fig.tight_layout(rect=(0,0,.94,1));cax=fig.add_axes([.95,.19,.015,.62]);fig.colorbar(im,cax=cax,label='Mean row proportion across three seeds')
    for extension in ['png','pdf']:fig.savefig(OUT/f'Confusion_{experiment}_ConvNeXt.{extension}',dpi=300,bbox_inches='tight')
    plt.close(fig)

findings={'normal_Ponni_by_split':normal_summary.to_dict('records'),'available_support':support.to_dict('records'),'worst_f1_losses':wide[wide.experiment.eq('P5')&wide.model.eq('convnext_tiny')].sort_values('f1_loss',ascending=False).head(5).to_dict('records'),'paired_score_gains':gains,'matched_pairs':500,'exact_age_strata_verified':len(ages),'no_retraining':True,'no_external_validation':True,'score_bootstrap_seed_rule':'20261121 + protocol index*10 + target index; fixed 2000 replicates'}
(OUT/'summary.json').write_text(json.dumps(findings,indent=2),encoding='utf-8')
print(json.dumps({k:findings[k] for k in ['normal_Ponni_by_split','available_support','worst_f1_losses','exact_age_strata_verified']},indent=2))
