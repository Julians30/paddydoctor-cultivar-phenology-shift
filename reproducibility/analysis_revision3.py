"""Cultivar association and reliability analyses for Paddy Doctor; no network training.

Run from the portable Online Resource 2 folder with --project . --output results.
No source/target test labels are used to fit a temperature or acceptance threshold.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr,binomtest
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--bootstrap',type=int,default=2000)
args=parser.parse_args();P=args.project.resolve();O=args.output.resolve();O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(P));import paddy14_analysis as b
sources,entries,hist,audits,rules=b.inspect_sources(P,True)
B=args.bootstrap
def save(rows,name):pd.DataFrame(rows).to_csv(O/(name+'.csv'),index=False)
def scores(y,p):
    d=b.metrics(y,p);d['brier_per_class']=d['brier']/p.shape[1];return d
def ordered_risk(y,p,ids):
    # Stable image-identifier tie-break; no labels determine the ranking.
    order=np.lexsort((np.asarray(ids,dtype=str),-p.max(1)))
    wrong=(p.argmax(1)!=y)[order].astype(float);n=len(y)
    risk=np.cumsum(wrong)/np.arange(1,n+1)
    ideal=np.cumsum(np.sort(wrong))/np.arange(1,n+1)
    return float(risk.mean()),float((risk-ideal).mean())

def ece_bootstrap(y,p,idx,bins,adaptive):
    """Vectorized ECE; repeated quantiles create empty bins, contributing zero."""
    conf=p.max(1)[idx];correct=(p.argmax(1)==y)[idx]
    edges=np.quantile(conf,np.linspace(0,1,bins+1)[1:-1],axis=1).T if adaptive else np.broadcast_to(np.linspace(0,1,bins+1)[1:-1],(len(idx),bins-1))
    groups=(conf[:,:,None]>edges[:,None,:]).sum(2)
    codes=(groups+np.arange(len(idx))[:,None]*bins).ravel()
    total=np.bincount(codes,weights=(conf-correct).ravel(),minlength=len(idx)*bins).reshape(len(idx),bins)
    return np.abs(total).sum(1)/idx.shape[1]

# The cultivar rule is fitted only to OTHER training rows. Full-pool purity is
# recorded separately as resubstitution, never as a held-out performance score.
metadata=[];mapping_rows=[];concentration=[];assignments={}
for ex,path,split in [
 ('P4',P/'protocol_integrity/data/splits/P4_matched_domain_assignments.csv','domain_split'),
 ('P5',P/'revision2_results/P5_frozen_pair_image_assignments.csv','pair_split')]:
    a=pd.read_csv(path);assignments[ex]=a;a=a[a.domain.eq('OTHER')].copy();classes=sorted(a.label.unique())
    train=a[a[split].eq('train')];ct=pd.crosstab(train.variety,train.label).reindex(columns=classes,fill_value=0)
    # Lexicographic order resolves an exact training-count tie deterministically.
    lookup=ct.idxmax(axis=1).to_dict();fallback=train.label.value_counts().reindex(classes,fill_value=0).idxmax()
    for cultivar,row in ct.iterrows():mapping_rows.append(dict(experiment=ex,cultivar=cultivar,training_n=int(row.sum()),predicted_class=lookup[cultivar],tie_count=int((row==row.max()).sum())))
    for part in ['train','validation','test']:
        d=a[a[split].eq(part)];pred=d.variety.map(lookup).fillna(fallback)
        metadata.append(dict(experiment=ex,evaluation=part,n=len(d),accuracy=accuracy_score(d.label,pred),balanced_accuracy=balanced_accuracy_score(d.label,pred),macro_f1=f1_score(d.label,pred,labels=classes,average='macro',zero_division=0),unseen_cultivar_n=int((~d.variety.isin(lookup)).sum()),fit='training_only'))
    full=pd.crosstab(a.variety,a.label).reindex(columns=classes,fill_value=0)
    metadata.append(dict(experiment=ex,evaluation='all_pool_resubstitution',n=len(a),accuracy=full.max(axis=1).sum()/len(a),balanced_accuracy=np.nan,macro_f1=np.nan,unseen_cultivar_n=0,fit='same_full_pool'))
    for cl in classes:
        counts=train[train.label.eq(cl)].variety.value_counts();share=counts/counts.sum()
        concentration.append(dict(experiment=ex,**{'class':cl},n_train=len(train[train.label.eq(cl)]),cultivars=len(counts),largest_cultivar=counts.index[0],largest_share=share.max(),hhi=float((share**2).sum())))
save(metadata,'cultivar_only_baseline');save(mapping_rows,'cultivar_training_rules');save(concentration,'class_cultivar_concentration')
classperf=pd.read_csv(P/'revision2_results/class_control_transfer_differences.csv')
conc=pd.DataFrame(concentration);relations=[]
for ex in ['P4','P5']:
    d=classperf[(classperf.experiment==ex)&(classperf.target=='ADT45')].merge(conc[conc.experiment==ex],on=['experiment','class'])
    save(d,f'{ex}_ADT45_class_loss_and_OTHER_concentration')
    for model,g in d.groupby('model'):
        for field in ['largest_share','hhi']:
            rho=spearmanr(g[field],g.f1_loss).statistic
            relations.append(dict(experiment=ex,model=model,concentration=field,n_classes=len(g),spearman_rho=rho))
save(relations,'class_loss_concentration_correlations')

# Seed-wise effects, conditional image intervals, and a sensitivity bootstrap
# resampling the three paired seed indices as well as class-stratified images.
effects=[];seed_effects=[];ranking=[];class_ci=[];ece=[]
for ex in ['P4','P5']:
 for target in ['ADT45','OTHER']:
    opposite='OTHER' if target=='ADT45' else 'ADT45'
    ref=next(e for e in entries if e['experiment']==ex and e['target']==target)
    y=ref['y'];k=len(ref['classes']);idx=b.boot_indices(y,B,30100+(ex=='P5')*100+(target=='OTHER'))
    draws=np.random.default_rng(30200+(ex=='P5')*100+(target=='OTHER')).integers(0,3,(B,3))
    for model in b.MODELS:
        c=sorted([e for e in entries if e['experiment']==ex and e['target']==target and e['source']==target and e['model']==model],key=lambda e:e['seed'])
        x=sorted([e for e in entries if e['experiment']==ex and e['target']==target and e['source']==opposite and e['model']==model],key=lambda e:e['seed'])
        for e in c+x:assert np.array_equal(e['df'].image_id,ref['df'].image_id)
        seed_boot=[];points=[];control=[];transfer=[]
        for ce,xe in zip(c,x):
            cp=ce['p'].argmax(1);xp=xe['p'].argmax(1)
            cf=b.manual_f1(y,cp,k);xf=b.manual_f1(y,xp,k);delta=cf-xf
            seed_boot.append(b.bootstrap_f1(y,[cp],idx,k)-b.bootstrap_f1(y,[xp],idx,k));points.append(delta);control.append(cf);transfer.append(xf)
            seed_effects.append(dict(experiment=ex,target=target,model=model,seed=ce['seed'],control_f1=cf,transfer_f1=xf,loss=delta,relative_loss_percent=100*delta/cf))
            for condition,pred in [('control',cp),('transfer',xp)]:
                for j,cl in enumerate(ref['classes']):
                    n=int((y==j).sum());correct=int(((y==j)&(pred==j)).sum());ci=binomtest(correct,n).proportion_ci(confidence_level=.95,method='exact')
                    class_ci.append(dict(experiment=ex,target=target,model=model,seed=ce['seed'],condition=condition,**{'class':cl},n=n,correct=correct,recall=correct/n,ci_low=ci.low,ci_high=ci.high))
        sb=np.array(seed_boot).T;conditional=sb.mean(1);hier=sb[np.arange(B)[:,None],draws].mean(1)
        effects.append(dict(experiment=ex,target=target,model=model,n=len(y),control_f1=np.mean(control),transfer_f1=np.mean(transfer),loss=np.mean(points),seed_loss_min=min(points),seed_loss_max=max(points),relative_loss_percent=100*np.mean(points)/np.mean(control),image_ci_low=np.quantile(conditional,.025),image_ci_high=np.quantile(conditional,.975),seed_image_ci_low=np.quantile(hier,.025),seed_image_ci_high=np.quantile(hier,.975)))
        # Paired differences in ECE, recomputing bin boundaries in every sample.
        for condition,runs in [('control',c),('transfer',x)]:
            for adaptive,bins in [(False,15),(True,10)]:
                def ec(y0,p):return b.calibration_bins(p.argmax(1)==y0,p.max(1),bins,adaptive)[0]
                point=np.mean([ec(y,e['p'])-ec(y,e['q']) for e in runs]);rep=np.zeros(B)
                for e in runs:rep+=(ece_bootstrap(y,e['p'],idx,bins,adaptive)-ece_bootstrap(y,e['q'],idx,bins,adaptive))/3
                ece.append(dict(experiment=ex,target=target,model=model,condition=condition,bins=bins,adaptive=adaptive,raw_minus_ts=point,ci_low=np.quantile(rep,.025),ci_high=np.quantile(rep,.975)))
save(effects,'loss_seed_and_image_sensitivity');save(seed_effects,'loss_by_seed');save(class_ci,'class_recall_exact_intervals');save(ece,'paired_ECE_changes')
ef=pd.DataFrame(effects)
for ex,g in ef.groupby('experiment'):
    wide=g.pivot(index='model',columns='target',values='transfer_f1')
    ranking.append(dict(experiment=ex,n_models=len(wide),spearman_transfer_direction=spearmanr(wide.ADT45,wide.OTHER).statistic))
save(ranking,'direction_ranking_sensitivity')

aurc=[];strict=[];ensemble=[];ensemble_predictions=[];temperature_fits=[]
# Temperatures for the mean-probability ensemble are fitted only on source
# validation. Same fitted value is reused on source and opposite target tests.
for ex in ['P1','P4','P5']:
 source_names=['MIXED'] if ex=='P1' else ['ADT45','OTHER']
 for source in source_names:
  for model in b.MODELS:
    vr=[]
    for seed in b.SEEDS:
        run_id=rules[(rules.experiment==ex)&(rules.source==source)&(rules.model==model)&(rules.seed==seed)].iloc[0].run_id
        a=np.load(P/'inputs/validation_cache'/(run_id+'.npz'),allow_pickle=False);order=np.argsort(a['image_ids'].astype(str))
        vr.append(dict(seed=seed,y=a['y_true'][order].astype(int),ids=a['image_ids'][order].astype(str),logits=a['logits'][order].astype(float),p=softmax(a['logits'][order].astype(float),axis=1)))
    for v in vr[1:]:assert np.array_equal(v['ids'],vr[0]['ids']) and np.array_equal(v['y'],vr[0]['y'])
    vp=np.mean([v['p'] for v in vr],axis=0);vy=vr[0]['y'];logp=np.log(np.clip(vp,b.EPS,1))
    def objective(log_t):
        q=softmax(logp/np.exp(log_t),axis=1);return -np.log(np.clip(q[np.arange(len(vy)),vy],b.EPS,1)).mean()
    fit=minimize_scalar(objective,bounds=(np.log(.05),np.log(20)),method='bounded',options={'xatol':1e-8,'maxiter':500});T=float(np.exp(fit.x));assert fit.success
    temperature_fits.append(dict(experiment=ex,source=source,model=model,temperature=T,n_validation=len(vy),optimizer_success=bool(fit.success),n_evaluations=fit.nfev,validation_raw_nll=float(objective(0)),validation_ts_nll=float(fit.fun),lower_bound=.05,upper_bound=20))
    target_names=['MIXED'] if ex=='P1' else ['ADT45','OTHER']
    for target in target_names:
        runs=sorted([e for e in entries if e['experiment']==ex and e['source']==source and e['target']==target and e['model']==model],key=lambda e:e['seed']);ref=runs[0];y=ref['y'];ids=ref['df'].image_id.to_numpy()
        for e in runs:assert np.array_equal(e['df'].image_id,ids)
        ep=np.mean([e['p'] for e in runs],axis=0);eq=softmax(np.log(np.clip(ep,b.EPS,1))/T,axis=1)
        condition='control' if source==target else 'transfer'
        for method,p in [('raw',ep),('ts',eq)]:
            a,ea=ordered_risk(y,p,ids);single=np.mean([scores(y,e['p'] if method=='raw' else e['q'])['macro_f1'] for e in runs])
            ensemble.append(dict(experiment=ex,source=source,target=target,model=model,condition=condition,method=method,temperature=T,**scores(y,p),aurc=a,eaurc=ea,single_seed_mean_f1=single,ensemble_minus_single_mean_f1=scores(y,p)['macro_f1']-single))
            for i,iid in enumerate(ids):ensemble_predictions.append(dict(experiment=ex,source=source,target=target,model=model,method=method,image_id=iid,true_class=ref['classes'][y[i]],predicted_class=ref['classes'][p[i].argmax()],**{'prob__'+cl:float(p[i,j]) for j,cl in enumerate(ref['classes'])}))
        for e,v in zip(runs,vr):
            for method,p in [('raw',e['p']),('ts',e['q'])]:
                a,ea=ordered_risk(y,p,ids);aurc.append(dict(experiment=ex,source=source,target=target,model=model,condition=condition,seed=e['seed'],method=method,aurc=a,eaurc=ea))
                if ex=='P1':continue
                vprob=v['p'] if method=='raw' else softmax(v['logits']/e['temperature'],axis=1)
                for limit in [.01,.02,.05]:
                    threshold,n,vrisk,eligible=b.validation_rule(v['y'],vprob,risk_limit=limit)
                    selected=p.max(1)>=threshold;wrong=p.argmax(1)!=y
                    strict.append(dict(experiment=ex,source=source,target=target,model=model,condition=condition,seed=e['seed'],method=method,risk_limit=limit,threshold=threshold,eligible=eligible,validation_n=len(v['y']),validation_accepted=n,validation_risk=vrisk,target_n=len(y),target_accepted=int(selected.sum()),target_errors=int((selected&wrong).sum()),target_coverage=selected.mean(),target_risk=float(wrong[selected].mean()) if selected.any() else np.nan,zero_threshold=threshold==0))
save(aurc,'AURC_by_run');save(strict,'stricter_source_rules');save(ensemble,'ensemble_metrics');save(ensemble_predictions,'ensemble_predictions');save(temperature_fits,'ensemble_temperature_fits')
config={'bootstrap_replicates':B,'seed_bootstrap':'resample 3 paired seed indices with replacement using shared class-stratified image samples; exploratory with only 3 seeds','ece_intervals':'percentile, paired raw-minus-TS, conditional on fixed fitted models and temperatures; adaptive boundaries recomputed in each replicate','aurc':'mean cumulative error at retained counts 1..N; confidence descending, image_id ascending tie break; E-AURC subtracts finite-sample optimal ordering at same error count','metadata_baseline':'cultivar-majority rule fitted only to training rows, lexicographic ties and training-majority fallback; full-pool score separately labeled resubstitution','new_network_training':False,'ensemble':'mean raw class probabilities of 3 seeds within same architecture; one scalar T fitted on source-validation ensemble probabilities using bounded log-temperature minimization [.05,20], xatol 1e-8 maxiter 500; no target labels for fitting','strict_rules':'source-only empirical risk limits .01,.02,.05, minimum 20 accepted validation images and 20% coverage; no eligible threshold means reject all'}
(O/'analysis_config.json').write_text(json.dumps(config,indent=2))
print(json.dumps({'result_csvs':len(list(O.glob('*.csv'))),'metadata':metadata,'ranking':ranking,'p5_ensemble_raw':pd.DataFrame(ensemble).query("experiment=='P5' and condition=='transfer' and method=='raw'")[['target','model','macro_f1','ensemble_minus_single_mean_f1']].to_dict('records')},indent=2))
