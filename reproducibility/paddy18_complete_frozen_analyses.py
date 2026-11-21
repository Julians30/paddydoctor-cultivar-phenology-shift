"""Calibration and uncertainty analyses on frozen P4/P5 predictions.

No neural-network training. Calibrators use source validation only unless
explicitly labelled target-informed. Conditional image and paired-seed
sensitivities do not claim independent-cohort or field coverage.
"""
from pathlib import Path
import argparse,json,sys,hashlib,warnings,itertools
import numpy as np
import pandas as pd
from scipy.special import softmax,logsumexp,ndtr,ndtri
from scipy.optimize import minimize,minimize_scalar
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,log_loss

def save(d,out,name):
    pd.DataFrame(d).to_csv(out/(name+'.csv'),index=False)

def calibrate(x,y,kind,lam=0.):
    """Vector maps act on centered log probabilities; Dirichlet maps on log probabilities.
    This preserves a fully reconstructible input from archived probabilities.
    Both are fitted to mean NLL, regularized toward the identity map.
    """
    n,k=x.shape
    if kind=='temperature':
        r=minimize_scalar(lambda u:np.mean(logsumexp(x/np.exp(u),axis=1)-x[np.arange(n),y]/np.exp(u)),bounds=(np.log(.05),np.log(20)),method='bounded',options={'xatol':1e-8})
        return {'kind':kind,'temperature':float(np.exp(r.x)),'success':bool(r.success),'nll':float(r.fun),'gradient':np.nan}
    z=x-x.mean(1,keepdims=True) if kind=='vector' else x
    init=np.r_[np.ones(k),np.zeros(k)] if kind=='vector' else np.r_[np.eye(k).ravel(),np.zeros(k)]
    def obj(theta):
        if kind=='vector':a=theta[:k];b=theta[k:];s=z*a+b
        else:a=theta[:k*k].reshape(k,k);b=theta[k*k:];s=z@a.T+b
        p=softmax(s,axis=1);v=np.mean(logsumexp(s,axis=1)-s[np.arange(n),y]);p[np.arange(n),y]-=1;p/=n
        if kind=='vector':
            penalty=lam*(np.mean((a-1)**2)+np.mean(b*b));g=np.r_[(p*z).sum(0)+2*lam*(a-1)/k,p.sum(0)+2*lam*b/k]
        else:
            # ODIR penalizes only off-diagonal weights and intercepts, as specified in Kull et al.
            off=a-np.diag(np.diag(a));penalty=lam*(np.sum(off*off)/(k*(k-1))+np.mean(b*b));g=np.r_[(p.T@z+2*lam*off/(k*(k-1))).ravel(),p.sum(0)+2*lam*b/k]
        return float(v+penalty),g
    bounds=([(.05,20.)]*k+[(-10.,10.)]*k) if kind=='vector' else ([((.05,20.) if i==j else (-10.,10.)) for i in range(k) for j in range(k)]+[(-10.,10.)]*k)
    r=minimize(obj,init,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':2000,'maxls':100,'ftol':1e-10,'gtol':1e-7})
    solver='L-BFGS-B'
    if not r.success:
        r=minimize(obj,r.x,jac=True,bounds=bounds,method='SLSQP',options={'maxiter':2000,'ftol':1e-10});solver='SLSQP after L-BFGS-B'
    return {'kind':kind,'theta':r.x.tolist(),'lambda':lam,'success':bool(r.success),'solver':solver,'objective':float(r.fun),'gradient':float(np.linalg.norm(r.jac,np.inf)),'iterations':r.nit,'k':k,'bounds':'diagonal .05 to 20; off-diagonal and intercept -10 to 10; selected before test evaluation'}

def apply(x,fit):
    k=x.shape[1];kind=fit['kind']
    if kind=='temperature':return softmax(x/fit['temperature'],axis=1)
    theta=np.array(fit['theta'])
    if kind=='vector':return softmax((x-x.mean(1,keepdims=True))*theta[:k]+theta[k:],axis=1)
    return softmax(x@theta[:k*k].reshape(k,k).T+theta[k*k:],axis=1)

def select_calibrator(x,y,kind,seed):
    folds=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(x,y));rows=[]
    for lam in [.001,.01,.1,1.,10.]:
        values=[]
        for tr,va in folds:
            f=calibrate(x[tr],y[tr],kind,lam);p=apply(x[va],f);values.append(log_loss(y[va],p,labels=np.arange(x.shape[1])))
            if not f['success']:raise RuntimeError(f'Calibration CV did not converge: {kind}, lambda={lam}, gradient={f["gradient"]}')
        rows.append((float(np.mean(values)),lam))
    best=min(rows,key=lambda v:(v[0],-v[1]))[1];fit=calibrate(x,y,kind,best)
    if not fit['success']:raise RuntimeError('Final calibrator did not converge')
    fit['cv_mean_nll']=dict((str(lam),v) for v,lam in rows);return fit

def bca(point,rep,jack):
    # Midrank ties keep degenerate / discrete F1 bootstrap distributions explicit.
    if np.allclose(rep,rep[0],rtol=0,atol=1e-14):return float(rep[0]),float(rep[0]),'constant_bootstrap'
    frac=(np.sum(rep<point)+.5*np.sum(rep==point))/len(rep);frac=np.clip(frac,.5/len(rep),1-.5/len(rep));z0=ndtri(frac)
    inf=jack.mean()-jack;den=6*np.sum(inf**2)**1.5
    acc=float(np.sum(inf**3)/den) if den>1e-20 else 0.
    zz=ndtri(np.array([.025,.975]));div=1-acc*(z0+zz)
    if np.any(np.abs(div)<1e-8):return np.nan,np.nan,'singular_acceleration'
    q=ndtr(z0+(z0+zz)/div);lo,hi=np.quantile(rep,np.sort(np.clip(q,0,1)))
    return float(lo),float(hi),'ok'

def main(project,out,B=5000,do_mixed=True):
    out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(project));import paddy14_analysis as base
    sources,entries,hist,checks,rules=base.inspect_sources(project,True)
    e=[x for x in entries if x['experiment'] in ['P4','P5']];cal_rows=[];fits=[];bin_rows=[];corrected=[];target_rows=[];target_fits=[];crc=[];bca_rows=[];paired_changes=[];new_predictions=[];l1corrected=[]
    # Source-only nested calibration comparison on both protocols and domains.
    fitted={};calprob={}
    for r in rules[rules.method.eq('raw') & rules.experiment.isin(['P4','P5'])].itertuples():
        a=np.load(project/'inputs/validation_cache'/(r.run_id+'.npz'),allow_pickle=False);x=a['logits'].astype(float);vy=a['y_true'].astype(int);xp=x-logsumexp(x,axis=1)[:,None]
        key=(r.experiment,r.source,r.model,r.seed)
        for kind in ['vector','dirichlet_odir']:
            cache=out/'fit_cache';cache.mkdir(exist_ok=True);cp=cache/(r.run_id+'__'+kind+'.json')
            f=json.loads(cp.read_text()) if cp.exists() else select_calibrator(xp,vy,kind,r.seed)
            cp.write_text(json.dumps(f,indent=2)+'\n');fitted[key+(kind,)]=f;fits.append(dict(experiment=r.experiment,source=r.source,model=r.model,seed=r.seed,method=kind,n_validation=len(vy),**f))
        print('Source calibrators',r.run_id,flush=True)
    for x in e:
        key={k:x[k] for k in ['experiment','source','target','model','seed']};y=x['y'];lp=np.log(np.clip(x['p'],base.EPS,1));condition='control' if x['source']==x['target'] else 'transfer'
        for kind in ['raw','temperature','vector','dirichlet_odir']:
            p=x['p'] if kind=='raw' else x['q'] if kind=='temperature' else apply(lp,fitted[(x['experiment'],x['source'],x['model'],x['seed'],kind)])
            calprob[(x['experiment'],x['source'],x['target'],x['model'],x['seed'],kind)]=p
            cal_rows.append(dict(**key,condition=condition,method=kind,**base.metrics(y,p)))
            # Analytic expectation of the authors' normal-bootstrap L1 adjustment.
            # Adaptive bin edges use confidence only, never correctness labels.
            conf0=p.max(1);correct0=(p.argmax(1)==y).astype(float);sortedconf=np.sort(conf0);chunks=np.array_split(sortedconf,5)
            edges0=np.unique([(chunks[j][-1]+chunks[j+1][0])/2 for j in range(4)])
            g0=np.searchsorted(edges0,conf0,side='left');plugin=0.;normal_expected=0.;small=False
            for j in range(len(edges0)+1):
                ii=np.flatnonzero(g0==j)
                if not len(ii):continue
                if len(ii)<2:small=True
                acc=correct0[ii].mean();mu=acc-conf0[ii].mean();se=np.sqrt(acc*(1-acc)/len(ii));v=abs(mu) if se==0 else se*np.sqrt(2/np.pi)*np.exp(-.5*(mu/se)**2)+mu*(2*ndtr(mu/se)-1)
                plugin+=len(ii)/len(y)*abs(mu);normal_expected+=len(ii)/len(y)*v
            l1corrected.append(dict(**key,method=kind,adaptive_bins=5,plugin_L1_ECE=plugin,normal_bias_adjusted_L1_ECE=2*plugin-normal_expected if not small else np.nan,singleton_present=small,interpretation='normal-approximation bias adjustment; signed value retained; not exact unbiased ECE'))
            for i,iid in enumerate(x['df'].image_id):new_predictions.append(dict(**key,method=kind,image_id=iid,y_true=int(y[i]),**{'prob__'+cl:float(p[i,j]) for j,cl in enumerate(x['classes'])}))
            # Kumar-style residual-variance correction on bins fixed using source validation.
            run_id=f"{x['experiment']}__train_{x['source']}__{x['model']}__seed_{x['seed']}";a=np.load(project/'inputs/validation_cache'/(run_id+'.npz'),allow_pickle=False)
            vl=a['logits'].astype(float);vlogp=vl-logsumexp(vl,axis=1)[:,None]
            vp=softmax(vl,axis=1) if kind=='raw' else softmax(vl/x['temperature'],axis=1) if kind=='temperature' else apply(vlogp,fitted[(x['experiment'],x['source'],x['model'],x['seed'],kind)])
            edges=np.unique(np.quantile(vp.max(1),np.linspace(0,1,6)))[1:-1];group=np.searchsorted(edges,p.max(1),side='left');correct=(p.argmax(1)==y).astype(float);conf=p.max(1);parts=[];n_single=0
            for j in range(len(edges)+1):
                ii=np.flatnonzero(group==j)
                if not len(ii):continue
                d=correct[ii]-conf[ii];plug=float(d.mean()**2)
                corr=float(np.var(d,ddof=1)/len(ii)) if len(ii)>1 else np.nan
                parts.append((len(ii),plug,corr))
                n_single+=len(ii)==1
                bin_rows.append(dict(**key,method=kind,bin=j,n=len(ii),mean_confidence=float(conf[ii].mean()),accuracy=float(correct[ii].mean()),plugin_squared=plug,variance_correction=corr))
            # No silent omission/imputation of singleton-bin variance.
            signed=sum(n*(v-c) for n,v,c in parts)/len(y) if not n_single else np.nan
            corrected.append(dict(**key,method=kind,bins_requested=5,bins_observed=len(parts),singleton_bins=n_single,plugin_L2_squared=sum(n*v for n,v,c in parts)/len(y),debiased_L2_squared=signed,debiased_L2_root=np.sqrt(max(0,signed)) if np.isfinite(signed) else np.nan,definition='weighted mean residual squared minus sample residual variance/n; source-fixed bins; L2, not unbiased L1 ECE'))
        # Target-labelled reference: 20 target-test labels per construction, excluded from each evaluation.
        if x['experiment']=='P5' and condition=='transfer':
            folds=list(StratifiedKFold(5,shuffle=True,random_state=18100+(x['target']=='OTHER')).split(lp,y))
            for split,(eval_idx,cal_idx) in enumerate(folds):
                f=calibrate(lp[cal_idx],y[cal_idx],'temperature');target_fits.append(dict(**key,split=split,calibration_image_ids=x['df'].image_id.iloc[cal_idx].tolist(),evaluation_image_ids=x['df'].image_id.iloc[eval_idx].tolist(),**f))
                assert len(cal_idx)==20 and len(eval_idx)==80 and not set(cal_idx)&set(eval_idx)
                for method,p in [('raw',x['p']),('source_temperature',x['q']),('target20_temperature',apply(lp,f))]:
                    target_rows.append(dict(**key,split=split,method=method,n_calibration=20,n_evaluation=80,**base.metrics(y[eval_idx],p[eval_idx])))
        # CRC prediction sets, with a separately assigned half of source validation used for temperature.
        # The diagnostic controls set miscoverage, not conditional error among accepted predictions.
        a=np.load(project/'inputs/validation_cache'/(run_id+'.npz'),allow_pickle=False);vl=a['logits'].astype(float);vy=a['y_true'].astype(int)
        # Identical split assignment for a checkpoint reused on both targets.
        temp_idx,crc_idx=next(StratifiedKFold(2,shuffle=True,random_state=x['seed']+18000).split(vl,vy))
        tf=calibrate(vl[temp_idx],vy[temp_idx],'temperature');cp=softmax(vl[crc_idx]/tf['temperature'],axis=1);tp=softmax(lp/tf['temperature'],axis=1)
        n=len(crc_idx);nonconf=1-cp[np.arange(n),vy[crc_idx]]
        for alpha in [.05,.10]:
            candidates=np.unique(np.r_[0.,nonconf,1.]);bound=np.array([(np.sum(nonconf>v)+1)/(n+1) for v in candidates]);lam=float(candidates[np.flatnonzero(bound<=alpha)[0]])
            sets=tp>=1-lam;covered=sets[np.arange(len(y)),y];size=sets.sum(1);singleton=size==1;wrong=tp.argmax(1)!=y
            crc.append(dict(**key,alpha=alpha,temperature=tf['temperature'],n_temperature=len(temp_idx),n_source_calibration=n,lambda_=lam,corrected_source_risk=float((np.sum(nonconf>lam)+1)/(n+1)),test_coverage=float(covered.mean()),mean_set_size=float(size.mean()),singleton_fraction=float(singleton.mean()),singleton_error=float(wrong[singleton].mean()) if singleton.any() else np.nan,exchangeability_guarantee_claimed=False,scope='retrospective source-calibrated set-miscoverage diagnostic; validation had already selected checkpoint'))
    # Paired image intervals for source-only calibration changes, shared indices across all configurations/seeds.
    for ex,target in itertools.product(['P4','P5'],['ADT45','OTHER']):
        g=[x for x in e if x['experiment']==ex and x['target']==target and x['source']!=target];y=g[0]['y'];idx=base.boot_indices(y,B,18050+(ex=='P5')*10+(target=='OTHER'))
        for model in base.MODELS:
            runs=sorted([x for x in g if x['model']==model],key=lambda x:x['seed'])
            for method in ['temperature','vector','dirichlet_odir']:
                for metric in ['nll','brier']:
                    diff=[]
                    for x in runs:
                        p=calprob[(ex,x['source'],target,model,x['seed'],method)];raw=x['p'];hot=np.eye(p.shape[1])[y]
                        diff.append((-np.log(np.clip(raw[np.arange(len(y)),y],base.EPS,1))+np.log(np.clip(p[np.arange(len(y)),y],base.EPS,1))) if metric=='nll' else ((raw-hot)**2).sum(1)-((p-hot)**2).sum(1))
                    d=np.mean(diff,axis=0);rep=d[idx].mean(1);lo,hi=np.quantile(rep,[.025,.975]);paired_changes.append(dict(experiment=ex,target=target,model=model,method=method,metric=metric,raw_minus_calibrated=float(d.mean()),image_ci95_lower=lo,image_ci95_upper=hi,n_images=len(y),replicates=B))
    # Class-F1 BCa uses paired stratified image resampling and a weighted within-stratum delete-one jackknife.
    for target in ['ADT45','OTHER']:
        g=[x for x in e if x['experiment']=='P5' and x['target']==target];ref=g[0];y=ref['y'];k=5;n=len(y);idx=base.boot_indices(y,B,18300+(target=='OTHER'))
        def classf(y,pred,w=None):
            cm=np.bincount(y*k+pred,weights=w,minlength=k*k).reshape(k,k);den=cm.sum(0)+cm.sum(1);return np.divide(2*cm.diagonal(),den,out=np.zeros(k),where=den>0)
        def bootclass(pred):
            yy=y[idx];pp=pred[idx];codes=yy*k+pp+np.arange(B)[:,None]*k*k;cm=np.bincount(codes.ravel(),minlength=B*k*k).reshape(B,k,k);den=cm.sum(1)+cm.sum(2);return np.divide(2*np.diagonal(cm,axis1=1,axis2=2),den,out=np.zeros((B,k)),where=den>0)
        for model in base.MODELS:
            c=sorted([x for x in g if x['model']==model and x['source']==target],key=lambda x:x['seed']);t=sorted([x for x in g if x['model']==model and x['source']!=target],key=lambda x:x['seed'])
            point={};rep={};jack={}
            for condition,runs in [('control',c),('transfer',t)]:
                preds=[x['p'].argmax(1) for x in runs];point[condition]=np.mean([classf(y,p) for p in preds],axis=0);rep[condition]=np.mean([bootclass(p) for p in preds],axis=0);vals=[]
                for i in range(n):
                    w=np.ones(n);w[y==y[i]]=20/19;w[i]=0;vals.append(np.mean([classf(y,p,w) for p in preds],axis=0))
                jack[condition]=np.array(vals)
            point['loss']=point['control']-point['transfer'];rep['loss']=rep['control']-rep['transfer'];jack['loss']=jack['control']-jack['transfer']
            for condition,j in itertools.product(['control','transfer','loss'],range(k)):
                lo,hi,status=bca(point[condition][j],rep[condition][:,j],jack[condition][:,j]);bca_rows.append(dict(target=target,model=model,condition=condition,**{'class':ref['classes'][j]},mean_seed_f1=point[condition][j],bca95_lower=lo,bca95_upper=hi,status=status,unique_images_per_class=20,replicates=B,seed_scope='three fixed training realizations'))
    for rows,name in [(cal_rows,'18_source_calibration_metrics'),(corrected,'18_bias_corrected_L2_calibration'),(l1corrected,'18_normal_bias_adjusted_L1_ECE'),(bin_rows,'18_source_fixed_calibration_bins'),(target_rows,'18_target20_label_reference'),(crc,'18_crc_prediction_sets'),(bca_rows,'18_class_F1_BCa'),(paired_changes,'18_paired_calibration_changes'),(new_predictions,'18_calibrated_test_probabilities')]:save(rows,out,name)
    (out/'18_source_calibration_fits.json').write_text(json.dumps(fits,indent=2)+'\n');(out/'18_target_calibration_assignments.json').write_text(json.dumps(target_fits,indent=2)+'\n')
    if do_mixed:mixed_models(e,out)
    config={'bootstrap':B,'primary_scope':'fixed-target macro-F1 loss remains primary; these are secondary retrospective diagnostics','calibration_CV':'3 stratified source-validation folds; lambda grid .001,.01,.1,1,10; no test tuning','vector_input':'centered log probabilities (identical to centered logits up to archived numeric precision)','dirichlet_input':'log probabilities; ODIR penalty, equal off-diagonal and intercept lambda selected by source CV','target_reference':'five disjoint 20-label calibration folds, each tested on its complement of 80; descriptive repetition, not five independent cohorts or a guaranteed upper bound','CRC':'temperature/source-calibration halves split deterministically; prediction-set miscoverage, not selective conditional risk; checkpoint previously selected using full validation; no formal guarantee claimed','calibration_bias':'L2 residual variance correction with source-fixed bins; singleton bins reported as undefined, never silently imputed; not an unbiased L1 ECE claim','BCa':'fixed seed-mean class F1; class-stratified paired image bootstrap; weighted within-class jackknife preserves class mass','retraining_performed':False}
    config.update(normal_bias_adjusted_L1='five adaptive confidence bins; analytic normal-bootstrap expectation; 2*plugin minus expected absolute gap; signed estimates retained; undefined with singleton bins',calibrator_bounds='diagonal .05..20, offdiagonal/intercept -10..10; L-BFGS-B and documented SLSQP fallback')
    (out/'18_config.json').write_text(json.dumps(config,indent=2)+'\n');print('Completed frozen-output analyses',flush=True)

def mixed_models(entries,out):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    import patsy
    coef=[];diagnostic=[];components=[];data=[]
    for target,model in itertools.product(['ADT45','OTHER'],sorted({x['model'] for x in entries})):
        runs=[x for x in entries if x['experiment']=='P5' and x['target']==target and x['model']==model];rows=[]
        for x in runs:
            for i,iid in enumerate(x['df'].image_id):rows.append(dict(image_id=str(iid),seed=str(x['seed']),transfer=int(x['source']!=target),label=x['classes'][x['y'][i]],correct=int(x['p'][i].argmax()==x['y'][i])))
        d=pd.DataFrame(rows);assert len(d)==600 and d.image_id.nunique()==100
        for prior in [.5,1.]:
            np.random.seed(18400);md=BinomialBayesMixedGLM.from_formula('correct ~ transfer + C(label)',{'image':'0 + C(image_id)','seed':'0 + C(seed)','seed_transfer':'0 + transfer:C(seed)'},d,vcp_p=prior,fe_p=2.)
            with warnings.catch_warnings(record=True) as caught:
                f=md.fit_vb(minim_opts={'maxiter':1500,'gtol':1e-6})
            j=md.exog_names.index('transfer');mean=float(f.fe_mean[j]);sd=float(f.fe_sd[j]);success=bool(f.optim_retvals.success)
            coef.append(dict(target=target,model=model,log_sd_prior=prior,transfer_log_odds=mean,posterior_sd=sd,odds_ratio=np.exp(mean),approx_CrI95_lower=np.exp(mean-1.96*sd),approx_CrI95_upper=np.exp(mean+1.96*sd),converged=success))
            diagnostic.append(dict(target=target,model=model,log_sd_prior=prior,success=success,iterations=int(f.optim_retvals.nit),message=str(f.optim_retvals.message),warnings='; '.join(str(w.message) for w in caught),n_rows=len(d),n_images=100,n_seeds=3,approximation='mean-field variational Bayes; intervals are approximate posterior credible intervals, not frequentist confidence intervals'))
            for name,v,sd0 in zip(md.vcp_names,f.vcp_mean,f.vcp_sd):components.append(dict(target=target,model=model,log_sd_prior=prior,component=name,log_sd_posterior_mean=v,log_sd_posterior_sd=sd0))
            print('GLMM',target,model,prior,success,flush=True)
        data.extend([dict(target=target,model=model,**r) for r in rows])
    save(coef,out,'18_crossed_GLMM_transfer_effects');save(diagnostic,out,'18_crossed_GLMM_diagnostics');save(components,out,'18_crossed_GLMM_variance_sensitivity');save(data,out,'18_crossed_GLMM_input')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=5000);p.add_argument('--skip-mixed',action='store_true');a=p.parse_args();main(a.project,a.output,a.bootstrap,not a.skip_mixed)
