"""Two additional cultivar designs and frozen-pretrained domain discrimination.
No Kaggle authentication or credential discovery. Images must already be local.
The 3-class experiments are sensitivities, not replacements for the primary P5.
"""
from pathlib import Path
import json,hashlib,sys,itertools,time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,log_loss,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CLASSES=['blast','dead_heart','hispa']
SEEDS=[2026,2027,2028]
BALANCED_VARIETIES=['AndraPonni','KarnatakaPonni','Onthanel','Ponni']

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(obj,indent=2,default=str)+'\n')

def pair_design(meta,protocol):
    """Freeze allocation from metadata alone, before any model outcomes."""
    rng=np.random.default_rng(19001 if protocol=='P6' else 19002);rows=[]
    a=meta[meta.variety.eq('ADT45')].copy() if protocol=='P6' else meta[meta.variety.eq('AndraPonni')].copy()
    b=meta[meta.variety.isin(BALANCED_VARIETIES)].copy() if protocol=='P6' else meta[~meta.variety.eq('AndraPonni')].copy()
    domains=['ADT45','OTHER_balanced'] if protocol=='P6' else ['AndraPonni','REST']
    for label in CLASSES:
        aa=a[a.label.eq(label)];bb=b[b.label.eq(label)];pairs=[]
        if protocol=='P6':
            ages=sorted(set(aa.age)&set(bb.age));variables=[(v,age) for v in BALANCED_VARIETIES for age in ages]
            bounds=[(0,int(((bb.variety==v)&(bb.age==age)).sum())) for v,age in variables]
            eq=np.array([[int(v==vv) for v,age in variables] for vv in BALANCED_VARIETIES]);ub=np.array([[int(age==ag) for v,age in variables] for ag in ages])
            result=linprog(rng.random(len(variables)),A_ub=ub,b_ub=[int((aa.age==ag).sum()) for ag in ages],A_eq=eq,b_eq=[20]*4,bounds=bounds,method='highs')
            if not result.success:raise ValueError(f'P6 exact-age allocation infeasible for {label}: {result.message}')
            allocation=np.rint(result.x).astype(int)
            if not np.allclose(allocation,result.x,atol=1e-7):raise ValueError('Non-integer network allocation')
            pools={age:list(rng.permutation(aa.loc[aa.age.eq(age),'image_id'].to_numpy())) for age in ages}
            for (v,age),n in zip(variables,allocation):
                chosen=rng.choice(bb.loc[bb.variety.eq(v)&bb.age.eq(age),'image_id'],n,replace=False)
                for iid in chosen:pairs.append((pools[age].pop(),iid,int(age),v))
            # Equal cultivar x class counts in every partition; pairs stay together.
            split=[]
            for v in BALANCED_VARIETIES:
                ids=[i for i,p in enumerate(pairs) if p[3]==v];assert len(ids)==20
                for j,i in enumerate(rng.permutation(ids)):split.append((i,'train' if j<12 else 'validation' if j<16 else 'test'))
            splits=dict(split)
        else:
            for age in sorted(set(aa.age)&set(bb.age)):
                pa=rng.permutation(aa.loc[aa.age.eq(age),'image_id']);pb=rng.permutation(bb.loc[bb.age.eq(age),'image_id'])
                pairs.extend((x,y,int(age),None) for x,y in zip(pa,pb))
            if len(pairs)<50:raise ValueError(f'P7 requires 50 exact-age pairs/class; {label} has {len(pairs)}')
            pairs=[pairs[i] for i in rng.choice(len(pairs),50,replace=False)]
            splits={i:('train' if j<30 else 'validation' if j<40 else 'test') for j,i in enumerate(rng.permutation(50))}
        for i,(ia,ib,age,variety) in enumerate(pairs):
            for domain,iid in zip(domains,[ia,ib]):rows.append(dict(image_id=iid,label=label,age=age,domain=domain,pair_split=splits[i],matched_pair_id=f'{protocol}_{label}_{i:03d}',match_seed=19001 if protocol=='P6' else 19002))
    d=pd.DataFrame(rows).merge(meta.drop(columns=['domain'],errors='ignore'),on=['image_id','label','age'],validate='one_to_one')
    assert d.image_id.nunique()==len(d) and d.sha256.nunique()==len(d)
    assert (d.groupby('matched_pair_id').pair_split.nunique()==1).all()
    assert (d.groupby(['pair_split','label','age','domain']).size().unstack(fill_value=0).nunique(axis=1)==1).all()
    if d.dropna(subset=['visual_group_id']).groupby('visual_group_id').pair_split.nunique().gt(1).any():raise ValueError('Visual group crosses partitions')
    if protocol=='P6':
        c=d[d.domain.eq(domains[1])].groupby(['pair_split','label','variety']).size().unstack(fill_value=0)
        assert (c.nunique(axis=1)==1).all() and list(c.columns)==sorted(BALANCED_VARIETIES)
    else:assert d.loc[d.domain.eq('REST'),'variety'].ne('AndraPonni').all()
    return d.sort_values(['domain','pair_split','label','image_id']).reset_index(drop=True),domains

def prepare(meta_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);meta=pd.read_csv(meta_path);designs={};counts=[]
    for protocol in ['P6','P7']:
        frozen=Path(meta_path).parent/'frozen_designs'/f'{protocol}_assignments.csv'
        if frozen.exists():
            manifest=json.loads((frozen.parent/'SHA256.json').read_text())
            if digest(frozen)!=manifest[frozen.name]:raise ValueError('Asignaciones originales modificadas')
            payload=frozen.read_bytes();d=pd.read_csv(frozen)
            domains=['ADT45','OTHER_balanced'] if protocol=='P6' else ['AndraPonni','REST']
            reference=meta.set_index('image_id').loc[d.image_id].reset_index()
            for col in ['label','age','variety','sha256']:
                if not (reference[col].to_numpy()==d[col].to_numpy()).all():raise ValueError('Asignaciones y metadata no coinciden: '+col)
            if d.image_id.duplicated().any() or d.sha256.duplicated().any():raise ValueError('Imágenes repetidas')
            if d.groupby('matched_pair_id').size().ne(2).any():raise ValueError('Pareja incompleta')
            if d.groupby('matched_pair_id').pair_split.nunique().ne(1).any():raise ValueError('Pareja en dos splits')
            if not (d.groupby(['pair_split','label','age','domain']).size().unstack(fill_value=0).nunique(axis=1)==1).all():raise ValueError('Edades no emparejadas')
            if d.dropna(subset=['visual_group_id']).groupby('visual_group_id').pair_split.nunique().gt(1).any():raise ValueError('Grupo visual cruza particiones')
        else:
            d,domains=pair_design(meta,protocol);payload=d.to_csv(index=False).encode()
        p=out/f'{protocol}_assignments.csv'
        if p.exists() and p.read_bytes()!=payload:
            old=pd.read_csv(p)
            try:pd.testing.assert_frame_equal(old,d,check_dtype=False)
            except AssertionError as e:raise RuntimeError('Las asignaciones guardadas difieren; no se sobrescriben.') from e
        p.write_bytes(payload);designs[protocol]=(d,domains)
        counts.append(dict(protocol=protocol,n_images=len(d),n_classes=3,domains=domains,assignments_sha256=digest(p),seed=int(d.match_seed.iloc[0])))
        d.groupby(['domain','pair_split','label']).size().rename('n').reset_index().to_csv(out/f'{protocol}_counts.csv',index=False)
    write(out/'design_manifest.json',{'metadata_sha256':digest(meta_path),'designs':counts,'selection':'metadata-only frozen design; three classes supported by both sensitivities','no_external_validation':True})
    return meta,designs

def run_training(meta_path,out,image_root,model_root,weight_cache,workers=2):
    import paddy19_training_core as core
    core.load_torch()
    if not core.torch.cuda.is_available():raise RuntimeError('Activa una GPU para los 48 entrenamientos nuevos.')
    if core.timm.__version__!='1.0.28':raise RuntimeError('Se requiere timm 1.0.28')
    out=Path(out);meta,designs=prepare(meta_path,out/'design');core.CLASSES=CLASSES
    # Verify available images before requesting any pretrained downloads.
    lookup,receipt=core.prepare_images(meta,{k:d for k,(d,ds) in designs.items()},image_root,'/content/paddy19_image_cache')
    receipt.to_csv(out/'verified_training_images.csv',index=False)
    core.ensure_pretrained(weight_cache)
    env=core.environment();core.freeze_json(out/'environment_lock.json',env)
    tasks_done=0;elapsed_start=time.perf_counter()
    for protocol,(d,domains) in designs.items():
        core.DOMAINS=domains;root=out/protocol;root.mkdir(exist_ok=True)
        plan=dict(protocol=protocol,classes=CLASSES,domains=domains,training_seeds=SEEDS,recipe=core.RECIPE,models=core.MODELS,environment=env,workers=workers,assignments_sha256=digest(out/'design'/f'{protocol}_assignments.csv'),source_sha256=digest(__file__),core_sha256=digest(core.__file__),pretrained={k:json.loads((Path(weight_cache)/(k+'.json')).read_text()) for k in core.MODELS})
        core.freeze_json(root/'experiment_plan.json',plan);signature=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        for key,seed,domain in itertools.product(core.MODELS,SEEDS,domains):
            print(f'[{tasks_done+1}/48] {protocol} | {key} | seed={seed} | fuente={domain}',flush=True)
            core.run_one(d,int(d.match_seed.iloc[0]),key,seed,domain,lookup,root,Path(model_root)/protocol,weight_cache,signature,workers)
            tasks_done+=1
            print(f'Avance: {tasks_done}/48; tiempo de esta sesión: {(time.perf_counter()-elapsed_start)/60:.1f} min',flush=True)
        core.summarize(root,expected_runs=24)
    summarize(out)

def summarize(out,B=5000):
    out=Path(out);results=[];intervals=[]
    for protocol in ['P6','P7']:
        runs=[]
        for marker in sorted((out/protocol/'runs').rglob('COMPLETE.json')):
            for p in sorted(marker.parent.glob('eval__*/predictions.csv')):
                d=pd.read_csv(p);prov=json.loads((marker.parent/'provenance.json').read_text());runs.append((prov,p.parent.name[6:],d.sort_values('image_id').reset_index(drop=True)))
        if len(runs)!=48:raise RuntimeError(f'{protocol}: {len(runs)}/48 evaluations; not complete')
        for target,model in itertools.product(sorted({x[1] for x in runs}),sorted({x[0]['model_key'] for x in runs})):
            selected=[r for r in runs if r[1]==target and r[0]['model_key']==model];y=selected[0][2].label.to_numpy();ids=selected[0][2].image_id.tolist();rng=np.random.default_rng(19100+(protocol=='P7'))
            index=np.concatenate([rng.choice(np.flatnonzero(y==cl),size=(B,int((y==cl).sum())),replace=True) for cl in CLASSES],axis=1);rep=[];point=[]
            for seed in SEEDS:
                c=next(d for p,t,d in selected if p['train_domain']==target and p['train_seed']==seed);t=next(d for p,t,d in selected if p['train_domain']!=target and p['train_seed']==seed)
                assert ids==c.image_id.tolist()==t.image_id.tolist()
                cp=c.y_pred_raw.to_numpy();tp=t.y_pred_raw.to_numpy();cf=f1_score(y,cp,labels=CLASSES,average='macro',zero_division=0);tf=f1_score(y,tp,labels=CLASSES,average='macro',zero_division=0)
                point.append(cf-tf);results.append(dict(protocol=protocol,target=target,model=model,seed=seed,control_f1=cf,transfer_f1=tf,loss=cf-tf,n_unique_test=len(y)))
                def bootf(pred):
                    yy=np.searchsorted(CLASSES,y)[index];pp=np.searchsorted(CLASSES,pred)[index];cm=np.bincount((yy*3+pp+np.arange(B)[:,None]*9).ravel(),minlength=B*9).reshape(B,3,3);den=cm.sum(1)+cm.sum(2);return np.divide(2*np.diagonal(cm,axis1=1,axis2=2),den,out=np.zeros((B,3)),where=den>0).mean(1)
                rep.append(bootf(cp)-bootf(tp))
            lo,hi=np.quantile(np.mean(rep,axis=0),[.025,.975]);intervals.append(dict(protocol=protocol,target=target,model=model,mean_seed_loss=np.mean(point),image_ci95_lower=lo,image_ci95_upper=hi,replicates=B,n_unique_test=len(y),scope='three fixed seeds; secondary 3-class sensitivity'))
    pd.DataFrame(results).to_csv(out/'19_losses_by_seed.csv',index=False);pd.DataFrame(intervals).to_csv(out/'19_paired_image_intervals.csv',index=False)

def domain_embeddings(meta_path,p4_path,p5_path,image_root,out,weight_cache,workers=2,B=5000):
    """Frozen ResNet features; scaler/train C selection on train/validation only."""
    import paddy19_training_core as core
    core.load_torch();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(meta_path);p4=pd.read_csv(p4_path).rename(columns={'domain_split':'pair_split'});p5=pd.read_csv(p5_path)
    if 'pair_split' not in p5 and 'split' in p5:p5=p5.rename(columns={'split':'pair_split'})
    specs={'P4_same5':p4[p4.label.isin(['blast','dead_heart','downy_mildew','hispa','normal'])].copy(),'P5':p5.copy()}
    selected=pd.concat([d for d in specs.values()]).drop_duplicates('image_id')
    lookup,receipt=core.prepare_images(meta,{'embedding':selected},image_root,'/content/paddy19_image_cache');receipt.to_csv(out/'verified_embedding_images.csv',index=False)
    core.ensure_pretrained(weight_cache,models={'resnet50':core.MODELS['resnet50']})
    cfg=json.loads((Path(weight_cache)/'resnet50.json').read_text());signature=dict(metadata_sha256=digest(meta_path),p4_assignments=digest(p4_path),p5_assignments=digest(p5_path),pretrained=cfg,environment=core.environment(),source_sha256=digest(__file__),transform='resize256 center224 bicubic ImageNet norm',tag=core.MODELS['resnet50'])
    core.freeze_json(out/'embedding_plan.json',signature)
    cache=out/'frozen_resnet_features.npz';df=selected.sort_values('image_id').reset_index(drop=True);core.CLASSES=sorted(meta.label.unique())
    if cache.exists():
        receipt_path=out/'features_sha256.json'
        if not receipt_path.exists() or json.loads(receipt_path.read_text())['sha256']!=digest(cache):raise RuntimeError('Caché de embeddings incompleta o modificada; retira frozen_resnet_features.npz y features_sha256.json para recalcularla.')
        a=np.load(cache,allow_pickle=False);assert a['image_ids'].tolist()==df.image_id.tolist();features=a['features']
    else:
        device=core.torch.device('cuda' if core.torch.cuda.is_available() else 'cpu');model=core.timm.create_model(core.MODELS['resnet50'],pretrained=False,num_classes=1000)
        model.load_state_dict(core.torch.load(Path(weight_cache)/'resnet50.pt',weights_only=True,map_location='cpu'));model.reset_classifier(0);model.to(device).eval();chunks=[]
        with core.torch.no_grad():
            for x,y in core.loader(df,lookup,19000,False,workers):chunks.append(model(x.to(device)).float().cpu().numpy())
        features=np.concatenate(chunks);tmp=cache.with_suffix('.tmp.npz');np.savez_compressed(tmp,features=features,image_ids=df.image_id.to_numpy(dtype=str));tmp.replace(cache);write(out/'features_sha256.json',dict(sha256=digest(cache)));del model
    if features.ndim!=2 or len(features)!=len(df) or not np.isfinite(features).all():raise ValueError('Embeddings inválidos')
    print('Embeddings completos:',features.shape,flush=True)
    fi={iid:i for i,iid in enumerate(df.image_id)};rows=[];predrows=[]
    for protocol,d in specs.items():
        print('Clasificador de dominio:',protocol,flush=True)
        d=d.sort_values('image_id').reset_index(drop=True);x=features[[fi[i] for i in d.image_id]];y=d.domain.eq('OTHER').astype(int).to_numpy();tr=d.pair_split.eq('train').to_numpy();va=d.pair_split.eq('validation').to_numpy();te=d.pair_split.eq('test').to_numpy()
        assert tr.any() and va.any() and te.any() and set(d.domain)=={'ADT45','OTHER'}
        scaler=StandardScaler().fit(x[tr]);xx=scaler.transform(x);cv=[]
        for c in [.001,.01,.1,1.,10.]:
            clf=LogisticRegression(C=c,max_iter=5000,solver='lbfgs',random_state=19000).fit(xx[tr],y[tr])
            if np.max(clf.n_iter_)>=5000:raise RuntimeError('El clasificador de dominio no convergió')
            cv.append((log_loss(y[va],clf.predict_proba(xx[va])),c,clf))
        vl,c,clf=min(cv,key=lambda z:(z[0],z[1]));prob=clf.predict_proba(xx[te])[:,1];yt=y[te];td=d.loc[te].reset_index(drop=True);auc=roc_auc_score(yt,prob);ba=balanced_accuracy_score(yt,prob>=.5);rng=np.random.default_rng(19200+(protocol=='P5'));strata=td.groupby(['domain','label']).indices
        idx=np.concatenate([rng.choice(ii,size=(B,len(ii)),replace=True) for ii in strata.values()],axis=1);boot=np.array([roc_auc_score(yt[ii],prob[ii]) for ii in idx]);lo,hi=np.quantile(boot,[.025,.975])
        rows.append(dict(protocol=protocol,n_train=int(tr.sum()),n_validation=int(va.sum()),n_test=int(te.sum()),selected_C=c,validation_nll=vl,test_auc=auc,auc_ci95_lower=lo,auc_ci95_upper=hi,test_balanced_accuracy=ba,replicates=B,scope='held-out separability of observed cultivar domains; not causal cultivar effect'))
        for j,r in td.iterrows():predrows.append(dict(protocol=protocol,image_id=r.image_id,label=r.label,domain=r.domain,prob_OTHER=prob[j]))
        np.savez_compressed(out/(protocol+'_domain_classifier.npz'),coef=clf.coef_,intercept=clf.intercept_,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,C=c)
    pd.DataFrame(rows).to_csv(out/'19_domain_embedding_metrics.csv',index=False);pd.DataFrame(predrows).to_csv(out/'19_domain_embedding_predictions.csv',index=False)

def export(out,destination):
    import zipfile
    out=Path(out);destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix not in ['.pt','.tmp'] and p.resolve()!=destination.resolve():z.write(p,p.relative_to(out))
    print('Resultados exportados:',destination,'SHA256',digest(destination))

def audit_historical(project,inventory_path,out):
    """Hash all available historical trained files; never infer an initial weight hash."""
    import torch
    folders={'P1':'05_P1_30ep_noES','P4':'06_P4_cross_cultivar_30ep_noES','P5':'07_P5_exact_age_matched_30ep_noES'};rows=[]
    for r in json.loads(Path(inventory_path).read_text()):
        p=Path(project)/'models'/folders[r['protocol']]/r['run']/'best_model.pt';row=dict(protocol=r['protocol'],run=r['run'],checkpoint_file_id=r['files'][0]['id'],exists=p.is_file())
        if p.is_file():
            try:
                from torch.torch_version import TorchVersion
                with torch.serialization.safe_globals([TorchVersion]):c=torch.load(p,map_location='cpu',weights_only=True)
                row.update(sha256=digest(p),size=p.stat().st_size,model=c.get('model_key'),seed=c.get('seed'),selected_epoch=c.get('epoch'),best_val_macro_f1=c.get('best_val_macro_f1'),historical_initialization_hash_retained=False)
                del c
            except Exception as e:row.update(audit_error=type(e).__name__,audit_message=str(e)[:300])
        rows.append(row)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'19_historical_checkpoint_audit.csv',index=False)
    print('Checkpoints encontrados y auditados:',sum(r['exists'] for r in rows),'/',len(rows))
