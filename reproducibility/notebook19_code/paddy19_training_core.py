"""Paddy Doctor training core for P6/P7, version 19.1.0.
Fresh network fits; frozen class/age matching; source-validation model selection.
No claim of independent acquisition cohorts or external validation.
"""
from pathlib import Path
import os, sys, json, hashlib, time, random, gc, shutil, platform, itertools
import numpy as np
import pandas as pd
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix

VERSION = '19.2.0'
CLASSES = ['blast', 'dead_heart', 'downy_mildew', 'hispa', 'normal']
DOMAINS = ['ADT45', 'OTHER']
MATCH_SEEDS = [31001, 31002, 31003, 31004, 31005]
TRAIN_SEEDS = [2026, 2027, 2028]
MODELS = {
    'resnet50': 'resnet50.a1_in1k',
    'convnext_tiny': 'convnext_tiny.in12k_ft_in1k',
    'vit_small_patch16_224': 'vit_small_patch16_224.augreg_in21k_ft_in1k',
    'swin_tiny_patch4_window7_224': 'swin_tiny_patch4_window7_224.ms_in1k',
}
RECIPE = dict(epochs=30, image_size=224, resize_size=256, batch_size=32,
              lr=3e-4, weight_decay=.05, warmup_epochs=2,
              warmup_start_factor=.1, cosine_eta_min=3e-6,
              crop_scale=[.85, 1.], crop_ratio=[.75, 4/3], hflip=.5,
              rotation_degrees=10, jitter=[.1, .1, .1, .02],
              mean=[.485, .456, .406], std=[.229, .224, .225],
              early_stopping=False, selection='source_validation_macro_f1',
              improvement_tolerance=1e-6, amp='fp16_on_cuda',
              loader_randomness='deterministic per-epoch seed; workers recreated each epoch',
              resume='completed runs skipped; interrupted run restarts from pretrained initialization')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def atomic_bytes(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, obj):
    atomic_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str)+'\n').encode())


def write_csv(path, df):
    atomic_bytes(path, df.to_csv(index=False, lineterminator='\n').encode())


def freeze_json(path, obj):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != obj:
            raise RuntimeError(f'Cambió la configuración o el entorno: {path}. No mezclar resultados. Usa otro EXPERIMENT_NAME para un experimento diferente.')
    else: write_json(path, obj)


def load_metadata(path, expected_n=10329):
    d = pd.read_csv(path, dtype={'image_id':str, 'label':str, 'variety':str, 'sha256':str})
    required={'image_id','label','variety','age','sha256'}
    if not required.issubset(d): raise ValueError(f'Metadata sin columnas: {required-set(d)}')
    if expected_n is not None and len(d)!=expected_n: raise ValueError(f'Se esperaban {expected_n} imágenes, hay {len(d)}')
    if d[list(required)].isna().any().any() or d.image_id.duplicated().any(): raise ValueError('Metadata incompleta o IDs repetidos')
    if d.sha256.duplicated().any(): raise ValueError('Hay duplicados exactos en el pool near-clean')
    age=pd.to_numeric(d.age, errors='raise')
    if not np.isfinite(age).all() or (age!=np.floor(age)).any(): raise ValueError('Edad debe ser entera')
    d['age']=age.astype(int)
    if any(Path(x).name!=x for x in d.image_id): raise ValueError('image_id debe ser solo un nombre de archivo')
    d['domain']=np.where(d.variety.eq('ADT45'),'ADT45','OTHER')
    return d.sort_values('image_id').reset_index(drop=True)


def construct_cohort(meta, match_seed):
    """New pairing without replacement within cohort; class-stratified pair split 60/20/20."""
    rng=np.random.default_rng(match_seed); records=[]
    support=meta.groupby(['label','age','domain']).size().unstack(fill_value=0).reindex(columns=DOMAINS,fill_value=0)
    eligible=support.min(axis=1).groupby(level=0).sum()
    if sorted(eligible[eligible>=100].index)!=CLASSES: raise ValueError('Las clases elegibles no coinciden con el P5 congelado')
    for label in CLASSES:
        candidates=[]
        sub=meta[meta.label.eq(label)]
        for age in sorted(sub.age.unique()):
            pools=[sub[sub.age.eq(age)&sub.domain.eq(domain)].sort_values('image_id') for domain in DOMAINS]
            pools=[p.iloc[rng.permutation(len(p))].reset_index(drop=True) for p in pools]
            for i in range(min(map(len,pools))):
                candidates.append(dict(label=label, age=int(age), ADT45_image_id=pools[0].iloc[i].image_id,
                                       OTHER_image_id=pools[1].iloc[i].image_id))
        chosen=[candidates[i] for i in rng.choice(len(candidates),100,replace=False)]
        chosen=sorted(chosen,key=lambda r:(r['age'],r['ADT45_image_id'],r['OTHER_image_id']))
        order=rng.permutation(100); split=np.empty(100,dtype=object)
        split[order[:60]]='train';split[order[60:80]]='validation';split[order[80:]]='test'
        for i,r in enumerate(chosen):
            r.update(matched_pair_id=f'M{match_seed}_{label}_{i:03d}',pair_split=split[i],match_seed=int(match_seed))
            records.append(r)
    pairs=pd.DataFrame(records);lookup=meta.set_index('image_id');rows=[]
    for r in records:
        for domain in DOMAINS:
            image_id=r[f'{domain}_image_id'];m=lookup.loc[image_id]
            rows.append(dict(image_id=image_id,label=r['label'],age=r['age'],variety=m.variety,
                             domain=domain,sha256=m.sha256,matched_pair_id=r['matched_pair_id'],
                             pair_split=r['pair_split'],match_seed=int(match_seed),
                             visual_group_id=m.get('visual_group_id',None)))
    long=pd.DataFrame(rows).sort_values(['domain','pair_split','label','image_id']).reset_index(drop=True)
    validate_cohort(long)
    return pairs,long


def validate_cohort(long):
    if len(long)!=1000 or long.image_id.nunique()!=1000: raise ValueError('P5 debe tener 1000 imágenes únicas')
    if long.sha256.nunique()!=1000: raise ValueError('Duplicados SHA dentro de la cohorte')
    for split,n in [('train',60),('validation',20),('test',20)]:
        sub=long[long.pair_split.eq(split)]
        counts=sub.groupby(['domain','label']).size().reindex(pd.MultiIndex.from_product([DOMAINS,CLASSES]),fill_value=0)
        if not counts.eq(n).all(): raise ValueError('Tamaño por clase/dominio incorrecto')
        age=sub.groupby(['label','age','domain']).size().unstack(fill_value=0).reindex(columns=DOMAINS,fill_value=0)
        if not age.ADT45.eq(age.OTHER).all(): raise ValueError('Edad no coincide exactamente entre dominios')
    if long.groupby('matched_pair_id').size().ne(2).any(): raise ValueError('Cada par debe contener dos imágenes')
    for col in ['pair_split','label','age']:
        if long.groupby('matched_pair_id')[col].nunique().ne(1).any(): raise ValueError(f'Par incoherente: {col}')
    if 'visual_group_id' in long:
        grouped=long.dropna(subset=['visual_group_id']).groupby('visual_group_id').pair_split.nunique()
        if grouped.gt(1).any(): raise ValueError('Grupo visual cruza particiones: revisar pool near-clean')


def prepare_design(meta, output, match_seeds=MATCH_SEEDS, reference=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);cohorts={};rows=[]
    for seed in match_seeds:
        pairs,long=construct_cohort(meta,seed);cohorts[seed]=long
        folder=output/f'match_{seed}';folder.mkdir(exist_ok=True)
        for name,df in [('pairs.csv',pairs),('assignments.csv',long)]:
            payload=df.to_csv(index=False,lineterminator='\n').encode();p=folder/name
            if p.exists() and p.read_bytes()!=payload: raise RuntimeError(f'Asignaciones congeladas diferentes: {p}')
            if not p.exists():atomic_bytes(p,payload)
        rows.append(dict(match_seed=seed,n_images=1000,n_pairs=500,assignments_sha256=sha(folder/'assignments.csv')))
    comparison=dict(cohorts)
    if reference is not None:comparison['original_2026']=reference
    overlap=[]
    for (a,da),(b,db) in itertools.combinations(comparison.items(),2):
        for split in ['all','train','validation','test']:
            aa=set(da.image_id if split=='all' else da.loc[da.pair_split.eq(split),'image_id'])
            bb=set(db.image_id if split=='all' else db.loc[db.pair_split.eq(split),'image_id'])
            overlap.append(dict(cohort_a=a,cohort_b=b,partition=split,shared_images=len(aa&bb),union_images=len(aa|bb),jaccard=len(aa&bb)/len(aa|bb)))
        for sa,sb in [('train','test'),('test','train')]:
            aa=set(da.loc[da.pair_split.eq(sa),'image_id']);bb=set(db.loc[db.pair_split.eq(sb),'image_id'])
            overlap.append(dict(cohort_a=a,cohort_b=b,partition=f'{sa}_vs_{sb}',shared_images=len(aa&bb),union_images=len(aa|bb),jaccard=len(aa&bb)/len(aa|bb)))
    write_csv(output/'overlap_between_cohorts.csv',pd.DataFrame(overlap))
    write_csv(output/'design_manifest.csv',pd.DataFrame(rows))
    write_csv(output/'candidate_support.csv',meta.groupby(['label','age','domain']).size().rename('n').reset_index())
    return cohorts


def calibration_errors(y,p,bins=15,adaptive=False):
    confidence=p.max(1); correct=(p.argmax(1)==y).astype(float)
    edges=np.unique(np.quantile(confidence,np.linspace(0,1,bins+1))) if adaptive else np.linspace(0,1,bins+1)
    if len(edges)<2:return float(abs(correct.mean()-confidence.mean())),float(abs(correct.mean()-confidence.mean()))
    errors=[];weights=[]
    for j in range(len(edges)-1):
        use=(confidence>=edges[j] if j==0 else confidence>edges[j]) & (confidence<=edges[j+1])
        if use.any():errors.append(abs(correct[use].mean()-confidence[use].mean()));weights.append(use.mean())
    return float(np.dot(errors,weights)),float(max(errors,default=0.))


def score(y,logits,temperature=1.):
    p=softmax(logits/temperature,axis=1);pred=p.argmax(1)
    ece,mce=calibration_errors(y,p);adaptive,_=calibration_errors(y,p,10,True)
    return dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                macro_f1=float(f1_score(y,pred,labels=range(len(CLASSES)),average='macro',zero_division=0)),
                nll=float(np.mean(logsumexp(logits/temperature,axis=1)-(logits/temperature)[np.arange(len(y)),y])),
                brier=float(np.mean(np.sum((p-np.eye(len(CLASSES))[y])**2,axis=1))),
                ece_15=ece,adaptive_ece_10=adaptive,mce_15=mce)


def fit_temperature(y,logits):
    result=minimize_scalar(lambda u:score(y,logits,np.exp(u))['nll'],method='bounded',
                           bounds=(np.log(.05),np.log(20)),options={'xatol':1e-5})
    if not result.success or not np.isfinite(result.fun):raise RuntimeError('Falló el ajuste de temperatura fuente')
    return float(np.exp(result.x)),dict(success=True,log_temperature=float(result.x),source_nll=float(result.fun),nfev=int(result.nfev))


def load_torch():
    global torch,timm,T,InterpolationMode,Image,DataLoader
    import torch,timm
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    from PIL import Image
    from torch.utils.data import DataLoader


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def state_hash(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        t=t.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def atomic_torch(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)


def environment():
    import importlib.metadata
    return dict(python=platform.python_version(),packages={x:importlib.metadata.version(x) for x in
                ['torch','torchvision','timm','numpy','pandas','scipy','scikit-learn','Pillow']},
                cuda=str(torch.version.cuda),cudnn=int(torch.backends.cudnn.version() or 0),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')


def ensure_pretrained(cache,models=MODELS):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    for key,tag in models.items():
        p=cache/(key+'.pt');manifest=cache/(key+'.json')
        if p.exists() and manifest.exists():
            info=json.loads(manifest.read_text())
            if info['tag']!=tag or sha(p)!=info['file_sha256']:raise RuntimeError('Caché de pesos inconsistente')
            print('Pesos verificados:',tag);continue
        print('Descargando una vez:',tag,flush=True)
        seed_all(0)
        model=timm.create_model(tag,pretrained=True,num_classes=1000).cpu()
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        cfg=json.loads(json.dumps(model.pretrained_cfg,default=str))
        atomic_torch(p,state)
        write_json(manifest,dict(tag=tag,file_sha256=sha(p),tensor_sha256=state_hash(state),pretrained_cfg=cfg))
        del model,state;gc.collect()


def transforms(training):
    if training:
        return T.Compose([T.Resize(256,interpolation=InterpolationMode.BICUBIC),
            T.RandomResizedCrop(224,scale=(.85,1.),ratio=(.75,4/3),interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(.5),T.RandomRotation(10,interpolation=InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=.1,contrast=.1,saturation=.1,hue=.02),T.ToTensor(),T.Normalize(RECIPE['mean'],RECIPE['std'])])
    return T.Compose([T.Resize(256,interpolation=InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize(RECIPE['mean'],RECIPE['std'])])


class ImageDataset:
    def __init__(self,df,lookup,training=False):
        self.df=df.reset_index(drop=True);self.lookup=lookup;self.transform=transforms(training)
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(self.lookup[r.image_id]) as im:x=self.transform(im.convert('RGB'))
        return x,CLASSES.index(r.label)


def worker_seed(worker_id):
    seed=torch.initial_seed()%(2**32);np.random.seed(seed);random.seed(seed)


def loader(df,lookup,seed,training,workers=2):
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(ImageDataset(df,lookup,training),batch_size=RECIPE['batch_size'],shuffle=training,
                      generator=generator,num_workers=workers,pin_memory=torch.cuda.is_available(),drop_last=False,
                      persistent_workers=False,worker_init_fn=worker_seed)


def predict(model,data_loader,device):
    model.eval();ys=[];zs=[]
    with torch.no_grad():
        for x,y in data_loader:
            with torch.amp.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                z=model(x.to(device,non_blocking=True))
            ys.append(y.numpy());zs.append(z.float().cpu().numpy().astype(np.float64))
    return np.concatenate(ys),np.concatenate(zs)


def fit(model,train_df,val_df,lookup,seed,device,history_path,workers=2,epochs=None):
    epochs=epochs or RECIPE['epochs'];optimizer=torch.optim.AdamW(model.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
    warm=torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=.1,end_factor=1.,total_iters=2)
    cosine=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,epochs-2),eta_min=3e-6)
    scheduler=torch.optim.lr_scheduler.SequentialLR(optimizer,[warm,cosine],milestones=[2])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');criterion=torch.nn.CrossEntropyLoss()
    best=-np.inf;best_epoch=None;best_state=None;history=[]
    val_loader=loader(val_df,lookup,seed,False,workers)
    for epoch in range(1,epochs+1):
        epoch_seed=(int(seed)*100003+epoch)%(2**32)
        seed_all(epoch_seed)
        train_loader=loader(train_df,lookup,epoch_seed,True,workers)
        start=time.perf_counter();model.train();loss_sum=0.;n=0
        for x,y in train_loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                logits=model(x);loss=criterion(logits,y)
            if not torch.isfinite(loss):raise RuntimeError('Loss no finita; se detiene sin marcar COMPLETE')
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            loss_sum+=float(loss.detach())*len(y);n+=len(y)
        yv,zv=predict(model,val_loader,device);metrics=score(yv,zv)
        if not np.isfinite(zv).all():raise RuntimeError('Logits de validación no finitos')
        if metrics['macro_f1']>best+RECIPE['improvement_tolerance']:
            best=metrics['macro_f1'];best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(dict(epoch=epoch,train_loss=loss_sum/n,val_loss=metrics['nll'],val_macro_f1=metrics['macro_f1'],
                            val_accuracy=metrics['accuracy'],learning_rate=optimizer.param_groups[0]['lr'],epoch_seconds=time.perf_counter()-start))
        write_csv(history_path,pd.DataFrame(history));scheduler.step()
        if epoch==1 or epoch%5==0:print(f'  época {epoch:02}/{epochs}: val F1={metrics["macro_f1"]:.4f}; mejor={best:.4f}',flush=True)
    model.load_state_dict(best_state)
    return dict(best_epoch=best_epoch,best_val_macro_f1=best,epochs_completed=epochs),best_state


def export_predictions(model,df,lookup,seed,device,destination,temperature,workers=2):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    y,z=predict(model,loader(df,lookup,seed,False,workers),device)
    if not np.isfinite(z).all():raise RuntimeError('Logits no finitos')
    raw=softmax(z,axis=1);scaled=softmax(z/temperature,axis=1)
    data=df.reset_index(drop=True).copy();data['y_true_idx']=y
    for name,p in [('raw',raw),('TS',scaled)]:
        data[f'y_pred_{name}']=[CLASSES[i] for i in p.argmax(1)];data[f'confidence_{name}']=p.max(1)
        for i,label in enumerate(CLASSES):data[f'prob_{name}__{label}']=p[:,i]
    for i,label in enumerate(CLASSES):data[f'logit__{label}']=z[:,i]
    write_csv(destination/'predictions.csv',data)
    np.savez_compressed(destination/'logits.npz',y_true=y,logits=z,image_id=df.image_id.to_numpy(dtype=str),class_names=np.array(CLASSES))
    precision,recall,f1,support=precision_recall_fscore_support(y,raw.argmax(1),labels=range(len(CLASSES)),zero_division=0)
    write_csv(destination/'per_class.csv',pd.DataFrame(dict(class_name=CLASSES,precision=precision,recall=recall,f1=f1,support=support)))
    cm=pd.DataFrame(confusion_matrix(y,raw.argmax(1),labels=range(len(CLASSES))),columns=CLASSES);cm.insert(0,'true_class',CLASSES)
    write_csv(destination/'confusion_counts.csv',cm)
    metrics=dict(raw=score(y,z),TS=score(y,z,temperature));write_json(destination/'metrics.json',metrics)
    return metrics


def prepare_images(meta,cohorts,image_root,cache_root):
    from PIL import Image
    image_root=Path(image_root);cache_root=Path(cache_root);cache_root.mkdir(parents=True,exist_ok=True)
    selected=set().union(*(set(d.image_id) for d in cohorts.values()))
    lookup={};metadata=meta.set_index('image_id');rows=[]
    for i,image_id in enumerate(sorted(selected),1):
        row=metadata.loc[image_id];source=image_root/row.label/image_id;dest=cache_root/row.label/image_id
        if not dest.exists():
            if not source.is_file():raise FileNotFoundError(f'No se encuentra {source}')
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        digest=sha(dest)
        if digest!=row.sha256:raise ValueError(f'Los bytes de {image_id} no coinciden con el pool auditado. No se entrena.')
        with Image.open(dest) as image:image.verify()
        lookup[image_id]=str(dest);rows.append(dict(image_id=image_id,sha256=digest))
        if i%500==0:print(f'  imágenes verificadas: {i}/{len(selected)}',flush=True)
    return lookup,pd.DataFrame(rows)


def complete_ok(run_dir,signature,model_dir):
    marker=Path(run_dir)/'COMPLETE.json'
    if not marker.exists():return False
    payload=json.loads(marker.read_text())
    if payload['signature']!=signature:raise RuntimeError('COMPLETE corresponde a otra configuración')
    for item in payload['files']:
        base=Path(model_dir) if item['area']=='models' else Path(run_dir)
        p=base/item['path']
        if not p.is_file() or sha(p)!=item['sha256']:raise RuntimeError(f'Corrida marcada completa pero archivo ausente/cambiado: {p}')
    return True


def run_one(cohort,match_seed,key,train_seed,train_domain,lookup,root,model_root,weight_cache,signature,workers=2,device=None):
    run_name=f'match_{match_seed}/train_{train_domain}__{key}__seed_{train_seed}'
    run_dir=Path(root)/'runs'/run_name;model_dir=Path(model_root)/run_name
    if complete_ok(run_dir,signature,model_dir):print('Completado, se omite:',run_name,flush=True);return
    run_dir.mkdir(parents=True,exist_ok=True);model_dir.mkdir(parents=True,exist_ok=True)
    print('\nENTRENANDO',run_name,flush=True);seed_all(train_seed)
    start=time.perf_counter();device=torch.device(device or 'cuda')
    model=timm.create_model(MODELS[key],pretrained=False,num_classes=1000)
    model.load_state_dict(torch.load(Path(weight_cache)/(key+'.pt'),map_location='cpu',weights_only=True))
    model.reset_classifier(len(CLASSES))
    initialization_hash=state_hash(model.state_dict())
    model=model.to(device)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    train=cohort[cohort.domain.eq(train_domain)&cohort.pair_split.eq('train')].reset_index(drop=True)
    val=cohort[cohort.domain.eq(train_domain)&cohort.pair_split.eq('validation')].reset_index(drop=True)
    provenance=dict(signature=signature,match_seed=match_seed,model_key=key,tag=MODELS[key],train_seed=train_seed,
        train_domain=train_domain,initialized_tensor_sha256=initialization_hash,
        pretrained=json.loads((Path(weight_cache)/(key+'.json')).read_text()),
        n_parameters=sum(p.numel() for p in model.parameters()),recipe=RECIPE)
    write_json(run_dir/'provenance.json',provenance)
    best,best_state=fit(model,train,val,lookup,train_seed,device,run_dir/'history.csv',workers)
    checkpoint=model_dir/'best_model.pt'
    atomic_torch(checkpoint,dict(model_state_dict=best_state,provenance=provenance,best=best,class_names=CLASSES))
    yv,zv=predict(model,loader(val,lookup,train_seed,False,workers),device)
    temperature,diagnostics=fit_temperature(yv,zv)
    write_json(run_dir/'temperature.json',dict(temperature=temperature,fit_domain=train_domain,diagnostics=diagnostics,
               note='source validation reused for checkpoint and temperature; no formal calibration guarantee'))
    export_predictions(model,val,lookup,train_seed,device,run_dir/'source_validation',temperature,workers)
    evaluations=[]
    for target in DOMAINS:
        test=cohort[cohort.domain.eq(target)&cohort.pair_split.eq('test')].reset_index(drop=True)
        metrics=export_predictions(model,test,lookup,train_seed,device,run_dir/f'eval__{target}',temperature,workers)
        evaluations.append(dict(match_seed=match_seed,model_key=key,train_seed=train_seed,train_domain=train_domain,
                                eval_domain=target,**{f'{cal}_{name}':value for cal,m in metrics.items() for name,value in m.items()}))
    write_csv(run_dir/'evaluations.csv',pd.DataFrame(evaluations))
    write_json(run_dir/'training_summary.json',dict(**best,wall_seconds=time.perf_counter()-start,peak_gpu_memory_gb=torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else 0.))
    files=[]
    for area,folder in [('results',run_dir),('models',model_dir)]:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.name!='COMPLETE.json' and not p.name.endswith('.tmp'):
                files.append(dict(area=area,path=p.relative_to(folder).as_posix(),sha256=sha(p)))
    write_json(run_dir/'COMPLETE.json',dict(signature=signature,files=files))
    del model,best_state;gc.collect();torch.cuda.empty_cache()


def summarize(root,expected_runs=120):
    root=Path(root);parts=[]
    for marker in sorted((root/'runs').rglob('COMPLETE.json')):parts.append(pd.read_csv(marker.parent/'evaluations.csv'))
    if not parts:raise RuntimeError('Todavía no hay corridas completas')
    evaluations=pd.concat(parts,ignore_index=True);write_csv(root/'evaluations_all.csv',evaluations)
    rows=[]
    for (match,key,seed,target),g in evaluations.groupby(['match_seed','model_key','train_seed','eval_domain']):
        control=g[g.train_domain.eq(target)];transfer=g[~g.train_domain.eq(target)]
        if len(control)!=1 or len(transfer)!=1:continue
        rows.append(dict(match_seed=match,model_key=key,train_seed=seed,target=target,
                         control_f1=float(control.raw_macro_f1.iloc[0]),transfer_f1=float(transfer.raw_macro_f1.iloc[0]),
                         loss=float(control.raw_macro_f1.iloc[0]-transfer.raw_macro_f1.iloc[0])))
    losses=pd.DataFrame(rows)
    if losses.empty:raise RuntimeError('Aún falta el dominio pareado para comparar el mismo objetivo')
    write_csv(root/'fixed_target_losses_by_seed.csv',losses)
    complete=(losses.groupby(['match_seed','model_key','target']).train_seed.nunique()==3)
    cohort=losses.groupby(['match_seed','model_key','target']).agg(n_seeds=('train_seed','nunique'),
        loss_mean=('loss','mean'),loss_sd_seeds=('loss','std'),loss_min_seed=('loss','min'),loss_max_seed=('loss','max'),
        control_f1=('control_f1','mean'),transfer_f1=('transfer_f1','mean')).reset_index()
    write_csv(root/'fixed_target_losses_by_cohort.csv',cohort)
    full=cohort[cohort.n_seeds.eq(3)]
    across=full.groupby(['model_key','target']).agg(n_rematchings=('match_seed','nunique'),loss_mean=('loss_mean','mean'),
        loss_sd_between_rematchings=('loss_mean','std'),loss_min=('loss_mean','min'),loss_max=('loss_mean','max'),
        positive_losses=('loss_mean',lambda s:int((s>0).sum()))).reset_index()
    write_csv(root/'rematching_sensitivity_summary.csv',across)
    n=len(parts);write_json(root/'completion_summary.json',dict(completed_runs=n,expected_runs=expected_runs,
        evaluations=len(evaluations),complete=n==expected_runs,statistical_note='overlapping repeated constructions; no independent-cohort p-value or external-validation claim'))
    print(f'Corridas completas: {n}/{expected_runs}; evaluaciones: {len(evaluations)}/{expected_runs*2}')
    return losses,cohort,across

