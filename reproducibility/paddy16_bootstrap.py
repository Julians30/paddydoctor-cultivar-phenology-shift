"""Paired class-stratified bootstrap for the five completed P5 rematchings.
No neural-network fitting; no bootstrap of rematchings as independent cohorts.
"""
from pathlib import Path
import argparse,json,hashlib,csv,itertools,zipfile,tempfile,platform,time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,balanced_accuracy_score


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def json_write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+'\n')


def scores(cm):
    """Batch confusion matrices (..., K, K) -> macro F1, balanced accuracy, per-class F1/recall."""
    tp=np.diagonal(cm,axis1=-2,axis2=-1);true=cm.sum(-1);pred=cm.sum(-2)
    f1=np.divide(2*tp,true+pred,out=np.zeros_like(tp,dtype=float),where=(true+pred)>0)
    recall=np.divide(tp,true,out=np.zeros_like(tp,dtype=float),where=true>0)
    return f1.mean(-1),recall.mean(-1),f1,recall


def indicators(y,pred,k):
    # pred has [architecture, seed, image]. Result [image, architecture*seed*K*K].
    a,s,n=pred.shape
    cells=y[None,None,:]*k+pred
    return np.eye(k*k,dtype=np.float64)[cells].transpose(2,0,1,3).reshape(n,-1)


def confusion_batch(weights,y,pred,k):
    a,s,n=pred.shape
    return (weights@indicators(y,pred,k)).reshape(len(weights),a,s,k,k)


def paired_seed_average(values,draws):
    # values [B, architecture, seed, ...]; draws [B, 3], paired source-control indices.
    shape=(len(draws),1,draws.shape[1])+(1,)*(values.ndim-3)
    idx=np.broadcast_to(draws.reshape(shape),(len(draws),values.shape[1],draws.shape[1])+values.shape[3:])
    return np.take_along_axis(values,idx,axis=2).mean(axis=2)


def interval(values):return np.quantile(values,[.025,.975],axis=0)


def verify_inputs(base):
    base=Path(base);r=base/'results';design=base/'design'
    plan=json.loads((r/'experiment_plan.json').read_text());status=json.loads((r/'completion_summary.json').read_text())
    if not status['complete'] or status['completed_runs']!=120 or status['evaluations']!=240:raise ValueError('El experimento 15 no está completo')
    if sha(r/'executed_source.py')!=plan['source_sha256']:raise ValueError('Código ejecutado distinto de su huella')
    if sha(design/'03_paddy_doctor_nearclean_frozen.csv')!=plan['metadata_sha256']:raise ValueError('Metadata alterada')
    markers=sorted((r/'runs').rglob('COMPLETE.json'))
    if len(markers)!=120:raise ValueError('Faltan marcadores de finalización')
    checked=0;excluded=0;signatures=set();initializations=[]
    for marker in markers:
        record=json.loads(marker.read_text());signatures.add(record['signature'])
        for entry in record['files']:
            if entry['area']=='models':excluded+=1;continue
            if entry['area']!='results':raise ValueError('Área de manifest desconocida')
            p=marker.parent/entry['path']
            if not p.is_file() or sha(p)!=entry['sha256']:raise ValueError(f'Archivo alterado o ausente: {p}')
            checked+=1
        prov=json.loads((marker.parent/'provenance.json').read_text())
        key=prov['model_key']
        if prov['tag']!=plan['models'][key] or prov['pretrained']!=plan['pretrained_weights'][key]:raise ValueError('Procedencia de pesos incoherente')
        initializations.append({name:prov[name] for name in ['model_key','train_seed','train_domain','match_seed','initialized_tensor_sha256']})
    expected_signature=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    if signatures!={expected_signature}:raise ValueError('Firma del plan inconsistente')
    assignments={}
    for seed in plan['match_seeds']:
        p=design/f'match_{seed}'/'assignments.csv'
        if sha(p)!=plan['assignments'][str(seed)]:raise ValueError('Asignaciones alteradas')
        d=pd.read_csv(p,dtype={'image_id':str});assignments[seed]=d
        if len(d)!=1000 or d.image_id.nunique()!=1000:raise ValueError('Cohorte no contiene 1000 IDs únicos')
        for split,n in [('train',60),('validation',20),('test',20)]:
            sub=d[d.pair_split.eq(split)]
            counts=sub.groupby(['label','age','domain']).size().unstack(fill_value=0)
            if not counts.ADT45.eq(counts.OTHER).all():raise ValueError('Emparejamiento de edad incorrecto')
            if not sub.groupby(['domain','label']).size().eq(n).all():raise ValueError('Tamaño de partición incorrecto')
    verification=dict(completed_runs=120,evaluations=240,result_files_sha256_verified=checked,
        checkpoint_hashes_recorded_but_weights_not_in_results_zip=excluded,executed_source_hash_verified=True,
        metadata_hash_verified=True,assignment_hashes_verified=5,plan_signature_verified=True,
        note='Checkpoint payloads were not supplied in the results ZIP; their bytes were not independently checked here.')
    return plan,assignments,verification,pd.DataFrame(initializations)


def load_predictions(base,plan,assignments,match,target):
    root=Path(base)/'results/runs'/f'match_{match}';models=list(plan['models']);seeds=plan['train_seeds'];classes=plan['class_names']
    expected=assignments[match].query('domain == @target and pair_split == "test"').sort_values('image_id').reset_index(drop=True)
    ids=expected.image_id.tolist();y=expected.label.map({x:i for i,x in enumerate(classes)}).to_numpy()
    if len(ids)!=100 or sorted(np.bincount(y).tolist())!=[20]*5:raise ValueError('Test no es 20 por clase')
    pc=[];pt=[];max_error=0.;checks=0
    for key in models:
        c=[];t=[]
        for seed in seeds:
            for domain,container in [(target,c),('OTHER' if target=='ADT45' else 'ADT45',t)]:
                folder=root/f'train_{domain}__{key}__seed_{seed}'/f'eval__{target}'
                d=pd.read_csv(folder/'predictions.csv',dtype={'image_id':str}).sort_values('image_id').reset_index(drop=True)
                if d.image_id.tolist()!=ids:raise ValueError('Control y transferencia no usan los mismos IDs de test')
                for col in ['label','age','domain','sha256','matched_pair_id']:
                    if not np.array_equal(d[col].astype(str),expected[col].astype(str)):raise ValueError(f'Metadata de test no coincide: {col}')
                pred=d.y_pred_raw.map({x:i for i,x in enumerate(classes)}).to_numpy()
                if pd.isna(pred).any() or not np.array_equal(d.y_true_idx.to_numpy(),y):raise ValueError('Orden de clases incorrecto')
                archived=json.loads((folder/'metrics.json').read_text())['raw']
                err=abs(f1_score(y,pred,labels=range(5),average='macro',zero_division=0)-archived['macro_f1'])
                if err>1e-12:raise ValueError('Macro-F1 no reproduce el archivo')
                max_error=max(max_error,err);checks+=1;container.append(pred)
        pc.append(c);pt.append(t)
    return expected,y,np.asarray(pc),np.asarray(pt),max_error,checks


def analyze(base,output,replicates=5000,random_seed=20261121):
    if replicates<1000:raise ValueError('Usa al menos 1000 réplicas para el análisis entregable')
    start=time.perf_counter();base=Path(base);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plan,assignments,verification,init=verify_inputs(base)
    init.to_csv(output/'initialized_weight_provenance.csv',index=False)
    models=list(plan['models']);seeds=plan['train_seeds'];classes=plan['class_names'];k=len(classes)
    rng=np.random.default_rng(random_seed);macro_rows=[];class_rows=[];seed_rows=[];replicate_payload={};max_error=0.;checks=0
    for match in plan['match_seeds']:
        for target in ['ADT45','OTHER']:
            metadata,y,pc,pt,err,nchecks=load_predictions(base,plan,assignments,match,target)
            max_error=max(max_error,err);checks+=nchecks;n=len(y)
            # Draw exactly 20 image indices per true class, shared by all architectures and seeds.
            weights=np.zeros((replicates,n),dtype=np.float64)
            for label in range(k):
                positions=np.flatnonzero(y==label)
                weights[:,positions]=rng.multinomial(len(positions),np.full(len(positions),1/len(positions)),size=replicates)
            draws=rng.integers(0,len(seeds),size=(replicates,len(seeds)))
            control_cm=confusion_batch(weights,y,pc,k);transfer_cm=confusion_batch(weights,y,pt,k)
            point_c=scores(confusion_batch(np.ones((1,n)),y,pc,k));point_t=scores(confusion_batch(np.ones((1,n)),y,pt,k))
            boot_c=scores(control_cm);boot_t=scores(transfer_cm)
            prefix=f'M{match}_{target}'
            for metric_index,metric in [(0,'macro_f1'),(1,'balanced_accuracy')]:
                # Mean of seed-specific nonlinear scores; never compute F1 after pooling seeds.
                differences=boot_c[metric_index]-boot_t[metric_index]
                fixed=differences.mean(axis=2);hierarchical=paired_seed_average(differences,draws)
                point=(point_c[metric_index]-point_t[metric_index])[0]
                fixed_ci=interval(fixed);hier_ci=interval(hierarchical)
                replicate_payload[f'{prefix}_{metric}_image']=fixed
                replicate_payload[f'{prefix}_{metric}_seed_image']=hierarchical
                for a,key in enumerate(models):
                    macro_rows.append(dict(match_seed=match,target=target,model_key=key,metric=metric,n_target_images=n,
                        n_training_seeds=len(seeds),control_score=float(point_c[metric_index][0,a].mean()),
                        transfer_score=float(point_t[metric_index][0,a].mean()),loss=float(point[a].mean()),
                        image_ci95_lower=float(fixed_ci[0,a]),image_ci95_upper=float(fixed_ci[1,a]),
                        seed_image_interval95_lower=float(hier_ci[0,a]),seed_image_interval95_upper=float(hier_ci[1,a]),
                        image_bootstrap_nonpositive_fraction=float(np.mean(fixed[:,a]<=0)),
                        seed_image_nonpositive_fraction=float(np.mean(hierarchical[:,a]<=0))))
                    for s,seed in enumerate(seeds):
                        low,high=interval(differences[:,a,s])
                        seed_rows.append(dict(match_seed=match,target=target,model_key=key,train_seed=seed,metric=metric,
                            loss=float(point[a,s]),image_ci95_lower=float(low),image_ci95_upper=float(high)))
            for metric_index,metric in [(2,'class_f1'),(3,'class_recall')]:
                dif=boot_c[metric_index]-boot_t[metric_index]
                fixed=dif.mean(2);hierarchical=paired_seed_average(dif,draws)
                fixed_ci=interval(fixed);hier_ci=interval(hierarchical)
                cci=interval(boot_c[metric_index].mean(2));tci=interval(boot_t[metric_index].mean(2))
                for a,key in enumerate(models):
                    for label,name in enumerate(classes):
                        class_rows.append(dict(match_seed=match,target=target,model_key=key,class_name=name,metric=metric,
                            n_unique_class_images=int((y==label).sum()),n_training_seeds=3,
                            control_score=float(point_c[metric_index][0,a,:,label].mean()),
                            transfer_score=float(point_t[metric_index][0,a,:,label].mean()),
                            control_image_ci95_lower=float(cci[0,a,label]),control_image_ci95_upper=float(cci[1,a,label]),
                            transfer_image_ci95_lower=float(tci[0,a,label]),transfer_image_ci95_upper=float(tci[1,a,label]),
                            loss=float((point_c[metric_index]-point_t[metric_index])[0,a,:,label].mean()),
                            image_ci95_lower=float(fixed_ci[0,a,label]),image_ci95_upper=float(fixed_ci[1,a,label]),
                            seed_image_interval95_lower=float(hier_ci[0,a,label]),seed_image_interval95_upper=float(hier_ci[1,a,label])))
            # Cross-check the vectorized confusion/F1 implementation with independent sklearn calls.
            for a in range(len(models)):
                for s in range(3):
                    for b in [0,replicates//2,replicates-1]:
                        for pred,score_batch in [(pc,boot_c),(pt,boot_t)]:
                            independent=f1_score(y,pred[a,s],labels=range(k),average='macro',sample_weight=weights[b],zero_division=0)
                            diff=abs(independent-score_batch[0][b,a,s]);max_error=max(max_error,diff);checks+=1
                            if diff>1e-12:raise ValueError('Bootstrap vectorizado no coincide con sklearn')
            print(f'Bootstrap terminado: matching {match}, objetivo {target}',flush=True)
    macro=pd.DataFrame(macro_rows);perclass=pd.DataFrame(class_rows);perseed=pd.DataFrame(seed_rows)
    macro.to_csv(output/'16_paired_bootstrap_by_matching.csv',index=False)
    perclass.to_csv(output/'16_class_f1_recall_bootstrap.csv',index=False)
    perseed.to_csv(output/'16_per_seed_paired_bootstrap.csv',index=False)
    np.savez_compressed(output/'16_macro_bootstrap_replicates.npz',**replicate_payload)
    # Reconstruct and verify all 120 original seed-level macro-F1 losses.
    original=pd.read_csv(base/'results/fixed_target_losses_by_seed.csv')
    joined=perseed[perseed.metric.eq('macro_f1')].merge(original,on=['match_seed','target','model_key','train_seed'],validate='one_to_one',suffixes=('_new','_original'))
    if len(joined)!=120 or not np.allclose(joined.loss_new,joined.loss_original,atol=1e-12,rtol=0):raise ValueError('No se reproducen las 120 pérdidas originales')
    global_summary=macro.groupby(['metric','model_key','target']).agg(n_rematchings=('match_seed','nunique'),
        mean_loss=('loss','mean'),sd_between_rematchings=('loss','std'),min_loss=('loss','min'),max_loss=('loss','max'),
        n_image_intervals_above_zero=('image_ci95_lower',lambda v:int((v>0).sum())),
        n_seed_image_intervals_above_zero=('seed_image_interval95_lower',lambda v:int((v>0).sum()))).reset_index()
    global_summary.to_csv(output/'16_descriptive_across_matching_summary.csv',index=False)
    overlap=pd.read_csv(base/'design/overlap_between_cohorts.csv')
    overlap.to_csv(output/'16_overlap_between_cohorts.csv',index=False)
    verification.update(dict(numerical_checks=checks,max_absolute_discrepancy=max_error,original_seed_losses_reproduced=120,
        macro_comparisons=40,matched_stratified_draws=True,replicates_per_comparison=replicates,
        neural_network_training_performed=False,elapsed_seconds=time.perf_counter()-start))
    json_write(output/'16_verification.json',verification)
    config=dict(replicates=replicates,random_seed=random_seed,interval_method='percentile 2.5% and 97.5%; not BCa',
        estimand='control minus transfer score on identical target images; average of three seed-specific scores',
        image_bootstrap='100 test images sampled within true class (20 per class), with replacement; same indices for control, transfer, all architectures and seeds within a matching/target',
        seed_image_sensitivity='same class-stratified image draws plus resampling three paired seed indices with replacement; shared indices across architectures',
        rematching_handling='five matching constructions analyzed separately; descriptive mean/SD/range only across constructions; no resampling of overlapping constructions as independent datasets',
        limits=['conditional on fitted networks, selected checkpoints and frozen matching assignments',
                'three seeds yield a limited exploratory seed-distribution sensitivity, not precise seed-population coverage',
                'no simultaneous coverage or multiplicity correction; nonpositive bootstrap fractions are not frequentist p-values',
                'no refitting, farm/session/plant cluster bootstrap or external-domain uncertainty',
                'no BCa correction; class F1/recall intervals have only 20 unique class images and are exploratory',
                'tests had been inspected before this retrospective analysis'],
        models=models,classes=classes,match_seeds=plan['match_seeds'],train_seeds=seeds,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    json_write(output/'16_bootstrap_config.json',config)
    report(output,macro,global_summary,verification,config)
    figures(output,macro,models,plan['match_seeds'],replicates)
    checksum(output)
    return macro,global_summary


def report(out,macro,summary,verification,config):
    f=macro[macro.metric.eq('macro_f1')];s=summary[summary.metric.eq('macro_f1')]
    lines=['# Bootstrap pareado — cinco emparejamientos P5','',
        f'Se ejecutaron {config["replicates"]:,} réplicas por comparación, sin entrenar redes ni utilizar GPU. Se conservaron las 120 corridas y 240 evaluaciones del notebook 15.',
        '', '## Resultado principal','',
        f'Las {len(f)} pérdidas medias de macro-F1 son positivas. Los intervalos por imágenes excluyen cero en {int((f.image_ci95_lower>0).sum())}/{len(f)} comparaciones. Los intervalos de sensibilidad por semillas e imágenes excluyen cero en {int((f.seed_image_interval95_lower>0).sum())}/{len(f)}.',
        '', 'Estos son intervalos marginales por emparejamiento y configuración; el conjunto no tiene una garantía simultánea del 95 %. El recuento no se interpreta como un p-valor global.',
        '', '| Configuración | Objetivo | Pérdida media entre cinco construcciones | Rango de pérdidas | IC por imágenes sobre cero | Sensibilidad semillas e imágenes sobre cero |',
        '| --- | --- | ---: | --- | ---: | ---: |']
    for r in s.itertuples():
        lines.append(f'| {r.model_key} | {r.target} | {r.mean_loss:.4f} | {r.min_loss:.4f}–{r.max_loss:.4f} | {r.n_image_intervals_above_zero}/5 | {r.n_seed_image_intervals_above_zero}/5 |')
    lines+=['','## Qué se remuestreó','',
        'En cada construcción y dominio objetivo se seleccionan 20 imágenes con reemplazo dentro de cada una de las cinco clases. Los mismos índices se aplican a control y transferencia, a las cuatro arquitecturas y a las tres semillas. Se calcula macro-F1 por semilla y después se promedian los tres resultados; no se agrupan 300 predicciones como si fueran 300 imágenes independientes.',
        '', 'El análisis adicional remuestrea tres índices de semilla emparejados con reemplazo y las mismas imágenes. Con solo tres semillas, ese intervalo es una sensibilidad exploratoria. No implica reentrenar redes ni estimar con precisión toda la variación posible de entrenamiento.',
        '', 'Los cinco emparejamientos comparten imágenes. Por eso no se remuestrean como cinco cohortes independientes ni se obtiene un IC global que presuponga su independencia. La media, desviación y rango entre construcciones se presentan descriptivamente. Los intervalos no cuantifican nuevas fincas, otra adquisición, nuevas asignaciones de entrenamiento ni variación de un nuevo ajuste.',
        '', '## Verificación','',
        f'Se verificaron las huellas de {verification["result_files_sha256_verified"]} archivos de resultados, las cinco asignaciones, los metadatos y el código ejecutado. Las 120 pérdidas originales se reprodujeron. Se efectuaron {verification["numerical_checks"]} comprobaciones numéricas con discrepancia máxima {verification["max_absolute_discrepancy"]:.3g}. Los archivos de pesos entrenados no estaban en el ZIP; sus bytes no se verificaron aquí.',
        '', '## Texto propuesto para el manuscrito','',
        '**Methods.** Within each of the five frozen matching constructions and each target domain, we generated 5,000 paired class-stratified bootstrap samples of the 100 target-test images, retaining 20 observations per true class. The same sampled image indices were applied to control and transferred predictions across all architectures and three training seeds. The statistic was the mean of seed-specific control-minus-transfer macro-F1 differences. Marginal percentile 95% intervals used the 2.5th and 97.5th percentiles. A separately labeled exploratory sensitivity additionally resampled three paired seed indices with replacement. Matching constructions were not resampled as independent cohorts because their image sets overlapped; between-construction variation was summarized descriptively. Models, checkpoints and matching assignments remained fixed.',
        '', f'**Results.** Positive fixed-target macro-F1 losses persisted across all 40 architecture–target–matching configurations. Image-conditional intervals excluded zero in {int((f.image_ci95_lower>0).sum())} of 40 comparisons; the seed-and-image sensitivity intervals excluded zero in {int((f.seed_image_interval95_lower>0).sum())} of 40. These marginal retrospective intervals do not establish simultaneous coverage, uncertainty under rematching or external-domain generalization.',
        '', '## Archivos','',
        '- `16_paired_bootstrap_by_matching.csv`: 40 contrastes macro-F1 y 40 de exactitud balanceada, con ambos intervalos.',
        '- `16_per_seed_paired_bootstrap.csv`: intervalos por semilla, manteniendo el par control/transferencia.',
        '- `16_class_f1_recall_bootstrap.csv`: F1 y recall por clase, media de las tres semillas; intervalos exploratorios.',
        '- `16_descriptive_across_matching_summary.csv`: resumen descriptivo entre construcciones, sin IC global de independencia.',
        '- `16_macro_bootstrap_replicates.npz`: réplicas para trazabilidad; ejes réplicas × arquitectura, orden en configuración.',
        '- Figuras PNG/PDF, configuración, verificación e inventario SHA-256.']
    (out/'16_Informe_Bootstrap_P5.md').write_text('\n'.join(lines)+'\n')


def figures(out,macro,models,matches,replicates=5000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    display={'resnet50':('ResNet-50','#216C92'),'convnext_tiny':('ConvNeXt-Tiny','#CC7B26'),
             'vit_small_patch16_224':('ViT-S/16','#27856D'),'swin_tiny_patch4_window7_224':('Swin-Tiny','#AE4962')}
    models=[key for key in display if key in models]
    # Each marker is one matching, not a single pooled effect.
    for mode,lo,hi,label in [('image','image_ci95_lower','image_ci95_upper','Conditional image bootstrap'),
                            ('seed_image','seed_image_interval95_lower','seed_image_interval95_upper','Seed-and-image sensitivity')]:
        fig,axes=plt.subplots(1,2,figsize=(11,7),sharex=True,sharey=True)
        for ax,target in zip(axes,['ADT45','OTHER']):
            for a,key in enumerate(models):
                sub=macro[(macro.metric=='macro_f1')&(macro.target==target)&(macro.model_key==key)].set_index('match_seed').loc[matches]
                y=a*6+np.arange(5)
                ax.errorbar(sub.loss,y,xerr=np.vstack([sub.loss-sub[lo],sub[hi]-sub.loss]),fmt='o',color=display[key][1],capsize=2,ms=4,lw=1.2)
            ax.axvline(0,ls='--',color='.4',lw=1);ax.grid(axis='x',alpha=.2);ax.set_title('Target '+target)
            ax.set_xlabel('Control − transfer macro-F1')
        ticks=[a*6+i for a in range(4) for i in range(5)]
        axes[0].set_yticks(ticks,[f'{display[models[a]][0]} · M{i+1}' for a in range(4) for i in range(5)])
        axes[0].invert_yaxis();fig.suptitle(label+f' · {replicates:,} paired replicates');fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/f'16_{mode}_intervals.{ext}',bbox_inches='tight')
        plt.close(fig)


def checksum(out):
    with (Path(out)/'SHA256.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['relative_path','size_bytes','sha256'])
        for p in sorted(Path(out).rglob('*')):
            if p.is_file() and p.name!='SHA256.csv':w.writerow([p.relative_to(out).as_posix(),p.stat().st_size,sha(p)])


def run_from_zip(input_zip,output,replicates=5000,random_seed=20261121):
    with tempfile.TemporaryDirectory(prefix='paddy16_') as temp:
        with zipfile.ZipFile(input_zip) as archive:
            for item in archive.infolist():
                dest=(Path(temp)/item.filename).resolve()
                if not dest.is_relative_to(Path(temp).resolve()):raise ValueError('Ruta de ZIP no segura')
            archive.extractall(temp)
        result=analyze(Path(temp),output,replicates,random_seed)
    json_write(Path(output)/'16_input_archive.json',dict(file_name=Path(input_zip).name,size_bytes=Path(input_zip).stat().st_size,sha256=sha(input_zip)))
    checksum(output)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input-zip',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--replicates',type=int,default=5000);parser.add_argument('--seed',type=int,default=20261121)
    args=parser.parse_args();run_from_zip(args.input_zip,args.output,args.replicates,args.seed)
