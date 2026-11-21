"""Balanced factorial sum-of-squares decomposition for completed P5 runs.
Descriptive ANOVA only: no F tests, p-values or random-effect variance estimates.
"""
from pathlib import Path
from itertools import combinations,product
from zipfile import ZipFile,ZIP_DEFLATED
import argparse,json,hashlib,tempfile,platform,math,csv
import numpy as np
import pandas as pd

MODELS=['resnet50','convnext_tiny','vit_small_patch16_224','swin_tiny_patch4_window7_224']
LABELS={'resnet50':'ResNet-50','convnext_tiny':'ConvNeXt-Tiny','vit_small_patch16_224':'ViT-S/16','swin_tiny_patch4_window7_224':'Swin-Tiny'}
RESPONSES=['control_f1','transfer_f1','loss']


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def helmert(n):
    q=np.zeros((n,n));q[:,0]=1/np.sqrt(n)
    for j in range(1,n):
        q[:j,j]=1/np.sqrt(j*(j+1));q[j,j]=-j/np.sqrt(j*(j+1))
    return q


def decompose(y,names):
    """Exact orthogonal fixed-table decomposition on a complete balanced Cartesian grid."""
    y=np.asarray(y,dtype=float)
    if y.ndim!=len(names) or not np.isfinite(y).all():raise ValueError('Cuadrícula incompleta o no finita')
    axes=tuple(range(y.ndim));effects={():np.asarray(y.mean()).reshape((1,)*y.ndim)};rows=[]
    centered=y-y.mean();total=float(np.sum(centered**2))
    for order in range(1,y.ndim+1):
        for subset in combinations(axes,order):
            complement=tuple(a for a in axes if a not in subset)
            effect=y.mean(axis=complement,keepdims=True) if complement else y.copy()
            for size in range(order):
                for proper in combinations(subset,size):effect=effect-effects[proper]
            effects[subset]=effect
            expanded=np.broadcast_to(effect,y.shape);ss=float(np.sum(expanded**2));df=math.prod(y.shape[a]-1 for a in subset)
            rows.append(dict(term=' × '.join(names[a] for a in subset),order=order,df=df,ss=ss,
                             mean_square=ss/df,percent_total_ss=100*ss/total if total>0 else np.nan,
                             factors='|'.join(names[a] for a in subset)))
    reconstructed=sum(np.broadcast_to(e,y.shape) for e in effects.values())
    error=float(np.max(np.abs(reconstructed-y)));ss_error=abs(sum(r['ss'] for r in rows)-total)
    # Independent orthonormal Helmert transform checks every component SS.
    z=y.copy()
    for axis,n in enumerate(y.shape):
        z=np.moveaxis(np.tensordot(helmert(n).T,z,axes=(1,axis)),0,axis)
    independent_errors=[]
    for row in rows:
        factors=row['factors'].split('|');slices=tuple(slice(1,None) if name in factors else slice(0,1) for name in names)
        independent=float(np.sum(z[slices]**2));independent_errors.append(abs(independent-row['ss']))
    scale=max(1.,total)
    if ss_error>1e-10*scale or error>1e-10 or max(independent_errors,default=0)>1e-10*scale:raise ValueError('Falla la descomposición independiente de SS')
    if sum(r['df'] for r in rows)!=y.size-1:raise ValueError('Grados de libertad incorrectos')
    audit=dict(n_cells=y.size,total_df=y.size-1,total_ss=total,reconstruction_max_error=error,
               ss_additivity_error=ss_error,helmert_max_ss_error=max(independent_errors,default=0),
               residual_df=0,within_cell_replication=0)
    return pd.DataFrame(rows),audit


def validate_zip(input_zip):
    with ZipFile(input_zip) as z:
        if z.testzip() is not None:raise ValueError('ZIP dañado')
        plan=json.loads(z.read('results/experiment_plan.json'));status=json.loads(z.read('results/completion_summary.json'))
        if not status['complete'] or status['completed_runs']!=120 or status['evaluations']!=240:raise ValueError('Experimento no está completo')
        signature=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();verified=0;markers=[]
        for name in z.namelist():
            if name.endswith('/COMPLETE.json'):
                markers.append(name);receipt=json.loads(z.read(name))
                if receipt['signature']!=signature:raise ValueError('Firma de plan diferente')
                parent=str(Path(name).parent)
                for item in receipt['files']:
                    if item['area']!='results':continue
                    if hashlib.sha256(z.read(parent+'/'+item['path'])).hexdigest()!=item['sha256']:raise ValueError('Resultados alterados')
                    verified+=1
        if len(markers)!=120:raise ValueError('Faltan corridas completas')
        import io
        d=pd.read_csv(io.BytesIO(z.read('results/fixed_target_losses_by_seed.csv')))
        # Verify all 120 pairs directly against their raw prediction tables, on identical image IDs.
        from sklearn.metrics import f1_score
        max_difference=0.
        for row in d.itertuples():
            prefix=f'results/runs/match_{row.match_seed}/'
            other='OTHER' if row.target=='ADT45' else 'ADT45'
            scores=[];frames=[]
            for domain in [row.target,other]:
                name=prefix+f'train_{domain}__{row.model_key}__seed_{row.train_seed}/eval__{row.target}/predictions.csv'
                p=pd.read_csv(io.BytesIO(z.read(name))).sort_values('image_id');frames.append(p)
                scores.append(f1_score(p.label,p.y_pred_raw,labels=plan['class_names'],average='macro',zero_division=0))
            if frames[0].image_id.tolist()!=frames[1].image_id.tolist():raise ValueError('Objetivos distintos')
            dif=max(abs(scores[0]-row.control_f1),abs(scores[1]-row.transfer_f1),abs(scores[0]-scores[1]-row.loss))
            max_difference=max(max_difference,dif)
        if max_difference>1e-12:raise ValueError('No se reproducen las pérdidas')
    return d,plan,dict(result_hashes_verified=verified,fixed_target_losses_reconstructed=120,max_score_difference=max_difference,
                       input_zip_sha256=sha(input_zip),input_zip_size=Path(input_zip).stat().st_size,
                       checkpoint_bytes_verified=False)


def analysis(input_zip,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    d,plan,verification=validate_zip(input_zip)
    matches=plan['match_seeds'];seeds=plan['train_seeds'];frames=[];within=[];audits=[]
    for target in ['ADT45','OTHER']:
        sub=d[d.target.eq(target)]
        expected=pd.MultiIndex.from_product([MODELS,matches,seeds],names=['model_key','match_seed','train_seed'])
        indexed=sub.set_index(['model_key','match_seed','train_seed'])
        if len(indexed)!=60 or indexed.index.duplicated().any() or set(indexed.index)!=set(expected):raise ValueError('Diseño no balanceado 4 × 5 × 3')
        indexed=indexed.loc[expected]
        for response in RESPONSES:
            values=indexed[response].to_numpy().reshape(4,5,3)
            components,audit=decompose(values,['Configuration','Matching','Seed'])
            components.insert(0,'response',response);components.insert(0,'target',target);frames.append(components)
            audits.append(dict(target=target,response=response,scope='four_configurations',**audit))
            for model in MODELS:
                grid=sub[sub.model_key.eq(model)].set_index(['match_seed','train_seed']).loc[pd.MultiIndex.from_product([matches,seeds])][response].to_numpy().reshape(5,3)
                terms,audit=decompose(grid,['Matching','Seed']);terms.insert(0,'model_key',model);terms.insert(0,'response',response);terms.insert(0,'target',target);within.append(terms)
                audits.append(dict(target=target,response=response,model_key=model,scope='within_configuration',**audit))
    global_terms=pd.concat(frames,ignore_index=True);local=pd.concat(within,ignore_index=True)
    global_terms.to_csv(output/'17_factorial_SS_by_target.csv',index=False)
    local.to_csv(output/'17_within_configuration_SS.csv',index=False)
    pd.DataFrame(audits).to_csv(output/'17_algebra_verification.csv',index=False)
    summaries=[]
    for (target,response,model),g in local.groupby(['target','response','model_key']):
        t=g.set_index('term')
        summaries.append(dict(target=target,response=response,model_key=model,
            matching_percent=t.loc['Matching','percent_total_ss'],seed_main_percent=t.loc['Seed','percent_total_ss'],
            matching_seed_interaction_percent=t.loc['Matching × Seed','percent_total_ss'],
            within_matching_seed_spread_percent=t.loc['Seed','percent_total_ss']+t.loc['Matching × Seed','percent_total_ss']))
    summary=pd.DataFrame(summaries);summary.to_csv(output/'17_seed_and_matching_summary.csv',index=False)
    d.to_csv(output/'17_verified_input_scores.csv',index=False)
    verification.update(dict(balanced_three_factor_tables=6,balanced_two_factor_tables=24,
        max_reconstruction_error=max(a['reconstruction_max_error'] for a in audits),
        max_helmert_ss_error=max(a['helmert_max_ss_error'] for a in audits),
        no_F_statistics=True,no_p_values=True,no_random_effect_variance_estimates=True))
    tests()
    verification['synthetic_factor_tests_passed']=True
    (output/'17_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    config=dict(design='separate target domains; 4 model configurations × 5 matching constructions × 3 training seeds',
        responses=RESPONSES,definition='orthogonal sum-of-squares decomposition of a complete balanced table; percentage is SS_term / SS_total × 100',
        interactions='all main effects, all two-way interactions and the three-way interaction retained',
        degrees_of_freedom='59 total; 3,4,2,12,6,8,24 for configuration, matching, seed and their interactions',
        replication='one score per factor combination; saturated decomposition has zero residual df; highest interaction is not an independently estimated random error',
        scope='descriptive contributions to observed finite-table variation, not causal effects or population variance components',
        dependencies='overlapping matching image sets and common target images across configurations; no independence-based F inference',
        seed_scope='seed main effect can cancel across configurations or matchings; seed-related interactions are reported',
        pretraining='configurations have different pretrained lineages, so configuration contribution is not architecture-only',
        bootstrap_relation='bootstrap 16 estimates paired image-conditional uncertainty separately for each matching; ANOVA does not replace it',
        models=MODELS,match_seeds=matches,train_seeds=seeds,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    (output/'17_anova_config.json').write_text(json.dumps(config,indent=2)+'\n')
    figures(output,global_terms,summary)
    report(output,global_terms,summary,verification)
    checksum(output)
    print(json.dumps({'output':str(output),'tables':30,'response':'macro-F1 control, transfer and loss','F_tests':False,'p_values':False}))
    return global_terms,summary


def tests():
    a=np.arange(4,dtype=float)[:,None,None];b=np.arange(5,dtype=float)[None,:,None];s=np.arange(3,dtype=float)[None,None,:]
    cases=[('Configuration',a+np.zeros((4,5,3))),('Matching',b+np.zeros((4,5,3))),('Seed',s+np.zeros((4,5,3))),
           ('Matching × Seed',(b-b.mean())*(s-s.mean())+np.zeros((4,5,3))),
           ('Configuration × Matching × Seed',(a-a.mean())*(b-b.mean())*(s-s.mean()))]
    for term,y in cases:
        table,_=decompose(y,['Configuration','Matching','Seed']);row=table.set_index('term').loc[term]
        if not np.isclose(row.percent_total_ss,100,atol=1e-10):raise ValueError('Caso analítico no recuperado')
    table,audit=decompose(np.ones((4,5,3)),['Configuration','Matching','Seed'])
    if not table.percent_total_ss.isna().all() or audit['total_ss']!=0:raise ValueError('Manejo incorrecto de respuesta constante')


def figures(out,global_terms,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors=['#2C6B92','#CB7F30','#3B8A76','#9A5774','#73819A','#A6A43D','#8C6852']
    terms=global_terms[(global_terms.response=='loss')&(global_terms.target=='ADT45')].term.tolist()
    fig,ax=plt.subplots(figsize=(8.2,4.4));bottom=np.zeros(2)
    short=['Configuration','Matching','Seed','Configuration × Matching','Configuration × Seed','Matching × Seed','Three-way interaction']
    for term,label,color in zip(terms,short,colors):
        vals=[global_terms[(global_terms.response=='loss')&(global_terms.target==target)&(global_terms.term==term)].percent_total_ss.iloc[0] for target in ['ADT45','OTHER']]
        ax.bar(['ADT45','OTHER'],vals,bottom=bottom,label=label,color=color,width=.48)
        for x,v,b in zip(range(2),vals,bottom):
            if v>=8:ax.text(x,b+v/2,f'{v:.1f}%',ha='center',va='center',color='white',fontsize=10)
        bottom+=vals
    ax.set_ylim(0,100);ax.set_ylabel('Share of observed total SS (%)');ax.set_xlabel('Target domain')
    ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,fontsize=9);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'17_factorial_loss_decomposition.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.4),sharex=True)
    cols=['matching_percent','seed_main_percent','matching_seed_interaction_percent'];labels=['Matching','Seed main effect','Matching × seed']
    for ax,target in zip(axes,['ADT45','OTHER']):
        d=summary[(summary.response=='loss')&(summary.target==target)].set_index('model_key').loc[MODELS];left=np.zeros(4)
        for col,label,color in zip(cols,labels,['#CB7F30','#3B8A76','#73819A']):
            ax.barh([LABELS[m] for m in MODELS],d[col],left=left,label=label,color=color)
            for y,v,l in zip(range(4),d[col],left):
                if v>=12:ax.text(l+v/2,y,f'{v:.0f}%',ha='center',va='center',color='white',fontsize=9)
            left+=d[col].to_numpy()
        ax.set_title('Target '+target);ax.set_xlim(0,100);ax.set_xlabel('Share of within-configuration SS (%)');ax.invert_yaxis()
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.02),ncol=3,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1))
    for ext in ['png','pdf']:fig.savefig(out/f'17_matching_seed_decomposition.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def report(out,global_terms,summary,v):
    lines=['# Descomposición ANOVA descriptiva de P5','',
        'Se añadió un diagnóstico de variabilidad a las 120 corridas nuevas. Complementa los intervalos bootstrap; no utiliza pruebas F ni valores p que supongan independencia de los emparejamientos.',
        '', '## Qué se descompone','',
        'Por cada dominio objetivo se organiza una tabla balanceada de 60 puntuaciones: cuatro configuraciones, cinco construcciones emparejadas y tres semillas. Se analizan por separado el macro-F1 de control, el macro-F1 transferido y su pérdida sobre las mismas imágenes objetivo.',
        '', 'La descomposición ortogonal conserva los tres efectos principales, las tres interacciones de dos factores y la interacción triple. Sus sumas de cuadrados suman la variación total observada. Los porcentajes no son efectos causales ni estimaciones de componentes de varianza poblacional.',
        '', '## Pérdida de macro-F1','', '| Componente | gl | Objetivo ADT45 % SS | Objetivo OTHER % SS |','| --- | ---: | ---: | ---: |']
    g=global_terms[global_terms.response.eq('loss')]
    for term in g.term.drop_duplicates():
        d=g[g.term.eq(term)].set_index('target');lines.append(f'| {term} | {int(d.df.iloc[0])} | {d.loc["ADT45","percent_total_ss"]:.2f} | {d.loc["OTHER","percent_total_ss"]:.2f} |')
    lines+=['', '## Semillas dentro de cada configuración','',
        'El efecto principal de semilla mide un patrón promedio compartido entre construcciones. Puede ser pequeño aunque algunas semillas difieran mucho dentro de cada construcción. Por eso se informa también la interacción emparejamiento × semilla. La suma de ambos términos equivale a la variación entre las tres semillas dentro de los cinco emparejamientos.',
        '', '| Configuración | Objetivo | Emparejamiento % | Semilla principal % | Interacción % | Semilla e interacción % |','| --- | --- | ---: | ---: | ---: | ---: |']
    for r in summary[summary.response.eq('loss')].itertuples():
        lines.append(f'| {LABELS[r.model_key]} | {r.target} | {r.matching_percent:.2f} | {r.seed_main_percent:.2f} | {r.matching_seed_interaction_percent:.2f} | {r.within_matching_seed_spread_percent:.2f} |')
    lines+=['','## Límites de interpretación','',
        'Hay una puntuación por combinación de factores. Al conservar todas las interacciones, el modelo es saturado: 59 grados de libertad de variación y cero grados de libertad residuales. La interacción triple no es una estimación independiente del error aleatorio.',
        '', 'Las construcciones comparten imágenes y las configuraciones se evalúan sobre los mismos objetivos. No se calcula un F convencional, ni valores p, ni se declara que el porcentaje de semilla estime la varianza de cualquier futuro entrenamiento. Las configuraciones tienen procedencias de preentrenamiento diferentes; el efecto de configuración no aísla solamente arquitectura.',
        '', 'Las tres semillas no son muestras de campo nuevas. La variación entre cinco construcciones se conserva como sensibilidad interna; no representa cinco poblaciones independientes. El resultado principal de transferencia sigue sustentándose en las diferencias pareadas y el bootstrap 16.',
        '', '## Verificación','',
        f'Se comprobaron las huellas de {v["result_hashes_verified"]} archivos de resultados y se reconstruyeron las 120 pérdidas desde las predicciones, con discrepancia máxima {v["max_score_difference"]:.3g}. Las 30 descomposiciones se verificaron con una transformación ortonormal de Helmert independiente, pruebas sintéticas de efectos puros e interacciones y sumas de grados de libertad. Discrepancia máxima en SS: {v["max_helmert_ss_error"]:.3g}.',
        '', '## Texto para el manuscrito','',
        '**Methods.** We added a descriptive factorial ANOVA decomposition of the observed macro-F1 scores and fixed-target losses, separately for each target domain. The balanced 4-configuration × 5-matching × 3-seed table was decomposed orthogonally into all main effects, all two-way interactions and the three-way interaction. Component contributions were reported as percentages of total corrected sum of squares. A complementary 5-matching × 3-seed decomposition within each configuration separated matching, seed and matching-by-seed terms. With one score per factor combination, the full decomposition had zero residual degrees of freedom. We therefore reported no F statistics, p-values or population variance components. This diagnostic did not replace the paired bootstrap and did not treat overlapping matching constructions as independent cohorts.',
        '', 'El paquete contiene los CSV de todos los términos para control, transferencia y pérdida, las figuras PNG/PDF, las verificaciones y el código reproducible.']
    (out/'17_Informe_ANOVA_Descriptiva_P5.md').write_text('\n'.join(lines)+'\n')


def checksum(out):
    with (Path(out)/'SHA256.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['relative_path','size_bytes','sha256'])
        for p in sorted(Path(out).rglob('*')):
            if p.is_file() and p.name!='SHA256.csv':w.writerow([p.relative_to(out).as_posix(),p.stat().st_size,sha(p)])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input-zip',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();analysis(args.input_zip,args.output)
