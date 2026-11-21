from pathlib import Path
import sys,json,pickle,hashlib,argparse
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.metrics import f1_score
sys.path.insert(0,str(Path(__file__).parent))
import paddy18_complete_frozen_analyses as a
P=Path(__file__).parent;R=P/'analysis18_results' if (P/'analysis18_results').exists() else P/'resultados';rng=np.random.default_rng(9001)
args=argparse.ArgumentParser();args.add_argument('--project',type=Path);args=args.parse_args()
checks={}
# Proper scores improve for a recoverably overconfident simulated source population.
y=rng.integers(0,3,3000);z=rng.normal(size=(3000,3));prob=np.exp(z-z.max(1)[:,None]);prob/=prob.sum(1,keepdims=True)
y=np.array([rng.choice(3,p=p) for p in prob]);tr=np.arange(2000);te=np.arange(2000,3000)
f=a.calibrate(z[tr]*3,y[tr],'temperature');checks['temperature_recovers_known_miscalibration']=abs(f['temperature']-3)<.5
for kind in ['vector','dirichlet_odir']:
 lp=z*3;lp-=np.log(np.exp(lp-lp.max(1)[:,None]).sum(1))[:,None]+lp.max(1)[:,None]
 fit=a.calibrate(lp[tr],y[tr],kind,.1);p=a.apply(lp[te],fit)
 checks[kind+'_proper_probabilities']=bool(fit['success'] and np.allclose(p.sum(1),1) and (p>0).all())
# Exact analytic normal absolute expectation vs independent Monte Carlo.
mu=.04;se=.03;analytic=se*np.sqrt(2/np.pi)*np.exp(-.5*(mu/se)**2)+mu*(2*ndtr(mu/se)-1)
checks['analytic_normal_bias_adjustment']=abs(np.abs(rng.normal(mu,se,1000000)).mean()-analytic)<.00015
gl=pd.read_csv(R/'18_crossed_GLMM_diagnostics.csv');checks['all_16_mixed_fits_converged']=bool(len(gl)==16 and gl.success.all())
fits=json.load(open(R/'18_source_calibration_fits.json'));checks['all_96_source_calibrators_converged']=len(fits)==96 and all(x['success'] for x in fits)
tf=json.load(open(R/'18_target_calibration_assignments.json'));checks['target_calibration_evaluation_disjoint']=all(len(set(f['calibration_image_ids'])&set(f['evaluation_image_ids']))==0 and len(f['calibration_image_ids'])==20 and len(f['evaluation_image_ids'])==80 for f in tf)
bc=pd.read_csv(R/'18_class_F1_BCa.csv');checks['120_BCa_intervals_finite_ordered']=bool(len(bc)==120 and np.isfinite(bc[['bca95_lower','bca95_upper']]).all().all() and (bc.bca95_lower<=bc.bca95_upper).all())
crc=pd.read_csv(R/'18_crc_prediction_sets.csv');checks['CRC_source_bound_met']=bool((crc.corrected_source_risk<=crc.alpha+1e-12).all())
wide=crc.pivot(index=['experiment','source','target','model','seed'],columns='alpha',values='mean_set_size');checks['CRC_nested_mean_set_sizes']=bool((wide[.05]>=wide[.10]-1e-12).all())
# Verify class-F1 points against sklearn, independently of the custom confusion-count routine.
if (P/'entries_cache.pkl').exists():entries,_,_=pickle.load(open(P/'entries_cache.pkl','rb'))
else:
 project=args.project or P
 if not (project/'paddy14_analysis.py').exists():raise FileNotFoundError('Pasa --project a la carpeta extraída de Online Resource 2 que contiene paddy14_analysis.py')
 sys.path.insert(0,str(project));import paddy14_analysis as base
 _,entries,_,_,_=base.inspect_sources(project,True)
err=0.
for _,row in bc[bc.condition.ne('loss')].iterrows():
 rr=[r for r in entries if r['experiment']=='P5' and r['target']==row.target and r['model']==row.model and ((r['source']==row.target)==(row.condition=='control'))]
 expected=np.mean([f1_score(r['y'],r['p'].argmax(1),labels=range(5),average=None,zero_division=0)[r['classes'].index(row['class'])] for r in rr]);err=max(err,abs(expected-row.mean_seed_f1))
checks['class_F1_points_match_sklearn']=err<1e-12
checks={k:bool(v) for k,v in checks.items()}
assert all(checks.values()),checks
write={'checks':checks,'max_class_F1_point_error':err,'source_script_sha256':hashlib.sha256((P/'paddy18_complete_frozen_analyses.py').read_bytes()).hexdigest()}
(P/'validation18.json').write_text(json.dumps(write,indent=2)+'\n')
print(json.dumps(write,indent=2))
cal=pd.read_csv(R/'18_source_calibration_metrics.csv');cal=cal[cal.condition.eq('transfer')]
cal.groupby(['experiment','target','method'])[['nll','brier','macro_f1']].mean().reset_index().to_csv(R/'18_calibration_summary.csv',index=False)
target=pd.read_csv(R/'18_target20_label_reference.csv');target.groupby(['target','method'])[['nll','brier']].mean().reset_index().to_csv(R/'18_target20_summary.csv',index=False)
crc['condition']=np.where(crc.source.eq(crc.target),'control','transfer');crc.groupby(['experiment','target','condition','alpha'])[['test_coverage','mean_set_size','singleton_fraction']].mean().reset_index().to_csv(R/'18_crc_summary.csv',index=False)
