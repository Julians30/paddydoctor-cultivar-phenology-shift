# Python export of 14_PaddyDoctor_Refuerzo_Confiabilidad_Sin_Reentrenar.ipynb
# Colab setup cells require IPython; portable entrypoints are in reproducibility/.

# Cell 2
from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import html
import io
import json
import platform
import re
import warnings
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score, log_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ['resnet50', 'convnext_tiny', 'vit_small_patch16_224', 'swin_tiny_patch4_window7_224']
LABELS = dict(zip(MODELS, ['ResNet-50', 'ConvNeXt-Tiny', 'ViT-S/16', 'Swin-Tiny']))
COLORS = dict(zip(MODELS, ['#23689b', '#df8126', '#32835d', '#b84453']))
SEEDS = [2026, 2027, 2028]
PROTOCOL_DIRS = {'P1': '05_P1_30ep_noES', 'P4': '06_P4_cross_cultivar_30ep_noES',
                 'P5': '07_P5_exact_age_matched_30ep_noES', 'cal': '09_calibration_reliability'}
GROUP = ['experiment', 'target', 'source', 'model']
EPS = 1e-12
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 300, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


class Source:
    """Lee archivos ZIP portables o resultados existentes en Drive, sin extraerlos."""
    def __init__(self, root, key, portable):
        self.key, self.portable = key, portable
        self.path = Path(root) / ('inputs/' + key.lower() + '.zip') if portable else Path(root) / 'results' / PROTOCOL_DIRS[key]
        if not self.path.exists():
            raise FileNotFoundError(f'No se encontró la entrada {self.path}')
        self.z = ZipFile(self.path) if portable else None
        self.names = [n for n in self.z.namelist() if not n.endswith('/')] if portable else [p.relative_to(self.path).as_posix() for p in self.path.rglob('*') if p.is_file()]

    def read(self, name):
        return self.z.read(name) if self.portable else (self.path / name).read_bytes()

    def csv(self, name):
        return pd.read_csv(io.BytesIO(self.read(name)))

    def suffix(self, suffix):
        matches = [n for n in self.names if n.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f'{self.key}: se esperaba un archivo {suffix}, encontrados {len(matches)}')
        return matches[0]


def manual_f1(y, pred, k):
    """Macro-F1 desde conteos; independiente de f1_score de sklearn."""
    cm = np.bincount(y * k + pred, minlength=k*k).reshape(k, k)
    denominator = cm.sum(0) + cm.sum(1)
    per_class = np.divide(2*np.diag(cm), denominator, out=np.zeros(k), where=denominator > 0)
    return float(per_class.mean())


def calibration_bins(correct, confidence, bins, adaptive=False):
    """ECE de máxima confianza; los empates no se separan en bins distintos.

    Adaptive ECE usa bordes cuantiles. Los bordes repetidos se fusionan,
    de modo que el número efectivo de intervalos puede ser menor al solicitado.
    No es el ACE que promedia todas las probabilidades de cada clase.
    """
    confidence, correct = np.asarray(confidence), np.asarray(correct, float)
    if adaptive:
        edges = np.unique(np.r_[0., np.quantile(confidence, np.linspace(0, 1, bins+1))[1:-1], 1.])
    else:
        edges = np.linspace(0, 1, bins+1)
    group = np.clip(np.searchsorted(edges, confidence, side='left')-1, 0, len(edges)-2)
    counts = np.bincount(group, minlength=len(edges)-1)
    sums = np.bincount(group, weights=correct, minlength=len(counts))
    confs = np.bincount(group, weights=confidence, minlength=len(counts))
    acc = np.divide(sums, counts, out=np.full(len(counts), np.nan), where=counts > 0)
    avg = np.divide(confs, counts, out=np.full(len(counts), np.nan), where=counts > 0)
    gap = np.abs(acc-avg)
    ece = float(np.nansum(counts/len(correct)*gap))
    mce = float(np.nanmax(gap))
    table = pd.DataFrame({'lower': edges[:-1], 'upper': edges[1:], 'n': counts,
                          'accuracy': acc, 'confidence': avg, 'absolute_gap': gap})
    return ece, mce, table


def metrics(y, p):
    pred = p.argmax(1)
    correct, confidence = pred == y, p.max(1)
    onehot = np.eye(p.shape[1])[y]
    result = {'macro_f1': manual_f1(y, pred, p.shape[1]), 'accuracy': float(correct.mean()),
              'brier': float(np.square(p-onehot).sum(1).mean()),
              'nll': float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1)).mean()),
              'overconfidence': float(confidence.mean()-correct.mean())}
    for b in [5, 10, 15]:
        result[f'ece_uniform_{b}'], result[f'mce_uniform_{b}'], _ = calibration_bins(correct, confidence, b)
        result[f'ece_adaptive_{b}'], _, _ = calibration_bins(correct, confidence, b, True)
    return result


def validation_rule(y, p, risk_limit=.05, minimum_coverage=.20):
    """Mayor cobertura con error EMPÍRICO <=5%, mínimo 20 casos y 20%.

    Se elige exclusivamente en validación fuente. No ofrece garantía de riesgo.
    Si ninguna regla es elegible se rechaza todo (umbral >1).
    """
    confidence, wrong = p.max(1), p.argmax(1) != y
    candidates = np.unique(np.r_[0., confidence])
    minimum = max(20, int(np.ceil(len(y)*minimum_coverage)))
    for threshold in candidates:
        selected = confidence >= threshold
        if selected.sum() >= minimum and wrong[selected].mean() <= risk_limit:
            return float(threshold), int(selected.sum()), float(wrong[selected].mean()), True
    return float(np.nextafter(1., 2.)), 0, np.nan, False


def inspect_sources(root, portable):
    sources = {key: Source(root, key, portable) for key in ['P1', 'P4', 'P5', 'cal']}
    temperatures = sources['cal'].csv(sources['cal'].suffix('09_temperature_parameters_validation.csv'))
    assert len(temperatures) == 60, 'Se requieren 60 temperaturas de validación'
    old_metrics = sources['cal'].csv(sources['cal'].suffix('09_test_calibration_by_run.csv'))
    old_metrics = old_metrics.rename(columns={'model_key': 'model', 'train_domain': 'source', 'eval_domain': 'target'})
    cache = Path(root) / ('inputs/validation_cache' if portable else 'results/09_calibration_reliability/validation_cache')
    entries, histories, verifications, rules = [], [], [], []
    for experiment in ['P1', 'P4', 'P5']:
        src = sources[experiment]
        for name in sorted(src.names):
            if name.endswith('/history.csv'):
                run = Path(name).parent.name
                match = re.fullmatch(r'(?:train_(ADT45|OTHER)__)?(.+)__seed_(\d+)', run)
                if not match:
                    continue
                source, model, seed = match.group(1) or 'MIXED', match.group(2), int(match.group(3))
                h = src.csv(name)
                h['experiment'], h['source'], h['model'], h['seed'] = experiment, source, model, seed
                assert len(h) == 30 and h.epoch.tolist() == list(range(1, 31)), name
                histories.append(h)
            if not name.endswith('/test_predictions.csv') or 'paired_test_predictions' in name:
                continue
            parts = Path(name).parts
            run = parts[-2] if experiment == 'P1' else parts[-3]
            match = re.fullmatch(r'(?:train_(ADT45|OTHER)__)?(.+)__seed_(\d+)', run)
            assert match, name
            source, model, seed = match.group(1) or 'MIXED', match.group(2), int(match.group(3))
            target = 'MIXED' if experiment == 'P1' else parts[-2].removeprefix('eval__')
            temp = temperatures[(temperatures.experiment == experiment) & (temperatures.model_key == model) &
                                (temperatures.seed == seed) & (temperatures.train_domain == source)]
            assert len(temp) == 1, (experiment, source, model, seed)
            temp = temp.iloc[0]
            classes = json.loads(temp.class_names_json)
            d = src.csv(name).sort_values('image_id').reset_index(drop=True)
            assert d.image_id.is_unique
            p = d[['prob__'+c for c in classes]].to_numpy(float)
            assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1+1e-6).all()
            assert np.allclose(p.sum(1), 1, atol=1e-5)
            p = p/p.sum(1, keepdims=True)
            y = d.y_true_idx.to_numpy(int)
            assert np.array_equal(np.array(classes)[y], d.y_true.to_numpy())
            assert np.array_equal(np.array(classes)[p.argmax(1)], d.y_pred.to_numpy())
            q = softmax(np.log(np.clip(p, EPS, 1))/float(temp.temperature), axis=1)
            assert np.array_equal(p.argmax(1), q.argmax(1)), 'T debe conservar la clase predicha'
            key = dict(experiment=experiment, source=source, target=target, model=model, seed=seed)
            stored = old_metrics
            for column, value in key.items():
                stored = stored[stored[column] == value]
            assert len(stored) == 1
            for method, prob in [('raw', p), ('ts', q)]:
                m = metrics(y, prob)
                numeric = {'macro_f1_sklearn': abs(m['macro_f1']-f1_score(y, prob.argmax(1), labels=range(len(classes)), average='macro', zero_division=0)),
                           'nll_sklearn': abs(m['nll']-log_loss(y, prob, labels=range(len(classes))))}
                for field, original in [('accuracy', 'accuracy'), ('brier', 'brier'), ('nll', 'nll'),
                                        ('ece_uniform_15', 'ece_15'), ('mce_uniform_15', 'mce_15')]:
                    numeric['archive_'+field] = abs(m[field]-float(stored.iloc[0][method+'_'+original]))
                for field, difference in numeric.items():
                    verifications.append({**key, 'method': method, 'check': field, 'absolute_difference': difference})
                assert max(numeric.values()) < 1e-8, (key, method, numeric)
            validation_file = cache/(str(temp.run_id)+'.npz')
            if not validation_file.exists():
                raise FileNotFoundError(f'Falta caché de validación: {validation_file}; ejecutar NB09 para generarlo no requiere reentrenar.')
            arrays = np.load(validation_file, allow_pickle=False)
            vy, logits, vids = arrays['y_true'].astype(int), arrays['logits'].astype(float), arrays['image_ids'].astype(str)
            assert logits.shape == (int(temp.n_validation), len(classes))
            assert len(set(vids)) == len(vids) and np.isfinite(logits).all()
            assert not set(vids) & set(d.image_id), 'Solapamiento validation/test'
            validation_prob = softmax(logits, axis=1)
            validation_nll = -np.log(np.clip(validation_prob[np.arange(len(vy)), vy], EPS, 1)).mean()
            assert abs(validation_nll-float(temp.validation_raw_nll)) < 1e-6, 'Caché de validación no coincide con NB09'
            thresholds = {}
            for method, prob in [('raw', softmax(logits, axis=1)), ('ts', softmax(logits/float(temp.temperature), axis=1))]:
                threshold, n, risk, eligible = validation_rule(vy, prob)
                thresholds[method] = threshold
                rules.append({**key, 'method': method, 'threshold': threshold, 'validation_n': len(vy),
                              'validation_accepted': n, 'validation_risk': risk, 'eligible': eligible,
                              'risk_limit': .05, 'minimum_coverage': .20, 'run_id': temp.run_id})
            entries.append({**key, 'df': d, 'classes': classes, 'p': p, 'q': q, 'y': y,
                            'thresholds': thresholds, 'temperature': float(temp.temperature)})
    assert len(entries) == 108 and len(histories) == 60, (len(entries), len(histories))
    unique_rules = pd.DataFrame(rules).drop(columns='target').drop_duplicates()
    assert len(unique_rules) == 120
    return sources, entries, pd.concat(histories, ignore_index=True), pd.DataFrame(verifications), unique_rules



# Cell 3
# Detectar Colab o el paquete portable local.
try:
    from google.colab import drive
except ImportError:
    drive = None

if drive is not None:
    drive.mount('/content/drive')
    PROJECT_ROOT = Path('/content/drive/MyDrive/PADDY_DOCTOR_Q1')
    PORTABLE = False
    OUTPUT_ROOT = PROJECT_ROOT / 'results' / '14_reliability_reinforcement'
else:
    candidates = [Path.cwd(), Path.cwd()/'PaddyDoctor_Refuerzo_14', Path.cwd()/'paddy14']
    PROJECT_ROOT = next((p for p in candidates if (p/'inputs/p1.zip').exists()), None)
    if PROJECT_ROOT is None:
        raise FileNotFoundError('Abra el notebook desde el paquete descomprimido o ejecútelo en Colab con Drive.')
    PORTABLE = True
    OUTPUT_ROOT = PROJECT_ROOT / 'results_reproducidos'

N_BOOTSTRAP = 2000
print('Entradas:', PROJECT_ROOT)
print('Salida:', OUTPUT_ROOT)
print('Modo portable:', PORTABLE)

# Cell 5
def build_tables(entries, out):
    rows, classes, bins, operating, curves = [], [], [], [], []
    for e in entries:
        key = {k: e[k] for k in GROUP+['seed']}
        for method, p in [('raw', e['p']), ('ts', e['q'])]:
            y, pred, confidence = e['y'], p.argmax(1), p.max(1)
            correct = pred == y
            rows.append({**key, 'method': method, 'n': len(y), **metrics(y, p)})
            cm = confusion_matrix(y, pred, labels=range(p.shape[1]))
            for k, label in enumerate(e['classes']):
                tp, support, predicted = cm[k, k], cm[k].sum(), cm[:, k].sum()
                classes.append({**key, 'method': method, 'class': label, 'support': support,
                                'precision': tp/predicted if predicted else 0., 'recall': tp/support if support else 0.,
                                'f1': 2*tp/(support+predicted) if support+predicted else 0.})
            for b in [5, 10, 15]:
                for adaptive in [False, True]:
                    _, _, table = calibration_bins(correct, confidence, b, adaptive)
                    for row in table.to_dict('records'):
                        bins.append({**key, 'method': method, 'requested_bins': b,
                                     'binning': 'adaptive_quantiles' if adaptive else 'uniform', **row})
            for policy, threshold in [('fixed_090', .90), ('fixed_095', .95), ('source_validation_5pct', e['thresholds'][method])]:
                sel = confidence >= threshold
                operating.append({**key, 'method': method, 'policy': policy, 'threshold': threshold,
                                  'n': len(y), 'accepted': int(sel.sum()), 'wrong_accepted': int((sel & ~correct).sum()),
                                  'coverage': sel.mean(), 'risk': float((~correct)[sel].mean()) if sel.any() else np.nan,
                                  'wrong_fraction_all_images': (sel & ~correct).mean()})
            order = np.argsort(-confidence, kind='stable')
            for coverage in np.linspace(.10, 1., 91):
                n = int(np.ceil(coverage*len(y)))
                curves.append({**key, 'method': method, 'coverage': n/len(y), 'requested_coverage': coverage,
                               'risk': float((~correct)[order[:n]].mean()), 'retained': n})
    products = {'metrics_by_run': pd.DataFrame(rows), 'class_metrics': pd.DataFrame(classes),
                'calibration_bins': pd.DataFrame(bins), 'operating_by_run': pd.DataFrame(operating),
                'risk_coverage_by_run': pd.DataFrame(curves)}
    for name, d in products.items():
        d.to_csv(out/(name+'.csv'), index=False)
    metrics_df = products['metrics_by_run']
    fields = [c for c in metrics_df if c not in GROUP+['seed', 'method', 'n']]
    summary = metrics_df.groupby(GROUP+['method'])[fields].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(out/'metrics_summary.csv', index=False)
    return products



# Cell 7
def boot_indices(y, n_boot, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(y == c), size=(n_boot, int((y == c).sum())), replace=True)
                           for c in np.unique(y)], axis=1)


def bootstrap_f1(y, predictions, indices, k):
    """Media de F1 de las semillas dentro de cada réplica; semillas fijas."""
    batch, n = indices.shape
    result = np.zeros(batch)
    for pred in predictions:
        codes = y[indices]*k+pred[indices]+np.arange(batch)[:, None]*k*k
        cm = np.bincount(codes.ravel(), minlength=batch*k*k).reshape(batch, k, k)
        denom = cm.sum(1)+cm.sum(2)
        diagonal = np.diagonal(cm, axis1=1, axis2=2)
        score = np.divide(2*diagonal, denom, out=np.zeros_like(diagonal, dtype=float), where=denom > 0).mean(1)
        result += score/len(predictions)
    return result


def bootstrap_policy(selected, wrong, indices):
    """Estima cobertura y media de riesgo por semilla. Riesgo indefinido si n=0."""
    selected = np.array(selected, dtype=bool)
    wrong = np.array(wrong, dtype=bool)
    counts = selected[:, indices].sum(2)
    errors = (selected & wrong)[:, indices].sum(2)
    risk = np.divide(errors, counts, out=np.full_like(errors, np.nan, dtype=float), where=counts > 0)
    all_valid = np.all(counts > 0, axis=0)
    mean_risk = np.full(indices.shape[0], np.nan)
    mean_risk[all_valid] = risk[:, all_valid].mean(0)
    return (counts/indices.shape[1]).mean(0), mean_risk


def holm(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    result = np.empty(len(order))
    result[order] = np.minimum(1., np.maximum.accumulate((len(order)-np.arange(len(order)))*pvalues[order]))
    return result


def paired_analyses(entries, out, n_boot=2000):
    effects, policy_rows, tests, persistence, persistence_summary = [], [], [], [], []
    lookup = {(e['experiment'], e['target'], e['source'], e['model'], e['seed']): e for e in entries}
    for ex in ['P4', 'P5']:
        for target in ['ADT45', 'OTHER']:
            opposite = 'OTHER' if target == 'ADT45' else 'ADT45'
            reference = lookup[(ex, target, target, MODELS[0], SEEDS[0])]
            y = reference['y']
            ids = reference['df'].image_id.to_numpy()
            idx = boot_indices(y, n_boot, 14000+(ex == 'P5')*100+(target == 'OTHER'))
            for model in MODELS:
                control = [lookup[(ex, target, target, model, s)] for s in SEEDS]
                cross = [lookup[(ex, target, opposite, model, s)] for s in SEEDS]
                for e in control+cross:
                    assert np.array_equal(e['df'].image_id, ids) and np.array_equal(e['y'], y)
                key = dict(experiment=ex, target=target, model=model, n=len(y), n_boot=n_boot)
                cp = np.array([e['p'].argmax(1) for e in control])
                xp = np.array([e['p'].argmax(1) for e in cross])
                delta = bootstrap_f1(y, cp, idx, len(reference['classes']))-bootstrap_f1(y, xp, idx, len(reference['classes']))
                point = np.mean([manual_f1(y, p, len(reference['classes'])) for p in cp])-np.mean([manual_f1(y, p, len(reference['classes'])) for p in xp])
                effects.append({**key, 'macro_f1_drop': point, 'ci_low': np.quantile(delta, .025), 'ci_high': np.quantile(delta, .975)})
                for i, seed in enumerate(SEEDS):
                    c, x = cp[i] == y, xp[i] == y
                    b, d = int((c & ~x).sum()), int((~c & x).sum())
                    tests.append({**key, 'seed': seed, 'control_correct_cross_wrong': b, 'reverse': d,
                                  'p_exact': binomtest(b, b+d, .5).pvalue if b+d else 1.})
                for method in ['raw', 'ts']:
                    probs = [e['p'] if method == 'raw' else e['q'] for e in cross]
                    conf = np.array([p.max(1) for p in probs])
                    for policy in ['fixed_090', 'fixed_095', 'source_validation_5pct']:
                        thresholds = np.array([.9 if policy == 'fixed_090' else .95 if policy == 'fixed_095' else e['thresholds'][method] for e in cross])
                        sel = conf >= thresholds[:, None]
                        wrong = xp != y
                        cov_boot, risk_boot = bootstrap_policy(sel, wrong, idx)
                        counts = sel.sum(1)
                        risk = np.divide((sel & wrong).sum(1), counts, out=np.full(3, np.nan), where=counts > 0)
                        valid = np.isfinite(risk_boot)
                        enough = valid.mean() >= .95 and np.all(counts > 0)
                        policy_rows.append({**key, 'method': method, 'policy': policy,
                                            'coverage': sel.mean(), 'coverage_ci_low': np.quantile(cov_boot, .025),
                                            'coverage_ci_high': np.quantile(cov_boot, .975),
                                            'risk': risk.mean() if np.all(counts > 0) else np.nan,
                                            'risk_ci_low': np.quantile(risk_boot[valid], .025) if enough else np.nan,
                                            'risk_ci_high': np.quantile(risk_boot[valid], .975) if enough else np.nan,
                                            'valid_risk_bootstrap_fraction': valid.mean(), 'zero_acceptance_seeds': int((counts == 0).sum())})
                cw, xw = cp != y, xp != y
                conf_raw = np.array([e['p'].max(1) for e in cross])
                conf_ts = np.array([e['q'].max(1) for e in cross])
                for j, image_id in enumerate(ids):
                    record = {**key, 'image_id': image_id, 'true_class': reference['classes'][y[j]],
                              'control_errors_3seeds': int(cw[:, j].sum()), 'cross_errors_3seeds': int(xw[:, j].sum()),
                              'correct_control_all_wrong_cross_all': bool((~cw[:, j]).all() and xw[:, j].all()),
                              'wrong_highconf090_raw_all': bool((xw[:, j] & (conf_raw[:, j] >= .9)).all()),
                              'wrong_highconf090_ts_all': bool((xw[:, j] & (conf_ts[:, j] >= .9)).all()),
                              'raw_confidence_mean': conf_raw[:, j].mean(), 'ts_confidence_mean': conf_ts[:, j].mean()}
                    for i, seed in enumerate(SEEDS):
                        record[f'cross_prediction_{seed}'] = reference['classes'][xp[i, j]]
                    for column in ['age', 'variety', 'matched_pair_id']:
                        if column in reference['df']:
                            record[column] = reference['df'].iloc[j][column]
                    persistence.append(record)
                persistence_summary.append({**key, 'cross_wrong_all_seeds': int(xw.all(0).sum()),
                                            'control_correct_cross_wrong_all': int(((~cw).all(0) & xw.all(0)).sum()),
                                            'wrong_highconf090_raw_all': int((xw & (conf_raw >= .9)).all(0).sum()),
                                            'wrong_highconf090_ts_all': int((xw & (conf_ts >= .9)).all(0).sum())})
    tests = pd.DataFrame(tests)
    tests['p_holm'] = np.nan
    # Familia explícita: 4 arquitecturas x 3 semillas dentro de protocolo y objetivo.
    for _, positions in tests.groupby(['experiment', 'target']).groups.items():
        tests.loc[positions, 'p_holm'] = holm(tests.loc[positions, 'p_exact'].to_numpy())
    products = {'fixed_target_f1_bootstrap': pd.DataFrame(effects),
                'cross_operating_bootstrap': pd.DataFrame(policy_rows), 'mcnemar_holm': tests,
                'persistent_errors_per_image': pd.DataFrame(persistence),
                'persistent_errors_summary': pd.DataFrame(persistence_summary)}
    for name, table in products.items():
        table.to_csv(out/(name+'.csv'), index=False)
    return products



# Cell 9
def save_figure(fig, root, name):
    fig.savefig(root/(name+'.png'), bbox_inches='tight', dpi=220)
    fig.savefig(root/(name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def make_figures(histories, tables, paired, figroot):
    for ex in ['P1', 'P4', 'P5']:
        sources = ['MIXED'] if ex == 'P1' else ['ADT45', 'OTHER']
        fig, axes = plt.subplots(2*len(sources), 4, figsize=(14, 5.5*len(sources)), squeeze=False)
        for r, source in enumerate(sources):
            for j, model in enumerate(MODELS):
                sub = histories[(histories.experiment == ex) & (histories.source == source) & (histories.model == model)]
                for seed, color in zip(SEEDS, ['#23689b', '#df8126', '#32835d']):
                    h = sub[sub.seed == seed].sort_values('epoch')
                    ax, loss = axes[2*r, j], axes[2*r+1, j]
                    ax.plot(h.epoch, h.val_macro_f1, color=color, label=str(seed), lw=1.4)
                    best = h.loc[h.val_macro_f1.idxmax()]
                    ax.scatter([best.epoch], [best.val_macro_f1], s=45, facecolors='none', edgecolors=color, zorder=3)
                    loss.plot(h.epoch, h.train_loss, color=color, lw=1.2)
                    loss.plot(h.epoch, h.val_loss, color=color, ls='--', lw=1.2)
                ax.set(title=f'{LABELS[model]} | {source}', ylabel='Validation macro-F1', xlabel='Epoch', xlim=(1, 30), ylim=(0, 1.02))
                loss.set(ylabel='Cross-entropy loss', xlabel='Epoch', xlim=(1, 30))
                ax.grid(alpha=.15); loss.grid(alpha=.15)
        axes[0, 0].legend(title='Training seed', fontsize=8)
        fig.suptitle(f'{ex}: validation trajectories and training/validation loss\nOpen circles: selected checkpoints; solid loss: train; dashed loss: validation', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .94))
        save_figure(fig, figroot, f'Figure_S14_{ex}_learning_curves')
    effects = paired['fixed_target_f1_bootstrap']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for r, ex in enumerate(['P4', 'P5']):
        for c, target in enumerate(['ADT45', 'OTHER']):
            ax = axes[r, c]
            for j, model in enumerate(MODELS):
                row = effects[(effects.experiment == ex) & (effects.target == target) & (effects.model == model)].iloc[0]
                ax.errorbar(row.macro_f1_drop, j, xerr=[[row.macro_f1_drop-row.ci_low], [row.ci_high-row.macro_f1_drop]], fmt='o', color=COLORS[model], capsize=3)
            ax.set(title=f'{ex} | target {target}', yticks=range(4), yticklabels=[LABELS[m] for m in MODELS], xlim=(0, .75))
            ax.axvline(0, color='gray', ls='--'); ax.grid(axis='x', alpha=.2)
            if r == 1:
                ax.set_xlabel('Macro-F1 loss: same-domain control minus transfer')
    fig.tight_layout(); save_figure(fig, figroot, 'Figure_14A_fixed_target_losses')
    curves = tables['risk_coverage_by_run']
    for method in ['raw', 'ts']:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        for r, ex in enumerate(['P4', 'P5']):
            for c, target in enumerate(['ADT45', 'OTHER']):
                ax = axes[r, c]
                for model in MODELS:
                    for control in [False, True]:
                        sub = curves[(curves.experiment == ex) & (curves.target == target) &
                                     (curves.model == model) & (curves.method == method)]
                        sub = sub[sub.source == target] if control else sub[sub.source != target]
                        g = sub.groupby('requested_coverage').risk.agg(['mean', 'std'])
                        ax.plot(g.index, g['mean'], color=COLORS[model], ls='--' if control else '-', lw=1.3 if control else 1.8, label=LABELS[model] if not control else None)
                        if not control:
                            ax.fill_between(g.index, np.maximum(0, g['mean']-g['std']), np.minimum(1, g['mean']+g['std']), color=COLORS[model], alpha=.08)
                ax.set(title=f'{ex} | target {target}', xlabel='Retained fraction', ylabel='Error among retained images', ylim=(0, .85))
                ax.grid(alpha=.15)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'{method.upper()}: confidence-ranked risk (solid: transfer; dashed: same-domain)\nShading: SD across 3 fixed seeds; these curves do not select deployment thresholds', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .93)); save_figure(fig, figroot, f'Figure_14B_risk_coverage_{method}')
    p = paired['persistent_errors_summary']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for r, ex in enumerate(['P4', 'P5']):
        for c, target in enumerate(['ADT45', 'OTHER']):
            ax = axes[r, c]
            sub = p[(p.experiment == ex) & (p.target == target)].set_index('model').loc[MODELS]
            x = np.arange(4)
            ax.bar(x-.18, sub.wrong_highconf090_raw_all/sub.n, width=.36, color='#23689b', label='Raw')
            ax.bar(x+.18, sub.wrong_highconf090_ts_all/sub.n, width=.36, color='#df8126', label='Temperature-scaled')
            ax.set(title=f'{ex} | target {target}', xticks=x, xticklabels=[LABELS[m] for m in MODELS], ylabel='Fraction of all target images')
            ax.tick_params(axis='x', labelrotation=20); ax.grid(axis='y', alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Wrong with confidence >=0.90 in all three seeds\nTemperature scaling changes confidence, not predicted labels', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93)); save_figure(fig, figroot, 'Figure_14C_persistent_confident_errors')
    metric = tables['metrics_by_run']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for r, ex in enumerate(['P4', 'P5']):
        for c, target in enumerate(['ADT45', 'OTHER']):
            ax = axes[r, c]
            for j, model in enumerate(MODELS):
                sub = metric[(metric.experiment == ex) & (metric.target == target) &
                             (metric.source != target) & (metric.model == model)]
                for offset, column, color, label in [(-.10, 'ece_uniform_15', '#23689b', 'Uniform (15 bins)'),
                                                      (.10, 'ece_adaptive_10', '#df8126', 'Quantile (10 requested bins)')]:
                    raw = sub[sub.method == 'raw'][column].mean()
                    ts = sub[sub.method == 'ts'][column].mean()
                    ax.plot([raw, ts], [j+offset, j+offset], color=color, lw=1.6)
                    ax.scatter([raw], [j+offset], facecolors='white', edgecolors=color, s=42, zorder=3)
                    ax.scatter([ts], [j+offset], color=color, marker='s', s=30, zorder=3)
                    if r == 0 and c == 0 and j == 0:
                        ax.plot([], [], color=color, label=label)
            ax.set(title=f'{ex} | target {target}', yticks=range(4), yticklabels=[LABELS[m] for m in MODELS],
                   xlim=(0, .5), xlabel='Expected calibration error')
            ax.grid(axis='x', alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc='upper center', bbox_to_anchor=(.5, .91), ncol=2, frameon=False)
    fig.suptitle('Calibration transfer and binning sensitivity\nOpen circles: raw; filled squares: temperature-scaled; means over 3 fixed seeds', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .86)); save_figure(fig, figroot, 'Figure_14D_calibration_binning')



# Cell 11
def make_report(root, histories, tables, paired, audits, rules, n_boot):
    metric = tables['metrics_by_run']
    policy = tables['operating_by_run']
    overview = metric.assign(scenario=np.where(metric.experiment == 'P1', 'IID', np.where(metric.source == metric.target, 'control', 'cross'))).groupby(['experiment', 'scenario', 'method'])[['macro_f1', 'ece_uniform_15', 'ece_adaptive_10', 'brier', 'nll']].mean().reset_index()
    overview.to_csv(root/'tables/scenario_descriptive_summary.csv', index=False)
    # No equivale a un bootstrap sobre modelos ni a poblaciones nuevas.
    curve_summary = histories.loc[histories.groupby(['experiment', 'source', 'model', 'seed']).val_macro_f1.idxmax(), ['experiment', 'source', 'model', 'seed', 'epoch', 'val_macro_f1']]
    curve_summary = curve_summary.rename(columns={'epoch': 'best_epoch', 'val_macro_f1': 'best_validation_macro_f1'})
    curve_summary.to_csv(root/'tables/checkpoint_summary.csv', index=False)
    selection = paired['cross_operating_bootstrap']
    source_policy = selection[(selection.method == 'ts') & (selection.policy == 'source_validation_5pct')]
    fixed_policy = policy[(policy.experiment == 'P5') & (policy.target == 'ADT45') & (policy.model == 'convnext_tiny') & (policy.method == 'raw') & (policy.policy == 'fixed_090')].groupby('source')[['coverage', 'risk']].mean().reset_index()
    findings = {'n_histories': int(histories.groupby(['experiment', 'source', 'model', 'seed']).ngroups),
                'n_test_evaluations': int(metric[metric.method == 'raw'].shape[0]), 'n_validation_caches': 60,
                'n_numeric_checks': len(audits), 'numeric_max_absolute_difference': float(audits.absolute_difference.max()),
                'best_checkpoint_at_epoch30': int((curve_summary.best_epoch == 30).sum()),
                'validation_rules_without_eligible_threshold': int((~rules.eligible).sum()), 'bootstrap_replicates': n_boot,
                'p5_convnext_target_ADT45_raw_threshold090': fixed_policy.to_dict('records')}
    (root/'summary.json').write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding='utf-8')
    question = '¿Persiste el deterioro del rendimiento y la confiabilidad bajo un cambio de dominio asociado al cultivar después de igualar clase y edad, y qué utilidad conservan la calibración en origen y las reglas de aceptación por confianza?'
    limitations = ('Los análisis nuevos son exploratorios y se basan en un test previamente inspeccionado. Los IC95% remuestrean imágenes dentro de clase y mantienen fijos los modelos y las tres semillas; no cubren nuevas fincas, fechas, cámaras o realizaciones de entrenamiento. En cada objetivo de P5, remuestrear imágenes equivale a remuestrear sus pares porque cada par aporta una imagen a ese objetivo. No se usa el contraste P4–P5 para calcular una fracción causal explicada por edad. Las reglas fuente usan una validación ya empleada para checkpoint y temperatura: satisfacer 5% de error empírico allí no garantiza 5% en otro dominio. Adaptive ECE se define sobre máxima confianza con bordes cuantiles y fusión de empates. Brier es la suma multiclase sin dividir por el número de clases; NLL usa logaritmo natural. No se compara el Brier de 5 y 10 clases como una escala idéntica. Intervalos de riesgo con menos de 95% de réplicas definidas se omiten. No se asigna riesgo cero al rechazo total. Las curvas de riesgo ordenadas en test son diagnósticos retrospectivos y se distinguen de umbrales elegidos en validación.')
    sections = [('<h1>Refuerzo metodológico de Paddy Doctor</h1><p>Julián Coronel Reyes · 20 noviembre 2026</p>'
                 '<p>Resultados calculados con predicciones e historiales existentes. No se realizó entrenamiento de redes.</p>'
                 '<h2>Pregunta propuesta</h2><p>'+html.escape(question)+'</p>'),
                '<h2>Verificación</h2><pre>'+html.escape(json.dumps(findings, indent=2, ensure_ascii=False))+'</pre>',
                '<h2>Resumen descriptivo</h2><p>Promedios de arquitecturas, semillas y direcciones; no son comparaciones inferenciales entre P1, P4 y P5.</p>'+overview.round(5).to_html(index=False),
                '<h2>Pérdidas sobre objetivos fijos</h2>'+paired['fixed_target_f1_bootstrap'].round(5).to_html(index=False),
                '<h2>Regla elegida en validación fuente</h2><p>Mayor cobertura con error empírico fuente ≤5%, al menos 20% y 20 casos. Tabla: probabilidades calibradas; se aplica la misma regla sin ajuste a test. IC condicionados a las reglas ya elegidas.</p>'+source_policy.round(5).to_html(index=False),
                '<h2>Fallos persistentes</h2>'+paired['persistent_errors_summary'].to_html(index=False),
                '<h2>Límites de interpretación</h2><p>'+html.escape(limitations)+'</p>',
                '<h2>Estructura sugerida</h2><ol><li>Rendimiento IID y estabilidad del entrenamiento</li><li>Transferencia sobre objetivos fijos</li><li>Sensibilidad con emparejamiento por clase y edad</li><li>Calibración y transferencia de umbrales</li><li>Fallos persistentes y resultados por clase</li><li>Selección de modelos y límites de validez</li></ol>',
                '<h2>Figuras</h2>']
    for name in sorted((root/'figures').glob('*.png')):
        sections.append(f'<figure><img src="figures/{name.name}" alt="{name.stem}"><figcaption>{name.stem}</figcaption></figure>')
    content = '<!doctype html><html lang="es"><meta charset="utf-8"><title>Paddy Doctor 14</title><style>body{font:16px/1.55 Arial,sans-serif;max-width:1300px;margin:40px auto;padding:0 24px;color:#182b3a}h1,h2{color:#153e5a}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto;margin:20px 0}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf3f8}pre{white-space:pre-wrap;background:#f3f6f8;padding:18px}img{max-width:100%}figure{margin:30px 0}figcaption{font-size:12px}</style>'+''.join(sections)+'</html>'
    (root/'Informe_PaddyDoctor_14.html').write_text(content, encoding='utf-8')
    (root/'METHODS_AND_LIMITATIONS.txt').write_text(question+'\n\n'+limitations, encoding='utf-8')
    return findings


def run_analysis(project_root, output_root, portable=False, n_boot=2000):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for sub in ['tables', 'figures']:
        (root/sub).mkdir(exist_ok=True)
    print('1/5 Verificando predicciones, clases, temperaturas y validación...')
    sources, entries, histories, audits, rules = inspect_sources(project_root, portable)
    audits.to_csv(root/'tables/numerical_verification.csv', index=False)
    rules.to_csv(root/'tables/source_validation_thresholds.csv', index=False)
    histories.to_csv(root/'tables/training_histories.csv', index=False)
    print('2/5 Calculando métricas y reglas de aceptación...')
    tables = build_tables(entries, root/'tables')
    print(f'3/5 Bootstrap pareado ({n_boot} réplicas), McNemar y fallos persistentes...')
    paired = paired_analyses(entries, root/'tables', n_boot)
    print('4/5 Generando figuras...')
    make_figures(histories, tables, paired, root/'figures')
    print('5/5 Generando informe y procedencia...')
    findings = make_report(root, histories, tables, paired, audits, rules, n_boot)
    config = {'analysis_version': '14.1.0', 'status': 'exploratory_retrospective', 'n_boot': n_boot,
              'bootstrap_unit': 'class-stratified target images; same indices across models and seeds',
              'fixed_training_seeds': SEEDS, 'source_risk_limit': .05, 'minimum_coverage': .20,
              'minimum_selected_validation': 20, 'fixed_confidence_thresholds': [.90, .95],
              'holm_family': '12 McNemar comparisons within each protocol and target',
              'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
              'retraining': False, 'external_validation': False}
    (root/'analysis_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    manifest = []
    if portable:
        input_files = sorted((Path(project_root)/'inputs').rglob('*'))
    else:
        input_files = [Path(project_root)/'results/09_calibration_reliability/09_temperature_parameters_validation.csv']
        input_files += sorted((Path(project_root)/'results/09_calibration_reliability/validation_cache').glob('*.npz'))
        input_files += [s.path/n for s in sources.values() for n in s.names if n.endswith(('history.csv', 'test_predictions.csv', '09_test_calibration_by_run.csv'))]
    for path in input_files:
        if path.is_file():
            manifest.append({'path': str(path.relative_to(project_root)), 'bytes': path.stat().st_size,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(manifest).to_csv(root/'input_sha256.csv', index=False)
    print(json.dumps(findings, indent=2, ensure_ascii=False))
    print('Terminado. Informe:', root/'Informe_PaddyDoctor_14.html')
    return findings



# Cell 13
findings = run_analysis(PROJECT_ROOT, OUTPUT_ROOT, portable=PORTABLE, n_boot=N_BOOTSTRAP)

# Cell 15
operating = pd.read_csv(OUTPUT_ROOT/'tables/cross_operating_bootstrap.csv')
selected = operating[(operating.experiment == 'P5') & (operating.method == 'ts') &
                     (operating.policy == 'source_validation_5pct')]
columns = ['target', 'model', 'coverage', 'risk', 'risk_ci_low', 'risk_ci_high']
print(selected[columns].round(4).to_string(index=False))
print('Informe HTML:', OUTPUT_ROOT/'Informe_PaddyDoctor_14.html')

# Cell 16
# Vista previa; los archivos originales permanecen disponibles en figures/.
try:
    from IPython.display import display, Image
except ImportError:
    print('Vistas previas disponibles en:', OUTPUT_ROOT/'figures')
else:
    for name in ['Figure_14A_fixed_target_losses.png', 'Figure_14B_risk_coverage_raw.png',
                 'Figure_14C_persistent_confident_errors.png', 'Figure_14D_calibration_binning.png']:
        display(Image(filename=str(OUTPUT_ROOT/'figures'/name), width=1050))
