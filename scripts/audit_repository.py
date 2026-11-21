"""Validate the distributed code inventory and frozen scientific file checksums."""
from pathlib import Path
import ast,csv,hashlib,json,subprocess,sys,zipfile
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproducibility'))
from restore_large_files import restore
restore()
errors=[]
notebooks=sorted((ROOT/'notebooks').glob('*.ipynb'))
exports=sorted(p for p in (ROOT/'scripts').glob('*.py') if p.name!='audit_repository.py')
if {p.name[:2] for p in notebooks}!={f'{i:02d}' for i in range(20)}:errors.append('Expected notebook identifiers 00 through 19')
if {p.stem for p in notebooks}!={p.stem for p in exports}:errors.append('Notebook/Python export inventory differs')
for path in notebooks:
    try:
        n=json.loads(path.read_text());assert n['nbformat']==4 and isinstance(n['cells'],list)
        assert all(c['cell_type'] in {'code','markdown','raw'} for c in n['cells'])
    except Exception as exc:errors.append(f'Notebook JSON {path.relative_to(ROOT)}: {exc}')
for path in list((ROOT/'scripts').glob('*.py'))+list((ROOT/'reproducibility').rglob('*.py')):
    try:ast.parse(path.read_text(),filename=str(path))
    except Exception as exc:errors.append(f'Python syntax {path.relative_to(ROOT)}: {exc}')
try:
    citation=yaml.safe_load((ROOT/'CITATION.cff').read_text())
    assert citation['cff-version']=='1.2.0' and citation['license']=='MIT'
    assert [(a['given-names'],a['family-names']) for a in citation['authors']]==[
        ('Julián','Coronel-Reyes'),('Bryan Iván','Barahona-Montalván'),('Bryan Orlando','Vélez-San Martín')]
    assert (ROOT/'LICENSE').read_text().startswith('MIT License\n')
except Exception as exc:errors.append(f'Citation/license: {exc}')

manifest=ROOT/'reproducibility/MANIFEST_SHA256.csv';count=0;listed=set()
try:
    for row in csv.DictReader(manifest.open()):
        rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
        p=manifest.parent/rel;assert p.stat().st_size==int(row['bytes'])
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
        listed.add(rel.as_posix());count+=1
    actual={p.relative_to(manifest.parent).as_posix() for p in manifest.parent.rglob('*') if p.is_file() and p!=manifest and '__pycache__' not in p.parts}
    assert listed==actual,'Manifest does not match distributed files'
except Exception as exc:errors.append(f'Frozen-output manifest: {exc}')

allowed_archives={
    'reproducibility/19_PaddyDoctor_RESULTADOS.zip',
    *['reproducibility/inputs/'+n for n in ['p1.zip','p4.zip','p5.zip','cal.zip','15_P5_Rematching_5x3_30ep_v1_RESULTADOS.zip']]}
tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
for item in tracked:
    p=ROOT/item;parts={x.lower() for x in Path(item).parts}
    if p.name.lower()=='kaggle.json' or '.kaggle' in parts or p.name.startswith('.env') or 'config' in parts:
        errors.append('Credential/private configuration path: '+item)
    if p.suffix.lower() in {'.pt','.pth','.ckpt','.jpg','.jpeg','.webp','.tif','.tiff'}:errors.append('Image/checkpoint payload: '+item)
    if p.suffix.lower()=='.zip':
        if item not in allowed_archives:errors.append('Unexpected archive: '+item);continue
        try:
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    q=Path(name)
                    assert not q.is_absolute() and '..' not in q.parts
                    assert q.suffix.lower() not in {'.pt','.pth','.ckpt','.jpg','.jpeg','.webp','.tif','.tiff'},name
                    assert q.name.lower()!='kaggle.json' and '.kaggle' not in {x.lower() for x in q.parts},name
        except Exception as exc:errors.append(f'Archive {item}: {exc}')
tables=ROOT/'results/manuscript_tables'
if len(list(tables.glob('manuscript_table_*.csv')))!=9:errors.append('Expected nine main tables')
if len(list(tables.glob('supplement_table_*.csv')))!=51:errors.append('Expected 51 supplementary tables')
if errors:
    print('\n'.join(errors));sys.exit(1)
print(f'Repository audit passed: {len(notebooks)} notebooks, {len(exports)} exports, {count} checksummed scientific files and three cited authors')
