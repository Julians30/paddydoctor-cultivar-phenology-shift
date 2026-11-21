"""Paddy19.2: resolve original images, including partial caches and ZIPs.

No network or credential access occurs in this module. Kaggle restoration is a
separate, explicitly confirmed notebook step. Only images in frozen protocols
are persisted; every image must match its historical SHA-256.
"""
from pathlib import Path, PurePosixPath
import os, json, hashlib, zipfile, shutil
import pandas as pd

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def required_images(meta_path,p4_path,p5_path,designs):
    meta=pd.read_csv(meta_path)
    p4=pd.read_csv(p4_path);p5=pd.read_csv(p5_path)
    labels=['blast','dead_heart','downy_mildew','hispa','normal']
    ids=set(p4.loc[p4.label.isin(labels),'image_id'])|set(p5.image_id)
    for d,domains in designs.values():ids.update(d.image_id)
    selected=meta[meta.image_id.isin(ids)].copy().sort_values('image_id')
    if len(selected)!=len(ids) or selected.image_id.duplicated().any():
        raise ValueError('Los IDs requeridos no coinciden con la metadata original.')
    return selected

def discover_sources(project,explicit='',local_root='/content',home=None):
    """Bounded search; do not walk models/results or inspect credentials."""
    project=Path(project);local_root=Path(local_root);home=Path(home) if home else Path.home()
    roots=[];archives=[];seen=set()
    bases=([Path(explicit).expanduser()] if explicit else [])+[
        project/'data',project/'inbox',local_root/'paddy-disease-classification',
        local_root/'paddy15_verified_images',local_root/'paddy19_image_cache',
        home/'.cache/kagglehub/competitions',local_root/'datasets',local_root/'data']
    # Other immediate runtime folders can contain an uploaded dataset ZIP.
    if local_root.exists():
        for p in sorted(local_root.iterdir()):
            if p.name not in ['drive','sample_data','paddy19_code','paddy19_verified']:
                bases.append(p)
    classes={'blast','dead_heart','hispa','normal','downy_mildew'}
    for base in bases:
        if not base.exists():continue
        if base.is_file():
            if zipfile.is_zipfile(base) and str(base.resolve()) not in seen:
                archives.append(base);seen.add(str(base.resolve()))
            continue
        for folder,dirs,files in os.walk(base):
            p=Path(folder);depth=len(p.relative_to(base).parts)
            dirs[:]=sorted(d for d in dirs if d not in ['models','results','exports','.git','config']) if depth<5 else []
            if p.name=='train_images' or classes.intersection(dirs) or (p/'blast').is_dir():
                key=str(p.resolve())
                if key not in seen:roots.append(p);seen.add(key)
                dirs[:]=[]
            for name in sorted(files):
                if name.lower().endswith('.zip'):
                    z=p/name;key=str(z.resolve())
                    if key not in seen:archives.append(z);seen.add(key)
    return roots,archives

def restore_verified(selected,roots,archives,destination):
    """Copy matching bytes atomically. ZIP entries are never extractall'ed."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    todo={r.image_id:r for r in selected.itertuples(index=False)};receipts=[]
    def install(iid,src=None,payload=None,origin=''):
        r=todo[iid];dst=destination/r.label/iid
        if src is not None and sha(src)!=r.sha256:return False
        if payload is not None and hashlib.sha256(payload).hexdigest()!=r.sha256:return False
        dst.parent.mkdir(parents=True,exist_ok=True)
        if src is None or Path(src).resolve()!=dst.resolve():
            tmp=dst.with_name(dst.name+'.restore.tmp')
            if src is None:tmp.write_bytes(payload)
            else:shutil.copyfile(src,tmp)
            tmp.replace(dst)
        from PIL import Image
        with Image.open(dst) as im:im.verify()
        receipts.append(dict(image_id=iid,label=r.label,sha256=r.sha256,origin=origin))
        del todo[iid]
        return True
    for root in [destination]+[Path(p) for p in roots]:
        for iid,r in list(todo.items()):
            for src in [root/r.label/iid,root/iid]:
                if src.is_file() and install(iid,src=src,origin=str(src)):break
        if not todo:break
    for archive in archives:
        if not todo:break
        try:
            with zipfile.ZipFile(archive) as z:
                for info in z.infolist():
                    p=PurePosixPath(info.filename.replace('\\','/'));iid=p.name
                    if info.is_dir() or iid not in todo or '..' in p.parts or p.is_absolute():continue
                    r=todo[iid]
                    if info.file_size>50*1024*1024:continue
                    if len(p.parts)>1 and p.parts[-2] not in [r.label,'train_images']:continue
                    install(iid,payload=z.read(info),origin=str(archive)+'::'+info.filename)
        except zipfile.BadZipFile:
            print('ZIP incompleto, se omite:',Path(archive).name,flush=True)
    report=destination.parent/'paddy19_restore_report.json'
    report.write_text(json.dumps(dict(required=len(selected),verified=len(selected)-len(todo),
        missing_count=len(todo),missing_ids=sorted(todo),image_root=str(destination)),indent=2)+'\n')
    print(f'Imágenes originales verificadas: {len(selected)-len(todo)}/{len(selected)}; faltan {len(todo)}',flush=True)
    return destination,sorted(todo),receipts
