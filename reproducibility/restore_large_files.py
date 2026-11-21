"""Reassemble exact large scientific files from checksummed transport parts."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def local(relative):
    p=(ROOT/relative).resolve()
    if ROOT.resolve() not in p.parents:raise ValueError('Unsafe relative path')
    return p

def restore():
    spec=json.loads((ROOT/'large_files.json').read_text())
    for item in spec['files']:
        target=local(item['path'])
        if target.exists():
            if target.stat().st_size==item['bytes'] and digest(target)==item['sha256']:continue
            raise ValueError('Existing file has an unexpected hash: '+item['path'])
        target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.restoring')
        try:
            with tmp.open('wb') as out:
                for part in item['parts']:
                    p=local(part['path'])
                    if p.stat().st_size!=part['bytes'] or digest(p)!=part['sha256']:raise ValueError('Part checksum mismatch: '+part['path'])
                    with p.open('rb') as inp:
                        for b in iter(lambda:inp.read(1024*1024),b''):out.write(b)
            if tmp.stat().st_size!=item['bytes'] or digest(tmp)!=item['sha256']:raise ValueError('Reconstructed checksum mismatch')
            tmp.replace(target)
        finally:
            if tmp.exists():tmp.unlink()
        print('Restored '+item['path'])

if __name__=='__main__':restore()
