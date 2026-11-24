"""Verify and concatenate all release segments into the original complete ZIP."""
import argparse,hashlib,json,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parts-dir',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,default=Path('ECE661_Full_Deliverables.zip'))
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    record=json.loads(Path(__file__).with_name('release_assets.json').read_text(encoding='utf-8'))['full_archive']
    whole=hashlib.sha256();total=0;stream=None
    temporary=a.output.with_name(a.output.name+'.assembling')
    if not a.verify_only:
        if a.output.exists() or temporary.exists():raise FileExistsError('Choose a new output path; existing files are preserved.')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        stream=temporary.open('xb')
    try:
        for part in record['parts']:
            name=part['name']
            if Path(name).name!=name or '\\' in name:raise ValueError('Invalid segment name')
            path=a.parts_dir/name
            if path.stat().st_size!=part['bytes']:raise ValueError('Incorrect segment size: '+name)
            h=hashlib.sha256()
            with path.open('rb') as source:
                for block in iter(lambda:source.read(4*1024*1024),b''):
                    h.update(block);whole.update(block);total+=len(block)
                    if stream:stream.write(block)
            if h.hexdigest()!=part['sha256']:raise ValueError('Segment SHA-256 mismatch: '+name)
            print('Verified '+name,flush=True)
        if total!=record['bytes'] or whole.hexdigest()!=record['sha256']:
            raise ValueError('Complete archive checksum mismatch')
    finally:
        if stream:stream.close()
    if not a.verify_only:os.replace(temporary,a.output)
    print(('Verified complete byte stream: ' if a.verify_only else 'Reconstructed archive: ')+record['sha256'],flush=True)

if __name__=='__main__':main()
