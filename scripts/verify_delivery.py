"""Reopen the real delivery, validate manifests, PDF structure and every ZIP CRC."""
import hashlib
import json
from pathlib import Path
import zipfile
from pypdf import PdfReader
from src.paths import delivery_dir,workspace_root


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    final=delivery_dir();manifest=json.loads((final/'SHA256_MANIFEST.json').read_text(encoding='utf-8'))
    for name,expected in manifest.items():
        path=final/name
        if not path.is_file() or sha(path)!=expected:raise ValueError('Final checksum mismatch: '+name)
    if sha(final/'Formal_Report.pdf')!=sha(final/'Report_Source/main.pdf'):raise ValueError('Delivered report differs from the compiled report source PDF')
    qa=json.loads((final/'QA_APPROVED.json').read_text(encoding='utf-8'));pages={}
    for name,expected in qa['files'].items():
        if sha(final/name)!=expected:raise ValueError('PDF changed after visual review: '+name)
        reader=PdfReader(final/name)
        if reader.is_encrypted:raise ValueError('Unexpected encrypted PDF')
        pages[name]=len(reader.pages)
        if any(not p.extract_text().strip() for p in reader.pages):raise ValueError('Blank/unextractable PDF page: '+name)
        if name=='Formal_Report.pdf':
            reference_pages=[i for i,p in enumerate(reader.pages) if p.extract_text().lstrip().startswith('References')]
            if not reference_pages or reference_pages[0]>6:raise ValueError('Formal report exceeds six body pages or has no separate references')
        if name=='ECE661_Poster.pdf':
            text=reader.pages[0].extract_text()
            if any(s.lower() in text.lower() for s in ['Homer Simpson','Lorem ipsum','insert title','insert author']):raise ValueError('Poster placeholder text retained')
            if len(reader.pages)!=1:raise ValueError('Expected one-page poster')
    archives={}
    for letter,count in [('A',0),('B',132),('C',231),('D',222)]:
        path=final/f'{letter}_Code.zip'
        with zipfile.ZipFile(path) as archive:
            names=archive.namelist()
            if archive.testzip() is not None:raise ValueError('Corrupt ZIP: '+path.name)
            patches=sum(n.endswith('/patch.pt') and '/artifacts/patches/' in n for n in names)
            if patches!=count:raise ValueError(f'Incorrect {letter} patch count: {patches}')
            if letter=='A' and sum(n.endswith('/best.pt') for n in names)!=3:raise ValueError('Expected three selected classifier checkpoints')
            if any(n.startswith(('/', '\\')) or '..' in Path(n).parts for n in names):raise ValueError('Unsafe archive member path')
            archives[path.name]=dict(members=len(names),patches=patches,crc='passed')
    with zipfile.ZipFile(final/'CIFAR10_Dataset.zip') as archive:
        if archive.testzip() is not None:raise ValueError('Dataset ZIP CRC failure')
        h=hashlib.md5()
        with archive.open('ECE661_Project/data/cifar-10-python.tar.gz') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        if h.hexdigest()!='c58f30108f718f92721af3b95e74349a':raise ValueError('Official dataset MD5 mismatch')
        archives['CIFAR10_Dataset.zip']=dict(official_md5=h.hexdigest(),crc='passed')
    with zipfile.ZipFile(final/'Shared_Results.zip') as archive:
        names=archive.namelist()
        if archive.testzip() is not None:raise ValueError('Shared result CRC failure')
        test=sum('/results/raw/' in n and n.endswith('.npz') for n in names)
        validation=sum('/results/validation_raw/' in n and n.endswith('.npz') for n in names)
        if (test,validation)!=(1638,1212):raise ValueError(f'Wrong raw condition counts: {test}, {validation}')
        archives['Shared_Results.zip']=dict(members=len(names),test_conditions=test,full_validation_conditions=validation,crc='passed')
    full=workspace_root()/'outputs/ECE661_Full_Deliverables.zip'
    with zipfile.ZipFile(full) as archive:
        if archive.testzip() is not None:raise ValueError('Full delivery CRC failure')
        for name,expected in manifest.items():
            h=hashlib.sha256()
            with archive.open('ECE661_Final/'+name) as stream:
                for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
            if h.hexdigest()!=expected:raise ValueError('Full archive content differs: '+name)
    # This receipt remains outside the frozen final folder, so it cannot invalidate
    # the self-consistent archive/manifest that it independently verifies.
    receipt=dict(status='verified',manifest_files=len(manifest),pdf_pages=pages,archives=archives,full_archive_sha256=sha(full),full_archive_bytes=full.stat().st_size)
    (workspace_root()/'outputs/FINAL_VERIFICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
