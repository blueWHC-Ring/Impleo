"""Download and verify versioned Impleo data; Python standard library only."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import urllib.request

PACKS = {'example': {'pack': 'example', 'parts': [{'name': 'example.tar.gz', 'bytes': 79511244, 'sha256': '86ee2ebea09ccc706170c65c9b0ccae9d06252f93e3717414da364efa6b2e9b9', 'url': 'https://github.com/blueWHC-Ring/Impleo-data/releases/download/v1.0.0/example.tar.gz'}]}, 'training': {'pack': 'training', 'parts': [{'name': 'training.tar.gz.part00', 'bytes': 1073741824, 'sha256': '96b8a0cffce249d2eb2cfdce47f9de32744f1d18ec487fc47b97d954aa64d39f', 'url': 'https://github.com/blueWHC-Ring/Impleo-data/releases/download/v1.0.0/training.tar.gz.part00'}, {'name': 'training.tar.gz.part01', 'bytes': 1073741824, 'sha256': '2b796b81f250aa5e208da0db3d82e817a607659538a25e06d3eb0b17f627b990', 'url': 'https://github.com/blueWHC-Ring/Impleo-data/releases/download/v1.0.0/training.tar.gz.part01'}, {'name': 'training.tar.gz.part02', 'bytes': 1073741824, 'sha256': '54fc471c749441893b448509f07ba1b4c69042b80c4a634915080b32e460d46d', 'url': 'https://github.com/blueWHC-Ring/Impleo-data/releases/download/v1.0.0/training.tar.gz.part02'}, {'name': 'training.tar.gz.part03', 'bytes': 1073741824, 'sha256': '9aec9a5c68d8b66d36fbde7c16802e31709e101684a42c978e31f569c071f575', 'url': 'https://github.com/blueWHC-Ring/Impleo-data/releases/download/v1.0.0/training.tar.gz.part03'}, {'name': 'training.tar.gz.part04', 'bytes': 1073741824, 'sha256': '6e3178b953224f52d5f9148365003eb40821539f1522cd356c185ad0e5da29fd', 'url': 'https://github.com/blueWHC-Ring/Impleo-data/releases/download/v1.0.0/training.tar.gz.part04'}, {'name': 'training.tar.gz.part05', 'bytes': 1073741824, 'sha256': '902f7db22d6351661c0ca0018472f1fa31bb0d75b30cf357985887435f302896', 'url': 'https://github.com/blueWHC-Ring/Impleo-data/releases/download/v1.0.0/training.tar.gz.part05'}, {'name': 'training.tar.gz.part06', 'bytes': 1073741824, 'sha256': 'f6361c62e42b9be6d1b0bd8c0b65a77c95f78624b0b2a9806b948639b5ef3ffe', 'url': 'https://github.com/blueWHC-Ring/Impleo-data/releases/download/v1.0.0/training.tar.gz.part06'}, {'name': 'training.tar.gz.part07', 'bytes': 652892140, 'sha256': 'd11de9844c3209a93e7162d286b1152d7aac2a77b83f4125b2fc6680d5c58769', 'url': 'https://github.com/blueWHC-Ring/Impleo-data/releases/download/v1.0.0/training.tar.gz.part07'}]}, 'training_metadata': {'pack': 'training_metadata', 'parts': [{'name': 'training_metadata.tar.gz', 'bytes': 129719, 'sha256': 'cdf728143f4425800282f013d1744d128b22f9626de4bd7b150a18ab0a92aa62', 'url': 'https://github.com/blueWHC-Ring/Impleo-data/releases/download/v1.0.0/training_metadata.tar.gz'}]}}

REMAP = {"pretraining/source_h5ad_audit.clean_v2.csv": "pretraining/source_assays.csv"}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

class JoinedReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None
    def readable(self): return True
    def read(self, size=-1):
        if size < 0: raise ValueError('Streaming reader requires a bounded read')
        blocks, remaining = [], size
        while remaining:
            if self.current is None:
                try: self.current = next(self.paths).open('rb')
                except StopIteration: break
            block = self.current.read(remaining)
            if not block:
                self.current.close(); self.current = None
            else:
                blocks.append(block); remaining -= len(block)
        return b''.join(blocks)
    def close(self):
        if self.current is not None: self.current.close()
        super().close()

def download(item, cache):
    path = cache / item['name']
    if path.exists() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256']:
        return path
    tmp = path.with_name(path.name + '.partial')
    for attempt in range(4):
        if tmp.exists() and tmp.stat().st_size >= item['bytes']:
            if tmp.stat().st_size == item['bytes'] and sha(tmp) == item['sha256']:
                tmp.replace(path)
                return path
            tmp.unlink()
        offset = tmp.stat().st_size if tmp.exists() else 0
        headers = {'User-Agent': 'Impleo-reproduction/0.2'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        print(f"Downloading {item['name']} ({offset}/{item['bytes']} bytes)", flush=True)
        try:
            request = urllib.request.Request(item['url'], headers=headers)
            with urllib.request.urlopen(request, timeout=120) as source:
                if source.status == 206:
                    if not source.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected byte range: ' + item['name'])
                else:
                    offset = 0  # A server without range support requires a fresh download.
                received, next_update = offset, offset + (32 << 20)
                with tmp.open('ab' if offset else 'wb') as target:
                    while True:
                        block = source.read(1 << 20)
                        if not block:
                            break
                        target.write(block)
                        received += len(block)
                        if received >= next_update:
                            print(f"  {item['name']}: {received/item['bytes']:.1%}", flush=True)
                            next_update = received + (32 << 20)
            if tmp.stat().st_size != item['bytes'] or sha(tmp) != item['sha256']:
                raise ValueError('Downloaded asset checksum/length mismatch: ' + item['name'])
            tmp.replace(path)
            return path
        except (OSError, ValueError) as error:
            if attempt == 3:
                raise
            print(f'Retrying {item["name"]}: {type(error).__name__}', flush=True)
            time.sleep(attempt + 1)

def extract(paths, out):
    with JoinedReader(paths) as stream, tarfile.open(fileobj=stream, mode='r|gz') as tar:
        for member in tar:
            name = member.name
            for old, new in REMAP.items():
                if name == old or name.startswith(old + '/'):
                    name = new + name[len(old):]; break
            rel = PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts or '\\' in name:
                raise ValueError('Unsafe archive path: ' + name)
            dest = (out / name).resolve()
            if not dest.is_relative_to(out.resolve()): raise ValueError('Archive escape: ' + name)
            if member.isdir(): dest.mkdir(parents=True, exist_ok=True); continue
            if not member.isfile(): raise ValueError('Unsupported archive member: ' + name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + '.extracting')
            with tar.extractfile(member) as source, tmp.open('wb') as target:
                shutil.copyfileobj(source, target, length=8 << 20)
            if dest.exists():
                if sha(dest) != sha(tmp):
                    tmp.unlink(); raise FileExistsError('Different existing asset: ' + str(dest))
                tmp.unlink()
            else: tmp.replace(dest)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'assets')
    ap.add_argument('--packs', nargs='+', choices=list(PACKS), default=['example'])
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--keep-downloads', action='store_true')
    ap.add_argument('--verify-only', action='store_true', help='Verify downloads without extracting')
    args = ap.parse_args()
    packs = PACKS
    if args.list:
        for key, pack in packs.items():
            print(key, sum(p['bytes'] for p in pack['parts']), 'download bytes')
        return
    selected = list(packs) if args.packs == ['all'] else list(dict.fromkeys(args.packs))
    if set(selected) - set(packs): raise ValueError('Unknown packs: ' + str(set(selected) - set(packs)))
    if 'training' in selected and 'training_metadata' not in selected: selected.append('training_metadata')
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / '.downloads'; cache.mkdir(exist_ok=True)
    for key in selected:
        receipt = args.out / ('.' + key + '.download.json')
        record = packs[key]
        if receipt.exists() and not args.verify_only:
            previous = json.loads(receipt.read_text())
            if previous.get('pack') != record: raise ValueError('Pack version changed: ' + key)
            print('Already installed: ' + key + '', flush=True)
            continue
        paths = [download(item, cache) for item in record['parts']]
        if not args.verify_only:
            print('Extracting ' + key, flush=True); extract(paths, args.out)
            receipt.write_text(json.dumps(dict(pack=record, remap=REMAP), indent=2), encoding='utf-8')
        if not args.keep_downloads:
            for path in paths: path.unlink()
        print('Verified: ' + key, flush=True)

if __name__ == '__main__': main()
