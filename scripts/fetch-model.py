#!/usr/bin/env python3
"""Stage a hash-pinned GGUF pair into a workload model volume."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        if stream.read(4) != b'GGUF':
            raise ValueError(f'{path}: not a GGUF file')
        stream.seek(0)
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stage(manifest, destination):
    entries = [manifest['model']]
    if manifest.get('projector'):
        entries.append(manifest['projector'])
    paths = set()
    for entry in entries:
        path = PurePosixPath(entry['path'])
        if path.parent != PurePosixPath('/models') or path.name in ('', '.', '..'):
            raise ValueError('Model paths must be direct files under /models')
        if path.name in paths or path.name == 'model.json':
            raise ValueError('Model/projector paths must be distinct from each other and model.json')
        paths.add(path.name)
        if not re.fullmatch(r'[0-9a-f]{64}', entry['sha256']):
            raise ValueError('Each entry requires a lowercase SHA256')
        url = urlparse(entry['source'])
        if url.scheme != 'https' or not url.netloc:
            raise ValueError('Each source must be an HTTPS URL')
    destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        target = destination / PurePosixPath(entry['path']).name
        if target.exists():
            if digest(target) != entry['sha256']:
                raise ValueError(f'{target}: existing file hash differs; use a fresh model directory')
        else:
            with tempfile.NamedTemporaryFile(dir=destination, prefix='.download-', delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    with urlopen(entry['source'], timeout=120) as response:
                        if urlparse(response.geturl()).scheme != 'https':
                            raise ValueError('Download redirected to a non-HTTPS URL')
                        for chunk in iter(lambda: response.read(1024 * 1024), b''):
                            stream.write(chunk)
                    stream.flush()
                    if digest(temporary) != entry['sha256']:
                        raise ValueError(f'{target.name}: downloaded SHA256 mismatch')
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
        print(f'Verified {target.name}: {entry["sha256"]}')
    with tempfile.NamedTemporaryFile(mode='w', dir=destination, prefix='.manifest-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(manifest, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.replace(temporary, destination / 'model.json')
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=Path('/models'))
    args = parser.parse_args()
    try:
        stage(json.loads(args.manifest.read_text()), args.destination)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
