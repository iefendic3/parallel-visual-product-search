"""Small, dependency-free helpers shared by the local ABO pipeline."""
import csv
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / 'data/abo/manifest.csv'
MANIFEST_FIELDS = [
    'image_id', 'product_id', 'title', 'image_path', 'category', 'source_url',
    'abo_image_id', 'domain_name',
]


def valid_id(value):
    if not value or value.startswith('#') or value != value.strip() or any(c in value for c in ',\r\n\0'):
        raise ValueError(f'Invalid vector image_id: {value!r}')
    return value


def manifest_rows(manifest):
    """Stream records. image_path is always relative to the manifest directory."""
    with Path(manifest).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        required = {'image_id', 'product_id', 'image_path'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Manifest must have columns: {sorted(required)}')
        for line, row in enumerate(reader, 2):
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError(f'Malformed manifest record near line {line}')
            valid_id(row['image_id'])
            if not row['image_path']:
                raise ValueError(f'Missing image_path near line {line}')
            yield row


def image_path(manifest, row):
    path = Path(row['image_path']).expanduser()
    return (Path(manifest).resolve().parent / path).resolve()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def atomic_text(path):
    """Replace a generated text file only after a successful write and fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    with atomic_text(path) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')
