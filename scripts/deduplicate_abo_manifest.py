#!/usr/bin/env python3
"""Create one vector record per local ABO image, without decoding any images."""
import argparse
import csv
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

from abo_common import PROJECT_ROOT, atomic_text, file_sha256, image_path, manifest_rows, valid_id, write_json

UNIQUE_FIELDS = ['image_id', 'abo_image_id', 'image_path']


def deduplicate_manifest(manifest, output, progress_every=50000):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output == manifest or (output.exists() and output.samefile(manifest)):
        raise ValueError('Unique-image output must not replace the original relationship manifest')
    if output.suffix.lower() != '.csv':
        raise ValueError('Unique-image output must end in .csv')
    started = time.monotonic()
    source_hash = file_sha256(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    with tempfile.TemporaryDirectory(prefix='abo-unique-', dir=output.parent) as temporary:
        db = sqlite3.connect(str(Path(temporary) / 'lookup.sqlite3'))
        try:
            db.execute('CREATE TABLE images (id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL)')
            db.execute('CREATE TABLE products (id TEXT PRIMARY KEY)')
            db.execute('CREATE TABLE image_products (image_id TEXT, product_id TEXT, PRIMARY KEY(image_id, product_id))')
            for records, row in enumerate(manifest_rows(manifest), 1):
                native_id = row.get('abo_image_id')
                if not native_id:
                    raise ValueError(f'Missing abo_image_id at manifest record {records}; cannot assign a stable vector ID')
                valid_id(native_id)
                path = str(image_path(manifest, row))
                inserted = db.execute('INSERT OR IGNORE INTO images VALUES (?, ?)', (native_id, path)).rowcount
                if not inserted:
                    previous = db.execute('SELECT path FROM images WHERE id=?', (native_id,)).fetchone()
                    if previous is not None and previous[0] != path:
                        raise ValueError(f'ABO image ID {native_id!r} identifies multiple distinct image paths: '
                                         f'{previous[0]!r} and {path!r}')
                    previous = db.execute('SELECT id FROM images WHERE path=?', (path,)).fetchone()
                    if previous is not None and previous[0] != native_id:
                        raise ValueError(f'Image path {path!r} has conflicting ABO image IDs: '
                                         f'{previous[0]!r} and {native_id!r}')
                product = row['product_id']
                if product:
                    db.execute('INSERT OR IGNORE INTO products VALUES (?)', (product,))
                    db.execute('INSERT OR IGNORE INTO image_products VALUES (?, ?)', (native_id, product))
                if records % progress_every == 0:
                    db.commit()
                    print(f'Checked {records:,} relationship records ({time.monotonic() - started:.1f}s)', file=sys.stderr)
            if not records:
                raise ValueError('Relationship manifest is empty')
            if file_sha256(manifest) != source_hash:
                raise ValueError('Relationship manifest changed during deduplication; output was not replaced')
            count = db.execute('SELECT COUNT(*) FROM images').fetchone()[0]
            summary = dict(source_records=records, unique_images=count, duplicate_records=records - count,
                           unique_products=db.execute('SELECT COUNT(*) FROM products').fetchone()[0],
                           images_with_multiple_products=db.execute(
                               'SELECT COUNT(*) FROM (SELECT image_id FROM image_products GROUP BY image_id HAVING COUNT(*) > 1)'
                           ).fetchone()[0])
            with atomic_text(output) as stream:
                writer = csv.DictWriter(stream, fieldnames=UNIQUE_FIELDS)
                writer.writeheader()
                # Sort by native ID rather than first occurrence: stable even if input rows are reordered.
                for native_id, path in db.execute('SELECT id, path FROM images ORDER BY id'):
                    writer.writerow(dict(image_id=native_id, abo_image_id=native_id,
                                         image_path=os.path.relpath(path, output.parent)))
        finally:
            db.close()
    write_json(str(output) + '.meta.json', dict(
        **summary, source_manifest=str(manifest), source_sha256=source_hash,
        output_sha256=file_sha256(output), id_scheme='native_abo_image_id',
    ))
    print('Unique-image summary: ' + ' '.join(f'{key}={value}' for key, value in summary.items()) +
          f' elapsed_seconds={time.monotonic() - started:.2f} output={output}', file=sys.stderr)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=PROJECT_ROOT / 'data/abo/manifest-full.csv')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/abo/unique-images.csv')
    args = parser.parse_args()
    try:
        deduplicate_manifest(args.manifest, args.output)
    except (OSError, ValueError, csv.Error, sqlite3.Error) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
