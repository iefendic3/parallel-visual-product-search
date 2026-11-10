#!/usr/bin/env python3
"""Join extracted ABO listings to local images; no downloads or image decoding."""
import argparse
import csv
import gzip
import html
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

from abo_common import DEFAULT_MANIFEST, MANIFEST_FIELDS, PROJECT_ROOT, atomic_text, valid_id


def localized_text(values):
    """Prefer en_US, then another English value, then the first available value."""
    if isinstance(values, str):
        return html.unescape(values.strip())
    if not isinstance(values, list):
        return ''
    candidates = [v for v in values if isinstance(v, dict) and isinstance(v.get('value'), str)]
    candidates = [v for v in candidates if v['value'].strip()]
    candidates.sort(key=lambda v: (0 if v.get('language_tag') == 'en_US' else
                                  1 if str(v.get('language_tag', '')).startswith('en') else 2))
    return html.unescape(candidates[0]['value'].strip()) if candidates else ''


def category_text(listing):
    nodes = listing.get('node') or []
    if not isinstance(nodes, list):
        nodes = []
    paths = [n.get('node_name') or n.get('path') for n in nodes if isinstance(n, dict)]
    paths = [p for p in paths if isinstance(p, str) and p]
    return ' | '.join(dict.fromkeys(paths)) or localized_text(listing.get('product_type'))


def listing_images(listing):
    main = listing.get('main_image_id')
    others = listing.get('other_image_id') or []
    if not isinstance(others, list):
        others = []
    return dict.fromkeys(v for v in [main, *others] if isinstance(v, str) and v)


def prepare_manifest(abo_root, output, limit=None, progress_every=1000):
    started = time.monotonic()
    root, output = Path(abo_root).resolve(), Path(output).resolve()
    images_root = root / 'images/small'
    metadata = root / 'images/metadata/images.csv.gz'
    shards = sorted((root / 'listings/metadata').glob('listings_*.json.gz'))
    if not metadata.is_file() or not images_root.is_dir() or not shards:
        raise ValueError(f'Expected images/metadata/images.csv.gz, images/small/, and '
                         f'listings/metadata/listings_*.json.gz below {root}')
    if any(output.is_relative_to(root / folder) for folder in ('images', 'listings', 'raw')):
        raise ValueError('Output must not overwrite files inside original ABO asset directories')
    if output.suffix.lower() != '.csv':
        raise ValueError('Manifest output must end in .csv')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(processed_listings=0, valid_images=0, skipped_images=0,
                   duplicate_references=0, skipped_listings=0, generated_rows=0)
    # Only metadata is indexed, on disk. No image bytes or full listing collection in RAM.
    with tempfile.TemporaryDirectory(prefix='abo-manifest-', dir=output.parent) as temporary:
        db = sqlite3.connect(str(Path(temporary) / 'lookup.sqlite3'))
        try:
            db.execute('CREATE TABLE images (id TEXT PRIMARY KEY, path TEXT NOT NULL)')
            db.execute('CREATE TABLE emitted (id TEXT PRIMARY KEY)')
            print(f'Indexing local image lookup: {metadata}', file=sys.stderr)
            with gzip.open(metadata, 'rt', newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                if not {'image_id', 'path'}.issubset(reader.fieldnames or []):
                    raise ValueError('Image metadata must contain image_id and path')
                for i, row in enumerate(reader, 1):
                    if not row.get('image_id') or not row.get('path'):
                        continue
                    db.execute('INSERT INTO images VALUES (?, ?)', (row['image_id'], row['path']))
                    if i % 100000 == 0:
                        print(f'Indexed {i:,} image paths', file=sys.stderr)
            db.commit()
            with atomic_text(output) as stream:
                writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                for shard in shards:
                    if limit is not None and summary['generated_rows'] >= limit:
                        break
                    print(f'Reading {shard.name}', file=sys.stderr)
                    with gzip.open(shard, 'rt', encoding='utf-8') as listings:
                        for line_number, line in enumerate(listings, 1):
                            if not line.strip():
                                continue
                            summary['processed_listings'] += 1
                            try:
                                listing = json.loads(line)
                            except json.JSONDecodeError:
                                summary['skipped_listings'] += 1
                                print(f'Skip malformed JSON: {shard.name}:{line_number}', file=sys.stderr)
                                continue
                            if not isinstance(listing, dict) or not isinstance(listing.get('item_id'), str) or not listing['item_id'].strip():
                                summary['skipped_listings'] += 1
                                continue
                            product = listing['item_id'].strip()
                            domain = listing.get('domain_name') or ''
                            if not isinstance(domain, str):
                                domain = ''
                            title = localized_text(listing.get('item_name')) or product
                            for native_id in listing_images(listing):
                                # Shared images occur across products/marketplaces in the real data.
                                # Preserve each association, with the native ABO ID in its own column.
                                identifier = valid_id(':'.join(quote(v, safe='') for v in (product, domain, native_id)))
                                found = db.execute('SELECT path FROM images WHERE id=?', (native_id,)).fetchone()
                                if found is None:
                                    summary['skipped_images'] += 1
                                    continue
                                path = (images_root / found[0]).resolve()
                                if not path.is_relative_to(images_root) or not path.is_file():
                                    summary['skipped_images'] += 1
                                    continue
                                inserted = db.execute('INSERT OR IGNORE INTO emitted VALUES (?)', (identifier,)).rowcount
                                if not inserted:
                                    summary['duplicate_references'] += 1
                                    continue
                                summary['valid_images'] += 1
                                writer.writerow(dict(
                                    image_id=identifier, product_id=product, title=title,
                                    image_path=os.path.relpath(path, output.parent),
                                    category=category_text(listing),
                                    source_url=f'https://m.media-amazon.com/image/I/{quote(native_id, safe="")}{path.suffix}',
                                    abo_image_id=native_id, domain_name=domain,
                                ))
                                summary['generated_rows'] += 1
                                if limit is not None and summary['generated_rows'] >= limit:
                                    break
                            if summary['processed_listings'] % progress_every == 0:
                                print(f"Listings={summary['processed_listings']:,} rows={summary['generated_rows']:,} "
                                      f"skipped_images={summary['skipped_images']:,}", file=sys.stderr)
                                db.commit()
                            if limit is not None and summary['generated_rows'] >= limit:
                                break
        finally:
            db.close()
    print('Manifest summary: ' + ' '.join(f'{key}={value}' for key, value in summary.items()) +
          f' elapsed_seconds={time.monotonic() - started:.2f} output={output}', file=sys.stderr)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--abo-root', type=Path, default=PROJECT_ROOT / 'data/abo')
    parser.add_argument('--output', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--limit', type=int, help='Maximum manifest rows; default: all listings')
    args = parser.parse_args()
    try:
        prepare_manifest(args.abo_root, args.output, args.limit)
    except (OSError, EOFError, ValueError, sqlite3.Error, csv.Error) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
