#!/usr/bin/env python3
"""Stream local ABO images into resumable, headerless 512D OpenCLIP vectors."""
import argparse
import csv
import json
import itertools
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from abo_common import (DEFAULT_MANIFEST, PROJECT_ROOT, atomic_text, file_sha256,
                        image_path, manifest_rows, write_json)
from clip_embeddings import (DIMENSIONS, MODEL_NAME, PIPELINE_VERSION, PRETRAINED,
                             ClipEncoder, ImageReadError, validate_embedding)


@contextmanager
def output_lock(output):
    # macOS/Linux: prevent concurrent processes from writing the same checkpoint/output.
    import fcntl
    with Path(str(output) + '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'Another embedding process is using {output}') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def image_stamp(path):
    stat = path.stat()
    return f'{stat.st_size}:{stat.st_mtime_ns}'


def vector_text(values):
    return ','.join(format(v, '.9g') for v in validate_embedding(values))


def embed_manifest(manifest, output, batch_size=16, limit=None, device='auto', encoder_factory=ClipEncoder):
    """Commit complete batches to SQLite; publish a validated CSV atomically at the end.

    A repeated command resumes automatically. limit selects the first N manifest rows,
    including unreadable rows. Failed images are retried on the next invocation.
    """
    started = time.monotonic()
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError('batch-size and limit must be positive')
    if output == manifest or output.suffix.lower() != '.csv':
        raise ValueError('Output must be a separate .csv file')
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(str(output) + '.checkpoint.sqlite3')
    config = dict(manifest_sha256=file_sha256(manifest), manifest=str(manifest),
                  model=MODEL_NAME, pretrained=PRETRAINED, dimensions=DIMENSIONS,
                  normalization='L2', dtype='float32', pipeline_version=PIPELINE_VERSION)
    summary = dict(selected=0, resumed=0, embedded=0, skipped=0, exported=0)
    with output_lock(output):
        if output.exists() and not checkpoint.exists():
            raise ValueError('Existing vectors have no checkpoint. Choose a new --output; '
                             'unverified vectors will not be appended or overwritten.')
        db = sqlite3.connect(checkpoint)
        try:
            db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS embeddings ('
                       'image_id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE NOT NULL, '
                       'stamp TEXT NOT NULL, vector TEXT NOT NULL)')
            existing = db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
            if existing is not None and json.loads(existing[0]) != config:
                raise ValueError('Checkpoint manifest/model settings changed. Choose a new --output.')
            if existing is None:
                if output.exists() or db.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0]:
                    raise ValueError('Checkpoint lacks provenance; choose a new --output')
                db.execute('INSERT INTO metadata VALUES (?, ?)', ('config', json.dumps(config, sort_keys=True)))
            db.execute('CREATE TEMP TABLE run_ids (image_id TEXT PRIMARY KEY)')
            db.commit()
            encoder = None
            batch, records = [], []

            def flush():
                if not records:
                    return
                vectors = encoder.encode_batch(batch)
                if len(vectors) != len(records):
                    raise ValueError('Model returned the wrong number of embeddings')
                entries = []
                for (identifier, ordinal, stamp, path), vector in zip(records, vectors):
                    if image_stamp(path) != stamp:
                        raise ValueError(f'Image changed during encoding: {path}')
                    entries.append((identifier, ordinal, stamp, vector_text(vector)))
                # One transaction: an interruption cannot leave half a batch in the checkpoint.
                with db:
                    db.executemany('INSERT INTO embeddings VALUES (?, ?, ?, ?)', entries)
                summary['embedded'] += len(entries)
                print(f"Selected={summary['selected']:,} embedded={summary['embedded']:,} "
                      f"resumed={summary['resumed']:,} skipped={summary['skipped']:,} "
                      f"elapsed_seconds={time.monotonic() - started:.1f}", file=sys.stderr)
                batch.clear()
                records.clear()

            for ordinal, row in enumerate(itertools.islice(manifest_rows(manifest), limit)):
                summary['selected'] += 1
                identifier = row['image_id']
                try:
                    db.execute('INSERT INTO run_ids VALUES (?)', (identifier,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f'Duplicate manifest image_id: {identifier}') from exc
                path = image_path(manifest, row)
                if path == output:
                    raise ValueError('Vector output must not overwrite a catalog image')
                existing = db.execute('SELECT ordinal, stamp FROM embeddings WHERE image_id=?', (identifier,)).fetchone()
                if existing is not None:
                    try:
                        unchanged = existing == (ordinal, image_stamp(path))
                    except OSError:
                        unchanged = False
                    if not unchanged:
                        raise ValueError(f'Previously embedded image changed or disappeared: {path}. Choose a new --output.')
                    summary['resumed'] += 1
                    if summary['resumed'] % 1000 == 0:
                        print(f"Resumed {summary['resumed']:,} cached vectors", file=sys.stderr)
                    continue
                if encoder is None:
                    print(f'Loading {MODEL_NAME}/{PRETRAINED} (device={device})', file=sys.stderr)
                    encoder = encoder_factory(device)
                    signature = json.dumps(encoder.signature, sort_keys=True)
                    stored = db.execute("SELECT value FROM metadata WHERE key='packages'").fetchone()
                    if stored is not None and stored[0] != signature:
                        raise ValueError('Embedding dependency versions changed. Restore the original environment '
                                         'or choose a new --output to avoid mixing preprocessing versions.')
                    db.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ('packages', signature))
                    db.commit()
                    print(f'Encoding on {encoder.device}', file=sys.stderr)
                try:
                    stamp = image_stamp(path)
                    tensor = encoder.prepare_image(path)
                except (OSError, ImageReadError) as exc:
                    summary['skipped'] += 1
                    print(f'Skip {identifier}: {exc}', file=sys.stderr)
                    continue
                batch.append(tensor)
                records.append((identifier, ordinal, stamp, path))
                if len(batch) >= batch_size:
                    flush()
            flush()
            if file_sha256(manifest) != config['manifest_sha256']:
                raise ValueError('Manifest changed during embedding; output was not replaced')
            count = db.execute('SELECT COUNT(*) FROM embeddings WHERE ordinal < ?', (summary['selected'],)).fetchone()[0]
            if count == 0:
                raise ValueError('No embeddings generated; check the manifest and image files')
            # Stream from disk in manifest order. Old output remains intact on any failure.
            with atomic_text(output) as stream:
                for identifier, text in db.execute('SELECT image_id, vector FROM embeddings WHERE ordinal < ? '
                                                   'ORDER BY ordinal', (summary['selected'],)):
                    validate_embedding([float(v) for v in text.split(',')])
                    stream.write(identifier + ',' + text + '\n')
                    summary['exported'] += 1
            packages = db.execute("SELECT value FROM metadata WHERE key='packages'").fetchone()
            write_json(str(output) + '.meta.json', dict(
                **config, packages=json.loads(packages[0]) if packages else {},
                rows=summary['exported'], vectors_sha256=file_sha256(output),
                limit=limit,
            ))
        finally:
            db.close()
    print('Embedding summary: ' + ' '.join(f'{k}={v}' for k, v in summary.items()) +
          f' elapsed_seconds={time.monotonic() - started:.2f} output={output}', file=sys.stderr)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/abo/vectors.csv')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int, help='Select first N manifest rows; omit for the entire manifest')
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    args = parser.parse_args()
    try:
        embed_manifest(args.manifest, args.output, args.batch_size, args.limit, args.device)
    except KeyboardInterrupt:
        parser.exit(130, '\nInterrupted. Completed batches are checkpointed; rerun the same command to resume.\n')
    except (OSError, ValueError, RuntimeError, sqlite3.Error, csv.Error) as exc:
        parser.exit(1, f'Error: {exc}\nCompleted batches remain in the checkpoint. '
                       'For device errors try --device cpu or a smaller --batch-size.\n')


if __name__ == '__main__':
    main()
