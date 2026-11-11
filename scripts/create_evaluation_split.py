#!/usr/bin/env python3
"""Create a deterministic, image-ID-disjoint split without rewriting vector values."""
import argparse
from contextlib import closing, contextmanager, ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sqlite3
import sys
import tempfile
import time

from abo_common import PROJECT_ROOT, valid_id
from clip_embeddings import DIMENSIONS

NUMBER = re.compile(rb'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?')
FLOAT32_MAX = 3.4028234663852886e38


def validate_record(raw, line_number):
    """Validate one headerless row; return only its ID, never converted coordinates."""
    fields = raw.rstrip(b'\r\n').split(b',')
    if len(fields) != DIMENSIONS + 1:
        raise ValueError(f'Line {line_number}: expected an image ID and {DIMENSIONS} coordinates; '
                         f'found {len(fields) - 1} coordinates')
    identifier = valid_id(fields[0].decode('utf-8'))
    if '"' in identifier:
        raise ValueError(f'Line {line_number}: quoted IDs are not supported by the C++ CSV format')
    for coordinate, field in enumerate(fields[1:], 1):
        token = field.strip(b' \t\r')
        if not NUMBER.fullmatch(token):
            raise ValueError(f'Line {line_number}, coordinate {coordinate}: invalid numeric value')
        value = float(token)
        if not math.isfinite(value) or abs(value) > FLOAT32_MAX:
            raise ValueError(f'Line {line_number}, coordinate {coordinate}: value must be finite and within float32 range')
    return identifier


def destination_path(path):
    path = Path(path).expanduser().absolute()
    # Do not follow a final-component symlink and inadvertently write somewhere else.
    if os.path.lexists(path):
        raise ValueError(f'Refusing to overwrite existing output: {path}')
    return path.parent.resolve() / path.name


@contextmanager
def temporary_output(destination):
    fd, name = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    path = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            yield path, stream
    finally:
        path.unlink(missing_ok=True)


def publish_outputs(outputs):
    """Publish with exclusive hard links, so even a concurrent creator is never overwritten."""
    published = []
    try:
        for temporary, destination in outputs:
            os.link(temporary, destination)
            published.append((temporary, destination))
    except BaseException:
        # Roll back only files created by this invocation, preserving concurrent files.
        for temporary, destination in reversed(published):
            if destination.exists() and destination.samefile(temporary):
                destination.unlink()
        raise


def verify_output(path, partition, expected_count, expected_hash, db):
    digest = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        for count, raw in enumerate(stream, 1):
            digest.update(raw)
            identifier = raw.split(b',', 1)[0].decode('utf-8')
            updated = db.execute(
                'UPDATE ids SET written=1 WHERE image_id=? AND partition=? AND written=0',
                (identifier, partition),
            ).rowcount
            if updated != 1:
                raise ValueError(f'{partition} output has a duplicate, overlapping, or unexpected image ID: {identifier!r}')
            if count % 10000 == 0:
                db.commit()
    if count != expected_count:
        raise ValueError(f'{partition} output count mismatch: expected {expected_count}, found {count}')
    if digest.hexdigest() != expected_hash:
        raise ValueError(f'{partition} output bytes changed during verification')
    db.commit()


def create_split(input_path, train_output, queries_output, query_count=500, seed=42, progress_every=10000):
    """Validate, sample IDs, copy raw rows, and verify both outputs using bounded RAM.

    Only IDs/partition flags are stored in an on-disk SQLite database. Two input
    passes and one verification pass over the staged outputs are performed.
    """
    started = time.monotonic()
    if query_count < 1:
        raise ValueError('query-count must be positive')
    source = Path(input_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f'Input must be a regular file: {source}')
    train, queries = destination_path(train_output), destination_path(queries_output)
    metadata = destination_path(str(queries) + '.meta.json')
    if train.suffix.lower() != '.csv' or queries.suffix.lower() != '.csv':
        raise ValueError('Train and query outputs must end in .csv')
    if len({source, train, queries, metadata}) != 4:
        raise ValueError('Input, train, query, and metadata paths must be distinct')
    for path in (train, queries):
        path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='evaluation-split-', dir=train.parent) as workspace:
        with closing(sqlite3.connect(str(Path(workspace) / 'ids.sqlite3'))) as db:
            db.execute('CREATE TABLE ids (ordinal INTEGER PRIMARY KEY, image_id TEXT UNIQUE NOT NULL, '
                       "partition TEXT NOT NULL DEFAULT 'train', written INTEGER NOT NULL DEFAULT 0)")
            input_digest = hashlib.sha256()
            total = 0
            print(f'Validating input vectors and unique IDs: {source}', file=sys.stderr)
            with source.open('rb') as stream:
                for total, raw in enumerate(stream, 1):
                    identifier = validate_record(raw, total)
                    input_digest.update(raw)
                    try:
                        db.execute('INSERT INTO ids (ordinal, image_id) VALUES (?, ?)', (total - 1, identifier))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f'Line {total}: duplicate image ID {identifier!r}') from exc
                    if total % progress_every == 0:
                        db.commit()
                        print(f'Validated {total:,} vectors ({time.monotonic() - started:.1f}s)', file=sys.stderr)
            db.commit()
            if total <= query_count:
                raise ValueError(f'Need more than {query_count} input vectors to leave a nonempty index set; found {total}')
            selected = random.Random(seed).sample(range(total), query_count)
            db.executemany("UPDATE ids SET partition='query' WHERE ordinal=?", ((ordinal,) for ordinal in selected))
            query_ids = {row[0] for row in db.execute("SELECT image_id FROM ids WHERE partition='query'")}
            if len(query_ids) != query_count:
                raise ValueError('Selected query ID count does not match query-count')
            db.commit()
            expected = {'train': total - query_count, 'query': query_count}
            with ExitStack() as stack:
                train_temp, train_stream = stack.enter_context(temporary_output(train))
                query_temp, query_stream = stack.enter_context(temporary_output(queries))
                metadata_temp, metadata_stream = stack.enter_context(temporary_output(metadata))
                streams = {'train': train_stream, 'query': query_stream}
                hashes = {name: hashlib.sha256() for name in streams}
                counts = dict.fromkeys(streams, 0)
                copied_digest = hashlib.sha256()
                print(f'Copying original rows: train={expected["train"]:,} query={query_count:,} seed={seed}', file=sys.stderr)
                with source.open('rb') as stream:
                    for raw in stream:
                        copied_digest.update(raw)
                        identifier = raw.split(b',', 1)[0].decode('utf-8')
                        partition = 'query' if identifier in query_ids else 'train'
                        streams[partition].write(raw)
                        hashes[partition].update(raw)
                        counts[partition] += 1
                if copied_digest.digest() != input_digest.digest():
                    raise ValueError('Input changed between validation and copying; outputs were not published')
                if counts != expected:
                    raise ValueError(f'Split count mismatch: expected {expected}, found {counts}')
                for stream in streams.values():
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
                print('Verifying staged output bytes, counts, unique IDs, and separation', file=sys.stderr)
                verify_output(train_temp, 'train', expected['train'], hashes['train'].hexdigest(), db)
                verify_output(query_temp, 'query', query_count, hashes['query'].hexdigest(), db)
                if db.execute('SELECT COUNT(*) FROM ids WHERE written<>1').fetchone()[0]:
                    raise ValueError('Some source image IDs are missing from the split')
                report = dict(input_vectors=total, train_vectors=expected['train'], query_vectors=query_count,
                              overlap=0, dimensions=DIMENSIONS, seed=seed, input=str(source),
                              train_output=str(train), queries_output=str(queries),
                              input_sha256=input_digest.hexdigest(), train_sha256=hashes['train'].hexdigest(),
                              queries_sha256=hashes['query'].hexdigest())
                provenance = dict(**report, query_ids=sorted(query_ids), python_version=platform.python_version(),
                                  sampling='random.sample_without_replacement_v1', source_order_preserved=True)
                metadata_stream.write((json.dumps(provenance, indent=2, sort_keys=True) + '\n').encode('utf-8'))
                metadata_stream.flush()
                os.fsync(metadata_stream.fileno())
                metadata_stream.close()
                publish_outputs([(train_temp, train), (query_temp, queries), (metadata_temp, metadata)])
    print(f'Split completed in {time.monotonic() - started:.2f}s', file=sys.stderr)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=PROJECT_ROOT / 'data/abo/vectors-full.csv')
    parser.add_argument('--train-output', type=Path, default=PROJECT_ROOT / 'data/abo/vectors-eval-train.csv')
    parser.add_argument('--queries-output', type=Path, default=PROJECT_ROOT / 'data/abo/queries-eval.csv')
    parser.add_argument('--query-count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        report = create_split(args.input, args.train_output, args.queries_output, args.query_count, args.seed)
    except KeyboardInterrupt:
        parser.exit(130, '\nInterrupted; existing inputs/outputs were not overwritten.\n')
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(1, f'Error: {exc}\n')
    for key, value in report.items():
        print(f'{key}={value}')


if __name__ == '__main__':
    main()
