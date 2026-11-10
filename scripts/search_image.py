#!/usr/bin/env python3
"""Embed a local query image and resolve custom C++ IVF-Flat results to ABO metadata."""
import argparse
import csv
import io
import json
import math
import subprocess
import sys
from pathlib import Path

from abo_common import DEFAULT_MANIFEST, PROJECT_ROOT, image_path, manifest_rows, valid_id
from clip_embeddings import MODEL_NAME, PRETRAINED, ClipEncoder, validate_embedding


def run_ann(cli, index, query, topk, nprobe, runner=subprocess.run):
    query = validate_embedding(query)
    command = [str(Path(cli).expanduser().resolve()), 'search', '--index', str(Path(index).expanduser().resolve()),
               '--query', ','.join(format(x, '.9g') for x in query),
               '--topk', str(topk), '--nprobe', str(nprobe)]
    try:
        result = runner(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ValueError(f'ANN executable not found: {cli}. Build build/ann_cli first.') from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f'ANN search failed: {(exc.stderr or str(exc)).strip()}') from exc
    reader = csv.DictReader(io.StringIO(result.stdout))
    if reader.fieldnames != ['rank', 'id', 'squared_l2', 'l2']:
        raise ValueError('Unexpected ANN output; expected rank,id,squared_l2,l2')
    neighbors = []
    for row in reader:
        try:
            rank = int(row['rank'])
            squared_l2, l2 = float(row['squared_l2']), float(row['l2'])
            if rank != len(neighbors) + 1 or not all(math.isfinite(v) and v >= 0 for v in (squared_l2, l2)):
                raise ValueError('Invalid rank or distance')
            neighbors.append(dict(rank=rank, image_id=valid_id(row['id']), squared_l2=squared_l2, l2=l2))
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError(f'Malformed ANN result: {row}') from exc
    return neighbors, result.stderr.strip()


def resolve_results(neighbors, manifest, topk, deduplicate=False):
    wanted = {n['image_id'] for n in neighbors}
    metadata = {}
    # Keep only metadata needed for this query, rather than loading the full manifest.
    for row in manifest_rows(manifest):
        identifier = row['image_id']
        if identifier in wanted:
            if identifier in metadata:
                raise ValueError(f'Duplicate manifest image_id: {identifier}')
            metadata[identifier] = row
    results, products = [], set()
    for neighbor in neighbors:
        item = metadata.get(neighbor['image_id'])
        if item is None:
            raise ValueError(f"ANN image_id {neighbor['image_id']!r} is not in the manifest; check index/manifest pairing")
        product = item.get('product_id') or neighbor['image_id']
        if deduplicate and product in products:
            continue
        products.add(product)
        result = dict(item, **neighbor)
        result['candidate_rank'] = neighbor['rank']
        result['rank'] = len(results) + 1
        result['image_path'] = str(image_path(manifest, item))
        results.append(result)
        if len(results) == topk:
            break
    return results


def search_image(image, index, manifest, cli, topk=10, nprobe=4, device='auto',
                 deduplicate=False, candidate_multiplier=5, encoder_factory=ClipEncoder,
                 runner=subprocess.run):
    if min(topk, nprobe, candidate_multiplier) < 1:
        raise ValueError('topk, nprobe, and candidate-multiplier must be positive')
    image, index, manifest, cli = [Path(p).expanduser().resolve() for p in (image, index, manifest, cli)]
    for path in (image, index, manifest, cli):
        if not path.is_file():
            raise ValueError(f'File not found: {path}')
    encoder = encoder_factory(device)
    query = encoder.encode_batch([encoder.prepare_image(image)])[0]
    count = topk * candidate_multiplier if deduplicate else topk
    neighbors, diagnostics = run_ann(cli, index, query, count, nprobe, runner)
    results = resolve_results(neighbors, manifest, topk, deduplicate)
    return dict(model=MODEL_NAME, pretrained=PRETRAINED, device=encoder.device,
                topk=topk, nprobe=nprobe, deduplicated=deduplicate,
                candidates_requested=count, candidates_returned=len(neighbors),
                returned=len(results), diagnostics=diagnostics, matches=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--index', type=Path, default=PROJECT_ROOT / 'build/abo.ivf')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--cli', type=Path, default=PROJECT_ROOT / 'build/ann_cli')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--nprobe', type=int, default=4)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--deduplicate-products', action='store_true', help='Best-effort unique product_id results')
    parser.add_argument('--candidate-multiplier', type=int, default=5, help='Candidates per requested product when deduplicating')
    parser.add_argument('--format', choices=('text', 'json'), default='text')
    args = parser.parse_args()
    try:
        result = search_image(args.image, args.index, args.manifest, args.cli, args.topk, args.nprobe,
                              args.device, args.deduplicate_products, args.candidate_multiplier)
    except (OSError, ValueError, RuntimeError, csv.Error) as exc:
        parser.exit(1, f'Error: {exc}\n')
    if args.format == 'json':
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for match in result['matches']:
            print(f"#{match['rank']} {match.get('title') or match['image_id']}\n"
                  f"   image_id={match['image_id']} product_id={match.get('product_id', '')}\n"
                  f"   squared_l2={match['squared_l2']:.9g} l2={match['l2']:.9g}\n"
                  f"   image={match['image_path']}")
    if result['diagnostics']:
        print(result['diagnostics'], file=sys.stderr)
    if args.deduplicate_products:
        print(f"Best-effort deduplication: {result['returned']}/{args.topk} unique products from "
              f"{result['candidates_returned']} candidates. Increase --candidate-multiplier or --nprobe "
              'if more matches are needed.', file=sys.stderr)


if __name__ == '__main__':
    main()
