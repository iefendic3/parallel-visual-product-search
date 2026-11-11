"""Unique-image preparation, checkpoint reuse, and one-to-many product lookup."""
import csv
from contextlib import redirect_stderr
import io
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from abo_common import MANIFEST_FIELDS, file_sha256, image_path, manifest_rows
from deduplicate_abo_manifest import UNIQUE_FIELDS, deduplicate_manifest
from embed_abo import embed_manifest
from search_image import load_relationships, resolve_results, search_image
from test_abo_pipeline import FakeEncoder


class UniqueImageTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='abo-unique-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.full = self.root / 'manifest-full.csv'
        self.unique = self.root / 'unique-images.csv'
        self.output = self.root / 'vectors-full.csv'
        self.logs = io.StringIO()
        redirect = redirect_stderr(self.logs)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.rows = []
        for identifier, product, domain, native, filename in [
            ('listing-z2', 'P2', 'amazon.com', 'Z', 'z.jpg'),
            ('listing-z1-uk', 'P1', 'amazon.co.uk', 'Z', 'z.jpg'),
            ('listing-z1', 'P1', 'amazon.com', 'Z', 'z.jpg'),
            ('listing-a', 'P2', 'amazon.com', 'A+image', 'a.jpg'),
            ('listing-c', 'P3', 'amazon.com', 'C', 'c.jpg'),
        ]:
            path = self.root / filename
            if not path.exists():
                path.write_bytes(('image-' + native).encode())
            self.rows.append(dict(image_id=identifier, product_id=product, domain_name=domain,
                                  abo_image_id=native, image_path=filename, title='Title ' + product))
        self.write_full()

    def write_full(self):
        with self.full.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def prepare(self):
        return deduplicate_manifest(self.full, self.unique)

    def embed(self, **options):
        return embed_manifest(self.unique, self.output, encoder_factory=options.pop('encoder_factory', FakeEncoder), **options)

    def vectors(self, path=None):
        with (path or self.output).open(newline='') as stream:
            return list(csv.reader(stream))

    def test_one_row_per_path_native_ids_and_original_manifest_unchanged(self):
        original = file_sha256(self.full)
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('No image bytes should be read')):
            summary = self.prepare()
        self.assertEqual(summary, dict(source_records=5, unique_images=3, duplicate_records=2,
                                       unique_products=3, images_with_multiple_products=1))
        self.assertEqual(file_sha256(self.full), original)
        rows = list(manifest_rows(self.unique, require_products=False))
        self.assertEqual([r['image_id'] for r in rows], ['A+image', 'C', 'Z'])
        self.assertTrue(all(r['image_id'] == r['abo_image_id'] for r in rows))
        self.assertEqual(len({image_path(self.unique, r) for r in rows}), 3)
        self.assertTrue(all(not Path(r['image_path']).is_absolute() for r in rows))
        metadata = json.loads(Path(str(self.unique) + '.meta.json').read_text())
        self.assertEqual(metadata['source_sha256'], original)

    def test_determinism_after_reordering_and_normalized_path_aliases(self):
        self.rows[1]['image_path'] = './subdir/../z.jpg'
        (self.root / 'subdir').mkdir()
        self.write_full()
        self.prepare()
        original = self.unique.read_bytes()
        self.rows.reverse()
        self.write_full()
        self.prepare()
        self.assertEqual(self.unique.read_bytes(), original)
        elsewhere = self.root / 'nested/unique.csv'
        deduplicate_manifest(self.full, elsewhere)
        self.assertTrue(all(image_path(elsewhere, r).is_file() for r in manifest_rows(elsewhere, require_products=False)))

    def test_does_not_deduplicate_distinct_paths_with_identical_bytes(self):
        (self.root / 'c.jpg').write_bytes((self.root / 'a.jpg').read_bytes())
        self.assertEqual(self.prepare()['unique_images'], 3)

    def test_missing_native_id_fails_without_replacing_existing_output(self):
        self.unique.write_text('previous output')
        self.rows[-1]['abo_image_id'] = ''
        self.write_full()
        with self.assertRaisesRegex(ValueError, 'Missing abo_image_id'):
            self.prepare()
        self.assertEqual(self.unique.read_text(), 'previous output')

    def test_one_native_id_cannot_identify_two_paths(self):
        self.rows[-1]['abo_image_id'] = 'Z'
        self.write_full()
        with self.assertRaisesRegex(ValueError, 'multiple distinct image paths'):
            self.prepare()
        self.assertFalse(self.unique.exists())

    def test_one_path_cannot_have_conflicting_native_ids(self):
        self.rows[1]['abo_image_id'] = 'different'
        self.write_full()
        with self.assertRaisesRegex(ValueError, 'conflicting ABO image IDs'):
            self.prepare()

    def test_cannot_replace_source_manifest(self):
        original = file_sha256(self.full)
        with self.assertRaisesRegex(ValueError, 'original relationship manifest'):
            deduplicate_manifest(self.full, self.full)
        self.assertEqual(file_sha256(self.full), original)

    def test_unique_csv_embeddings_and_resume_without_extra_model_load(self):
        self.prepare()
        encoder = FakeEncoder()
        prepare = MagicMock(wraps=encoder.prepare_image)
        encoder.prepare_image = prepare
        summary = self.embed(batch_size=2, encoder_factory=lambda device: encoder)
        self.assertEqual(summary['embedded'], 3)
        self.assertEqual(prepare.call_count, 3)
        rows = self.vectors()
        self.assertEqual([r[0] for r in rows], ['A+image', 'C', 'Z'])
        for row in rows:
            self.assertEqual(len(row), 513)
            self.assertTrue(all(math.isfinite(float(v)) for v in row[1:]))
            self.assertAlmostEqual(sum(float(v) ** 2 for v in row[1:]), 1, places=6)
        before = self.output.read_bytes()
        factory = MagicMock(side_effect=AssertionError('No model load on completed resume'))
        self.assertEqual(self.embed(encoder_factory=factory)['resumed'], 3)
        self.assertEqual(self.output.read_bytes(), before)
        factory.assert_not_called()

    def test_interrupted_unique_embedding_resumes_without_duplicate_vectors(self):
        self.prepare()
        encoder = FakeEncoder()
        original = encoder.encode_batch
        def interrupt(images):
            if encoder.calls == 1:
                raise KeyboardInterrupt()
            return original(images)
        encoder.encode_batch = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.embed(batch_size=1, encoder_factory=lambda device: encoder)
        summary = self.embed(batch_size=1)
        self.assertEqual(summary['resumed'], 1)
        self.assertEqual(summary['embedded'], 2)
        self.assertEqual(len({row[0] for row in self.vectors()}), 3)

    def test_unique_manifest_with_duplicate_paths_is_refused(self):
        with self.unique.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=UNIQUE_FIELDS)
            writer.writeheader()
            writer.writerows([dict(image_id='Z', abo_image_id='Z', image_path='z.jpg'),
                              dict(image_id='other', abo_image_id='other', image_path='z.jpg')])
        with self.assertRaisesRegex(ValueError, 'Duplicate image path'):
            self.embed()
        self.assertFalse(self.output.exists())

    def legacy_checkpoint(self):
        self.prepare()
        output = self.root / 'legacy-vectors.csv'
        embed_manifest(self.full, output, batch_size=2, encoder_factory=FakeEncoder)
        return output, Path(str(output) + '.checkpoint.sqlite3')

    def test_reuse_legacy_embeddings_under_native_ids_without_reencoding(self):
        output, checkpoint = self.legacy_checkpoint()
        source_hash = file_sha256(checkpoint)
        factory = MagicMock(side_effect=AssertionError('Existing images must not be encoded again'))
        summary = self.embed(reuse_checkpoint=checkpoint, reuse_manifest=self.full, encoder_factory=factory)
        self.assertEqual(summary['reused'], 3)
        self.assertEqual(summary['embedded'], 0)
        factory.assert_not_called()
        original = {r[0]: r[1:] for r in self.vectors(output)}
        reused = {r[0]: r[1:] for r in self.vectors()}
        self.assertEqual(reused['Z'], original['listing-z2'])
        self.assertEqual(reused['A+image'], original['listing-a'])
        self.assertEqual(file_sha256(checkpoint), source_hash)
        self.assertEqual(self.embed(encoder_factory=factory)['resumed'], 3)

    def test_reuse_then_extend_only_encodes_missing_images(self):
        self.prepare()
        legacy = self.root / 'legacy-vectors.csv'
        embed_manifest(self.full, legacy, limit=1, encoder_factory=FakeEncoder)
        summary = self.embed(reuse_checkpoint=Path(str(legacy) + '.checkpoint.sqlite3'), reuse_manifest=self.full)
        self.assertEqual(summary['reused'], 1)
        self.assertEqual(summary['embedded'], 2)
        self.assertEqual(len(self.vectors()), 3)

    def test_reuse_rejects_changed_model_provenance_and_corrupt_vectors(self):
        _, checkpoint = self.legacy_checkpoint()
        with sqlite3.connect(checkpoint) as db:
            config = json.loads(db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()[0])
            config['model'] = 'different-model'
            db.execute("UPDATE metadata SET value=? WHERE key='config'", (json.dumps(config),))
        with self.assertRaisesRegex(ValueError, 'incompatible model'):
            self.embed(reuse_checkpoint=checkpoint, reuse_manifest=self.full)
        with sqlite3.connect(checkpoint) as db:
            config['model'] = 'ViT-B-32'
            db.execute("UPDATE metadata SET value=? WHERE key='config'", (json.dumps(config),))
            db.execute("UPDATE embeddings SET vector='nan'")
        with self.assertRaisesRegex(ValueError, '512'):
            self.embed(reuse_checkpoint=checkpoint, reuse_manifest=self.full)
        self.assertFalse(self.output.exists())

    def test_reuse_rejects_wrong_manifest_and_changed_source_image(self):
        _, checkpoint = self.legacy_checkpoint()
        with self.assertRaisesRegex(ValueError, 'both'):
            self.embed(reuse_checkpoint=checkpoint)
        (self.root / 'z.jpg').write_bytes(b'changed catalog image')
        with self.assertRaisesRegex(ValueError, 'Reuse image changed'):
            self.embed(reuse_checkpoint=checkpoint, reuse_manifest=self.full)
        self.rows[0]['title'] = 'Changed metadata'
        self.write_full()
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.embed(reuse_checkpoint=checkpoint, reuse_manifest=self.full)

    def test_native_image_maps_to_all_listings_and_distinct_products(self):
        neighbors = [dict(rank=1, image_id='Z', squared_l2=0.25, l2=0.5)]
        all_matches = resolve_results(neighbors, self.full, 10)
        self.assertEqual([m['product_id'] for m in all_matches], ['P1', 'P1', 'P2'])
        self.assertEqual({m['listing_image_id'] for m in all_matches}, {'listing-z1', 'listing-z1-uk', 'listing-z2'})
        distinct = resolve_results(neighbors, self.full, 10, deduplicate=True)
        self.assertEqual([m['product_id'] for m in distinct], ['P1', 'P2'])
        self.assertTrue(all(m['image_id'] == 'Z' and m['squared_l2'] == 0.25 and m['l2'] == 0.5 for m in distinct))
        self.rows.reverse()
        self.write_full()
        self.assertEqual(resolve_results(neighbors, self.full, 10), all_matches)
        legacy = resolve_results([dict(rank=1, image_id='listing-z2', squared_l2=0, l2=0)], self.full, 10)
        self.assertEqual([m['product_id'] for m in legacy], ['P2'])

    def test_search_detects_conflicting_native_id_paths(self):
        self.rows[-1]['abo_image_id'] = 'Z'
        self.write_full()
        with self.assertRaisesRegex(ValueError, 'multiple distinct image paths'):
            load_relationships([dict(image_id='Z')], self.full)

    def search_setup(self):
        index, cli = self.root / 'index.ivf', self.root / 'ann_cli'
        index.touch()
        cli.touch()
        return index, cli

    def test_json_preserves_all_associations_even_with_one_ranked_match(self):
        index, cli = self.search_setup()
        runner = MagicMock(return_value=SimpleNamespace(stdout='rank,id,squared_l2,l2\n1,Z,0,0\n', stderr=''))
        result = search_image(self.root / 'z.jpg', index, self.full, cli, topk=1, encoder_factory=FakeEncoder, runner=runner)
        self.assertEqual(result['returned'], 1)
        self.assertEqual(len(result['image_matches'][0]['listings']), 3)
        json.dumps(result)

    def test_adaptive_candidate_growth_until_distinct_products_found(self):
        self.rows = []
        for i, product in enumerate(['P1', 'P1', 'P2', 'P3']):
            filename = f'candidate-{i}.jpg'
            (self.root / filename).write_bytes(str(i).encode())
            self.rows.append(dict(image_id=f'listing-{i}', abo_image_id=f'image-{i}',
                                  product_id=product, image_path=filename))
        self.write_full()
        index, cli = self.search_setup()
        def run(command, **kwargs):
            count = int(command[command.index('--topk') + 1])
            lines = ['rank,id,squared_l2,l2'] + [f'{i+1},image-{i},{i},{math.sqrt(i)}' for i in range(min(count, 4))]
            return SimpleNamespace(stdout='\n'.join(lines) + '\n', stderr='search_us=1')
        runner = MagicMock(side_effect=run)
        factory = MagicMock(return_value=FakeEncoder())
        result = search_image(self.root / 'candidate-0.jpg', index, self.full, cli, topk=3, nprobe=1,
                              deduplicate=True, candidate_multiplier=1, max_candidates=4,
                              encoder_factory=factory, runner=runner)
        self.assertEqual([int(c.args[0][c.args[0].index('--topk') + 1]) for c in runner.call_args_list], [3, 4])
        self.assertEqual(result['search_attempts'], 2)
        self.assertEqual([m['product_id'] for m in result['matches']], ['P1', 'P2', 'P3'])
        factory.assert_called_once()
        self.assertEqual(factory.return_value.calls, 1)
        self.assertEqual(runner.call_args_list[0].args[0][5], runner.call_args_list[1].args[0][5])

    def test_candidate_exhaustion_and_limit_stop_retries(self):
        index, cli = self.search_setup()
        runner = MagicMock(return_value=SimpleNamespace(stdout='rank,id,squared_l2,l2\n1,A+image,0,0\n', stderr=''))
        result = search_image(self.root / 'z.jpg', index, self.full, cli, topk=3,
                              deduplicate=True, candidate_multiplier=1, encoder_factory=FakeEncoder, runner=runner)
        self.assertEqual(result['returned'], 1)
        self.assertEqual(result['search_attempts'], 1)
        runner.return_value.stdout = 'rank,id,squared_l2,l2\n1,Z,0,0\n2,A+image,1,1\n'
        result = search_image(self.root / 'z.jpg', index, self.full, cli, topk=3, deduplicate=True,
                              candidate_multiplier=1, max_candidates=3, encoder_factory=FakeEncoder, runner=runner)
        self.assertEqual(result['returned'], 2)
        self.assertEqual(result['search_attempts'], 1)
        self.rows[-1]['product_id'] = 'P2'
        self.write_full()
        runner.return_value.stdout += '3,C,2,1.414213562\n'
        result = search_image(self.root / 'z.jpg', index, self.full, cli, topk=3, deduplicate=True,
                              candidate_multiplier=1, max_candidates=3, encoder_factory=FakeEncoder, runner=runner)
        self.assertEqual(result['returned'], 2)
        self.assertEqual(result['search_attempts'], 1)
        self.assertTrue(result['candidate_limit_reached'])
        with self.assertRaisesRegex(ValueError, 'at least topk'):
            search_image(self.root / 'z.jpg', index, self.full, cli, topk=3, max_candidates=2)

    @unittest.skipUnless(os.environ.get('ANN_CLI') or shutil.which('clang++') or shutil.which('c++'), 'C++ compiler unavailable')
    def test_native_vector_ids_work_through_real_cpp_engine(self):
        self.prepare()
        self.embed()
        cli = Path(os.environ['ANN_CLI']).resolve() if os.environ.get('ANN_CLI') else self.root / 'compiled_ann_cli'
        if not os.environ.get('ANN_CLI'):
            subprocess.run([shutil.which('clang++') or shutil.which('c++'), '-std=c++17', '-O2',
                            '-I' + str(ROOT / 'include'), str(ROOT / 'src/ivf_flat.cpp'),
                            str(ROOT / 'src/main.cpp'), '-o', str(cli)], check=True, capture_output=True)
        index = self.root / 'real.ivf'
        subprocess.run([str(cli), 'build', '--input', str(self.output), '--output', str(index),
                        '--clusters', '2', '--train-size', '0'], check=True, capture_output=True)
        result = search_image(self.root / 'z.jpg', index, self.full, cli, topk=3, nprobe=2,
                              deduplicate=True, encoder_factory=FakeEncoder)
        self.assertEqual(result['returned'], 3)
        self.assertEqual([m['product_id'] for m in result['matches'][:2]], ['P1', 'P2'])
        self.assertTrue(all(m['squared_l2'] == 0 for m in result['matches'][:2]))


if __name__ == '__main__':
    unittest.main()
