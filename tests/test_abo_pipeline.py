"""Local fixtures + mocked inference: no ABO downloads or pretrained weights."""
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from abo_common import MANIFEST_FIELDS, atomic_text, image_path, manifest_rows, valid_id
from clip_embeddings import (ClipEncoder, DIMENSIONS, ImageReadError, device_for_run,
                             normalize_embedding, validate_embedding)
from embed_abo import embed_manifest, output_lock
from prepare_abo_manifest import localized_text, prepare_manifest
from search_image import resolve_results, run_ann, search_image


class FakeEncoder:
    """Deterministic unit vectors from fixture bytes, never a pretrained model."""
    device = 'cpu'
    signature = {'test_encoder': '1'}

    def __init__(self, device='auto'):
        self.calls = 0

    def prepare_image(self, path):
        data = Path(path).read_bytes()
        if data == b'corrupt':
            raise ImageReadError('invalid image fixture')
        return data

    def encode_batch(self, images):
        self.calls += 1
        vectors = []
        for data in images:
            digest = hashlib.sha256(data).digest()
            vectors.append(normalize_embedding([float(digest[i % len(digest)] + 1) for i in range(DIMENSIONS)]))
        return vectors


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='abo-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.abo = self.root / 'abo'
        self.manifest = self.root / 'manifest.csv'
        self.output = self.root / 'vectors.csv'
        self.logs = io.StringIO()
        self.quiet = redirect_stderr(self.logs)
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def write_manifest(self, count=5, corrupt=None):
        rows = []
        for i in range(count):
            path = self.root / f'image {i}.jpg'
            path.write_bytes(b'corrupt' if i == corrupt else f'fixture-image-{i}'.encode())
            rows.append(dict(image_id=f'image_{i}', product_id=f'product_{i // 2}',
                             title=f'Product {i // 2}, multilingual č', image_path=path.name))
        self.write_rows(rows)
        return rows

    def write_rows(self, rows):
        with self.manifest.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def embed(self, **kwargs):
        return embed_manifest(self.manifest, self.output, encoder_factory=kwargs.pop('encoder_factory', FakeEncoder), **kwargs)

    def vectors(self):
        with self.output.open(newline='') as stream:
            return list(csv.reader(stream))

    def fixture_dataset(self):
        images = self.abo / 'images/small'
        metadata = self.abo / 'images/metadata'
        listings = self.abo / 'listings/metadata'
        for path in (images / 'aa', metadata, listings):
            path.mkdir(parents=True)
        for name in ('one.jpg', 'two.jpg'):
            (images / 'aa' / name).write_bytes(b'fixture-only')
        with gzip.open(metadata / 'images.csv.gz', 'wt', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_id', 'height', 'width', 'path'])
            writer.writerows([['a+id', 100, 100, 'aa/one.jpg'], ['b', 100, 100, 'aa/two.jpg'],
                              ['missing', 100, 100, 'aa/missing.jpg'], ['unsafe', 1, 1, '../../outside.jpg']])
        products = [
            dict(item_id='P1', domain_name='amazon.com',
                 item_name=[{'language_tag': 'de_DE', 'value': 'Deutsch'},
                            {'language_tag': 'en_US', 'value': 'Shoe &amp; boot, pair'}],
                 main_image_id='a+id', other_image_id=['a+id', 'b', 'missing', 'not-in-lookup', 'unsafe'],
                 product_type=[{'value': 'SHOES'}], node=[{'node_name': '/Shoes/Boots'}]),
            dict(item_id='P2', main_image_id='a+id'),
            dict(item_id='P1', domain_name='amazon.co.uk', main_image_id='a+id'),
            dict(item_id='P1', domain_name='amazon.com', main_image_id='a+id'),
            dict(main_image_id='b'),
        ]
        with gzip.open(listings / 'listings_0.json.gz', 'wt') as stream:
            for product in products:
                stream.write(json.dumps(product) + '\n')
            stream.write('{invalid-json\n')
        return products

    def test_manifest_join_optional_fields_shared_images_and_summary(self):
        self.fixture_dataset()
        summary = prepare_manifest(self.abo, self.manifest)
        rows = list(manifest_rows(self.manifest))
        self.assertEqual(len(rows), 4)
        self.assertEqual(summary['processed_listings'], 6)
        self.assertEqual(summary['skipped_images'], 3)
        self.assertEqual(summary['skipped_listings'], 2)
        self.assertEqual(summary['duplicate_references'], 1)
        self.assertEqual(rows[0]['title'], 'Shoe & boot, pair')
        self.assertEqual(rows[0]['category'], '/Shoes/Boots')
        self.assertEqual(rows[0]['image_id'], 'P1:amazon.com:a%2Bid')
        self.assertEqual(rows[0]['source_url'], 'https://m.media-amazon.com/image/I/a%2Bid.jpg')
        self.assertEqual(rows[2]['title'], 'P2')
        self.assertEqual(rows[2]['category'], '')
        self.assertEqual([r['product_id'] for r in rows], ['P1', 'P1', 'P2', 'P1'])
        self.assertTrue(all(image_path(self.manifest, r).is_file() for r in rows))
        self.assertEqual(len({r['image_id'] for r in rows}), 4)

    def test_manifest_determinism_limit_and_relative_output(self):
        self.fixture_dataset()
        prepare_manifest(self.abo, self.manifest)
        full = self.manifest.read_bytes()
        prepare_manifest(self.abo, self.manifest)
        self.assertEqual(full, self.manifest.read_bytes())
        limited = self.root / 'elsewhere/subset.csv'
        prepare_manifest(self.abo, limited, limit=1)
        row = next(manifest_rows(limited))
        self.assertEqual(row['image_id'], next(manifest_rows(self.manifest))['image_id'])
        self.assertFalse(Path(row['image_path']).is_absolute())
        self.assertTrue(image_path(limited, row).is_file())

    def test_manifest_does_not_overwrite_original_assets(self):
        self.fixture_dataset()
        with self.assertRaisesRegex(ValueError, 'original ABO'):
            prepare_manifest(self.abo, self.abo / 'images/small/aa/one.jpg')

    def test_localization_and_category_fallback(self):
        self.assertEqual(localized_text([{'value': 'default'}, {'language_tag': 'en_GB', 'value': 'English'}]), 'English')
        self.assertEqual(localized_text(None), '')
        self.assertEqual(localized_text(42), '')
        self.fixture_dataset()
        path = self.abo / 'listings/metadata/listings_0.json.gz'
        with gzip.open(path, 'wt') as stream:
            stream.write(json.dumps(dict(item_id='X', main_image_id='b', product_type=[{'value': 'HOME'}])) + '\n')
        prepare_manifest(self.abo, self.manifest)
        self.assertEqual(next(manifest_rows(self.manifest))['category'], 'HOME')

    def test_csv_header_and_ids_are_required(self):
        self.manifest.write_text('id,image_path\nx,test.jpg\n')
        with self.assertRaisesRegex(ValueError, 'columns'):
            list(manifest_rows(self.manifest))
        for identifier in ('', ' leading', 'bad,id', 'bad\nline', '#comment'):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                valid_id(identifier)

    def test_atomic_file_preserves_previous_output_on_failure(self):
        self.output.write_text('old output')
        with self.assertRaises(RuntimeError):
            with atomic_text(self.output) as stream:
                stream.write('partial output')
                raise RuntimeError('interrupted write')
        self.assertEqual(self.output.read_text(), 'old output')
        self.assertFalse(list(self.root.glob('.*.tmp')))

    def test_embedding_csv_float32_normalized_and_corrupt_image_skipped(self):
        self.write_manifest(corrupt=2)
        encoder = FakeEncoder()
        factory = MagicMock(return_value=encoder)
        summary = self.embed(batch_size=2, encoder_factory=factory)
        vectors = self.vectors()
        self.assertEqual(summary['skipped'], 1)
        self.assertEqual(summary['exported'], 4)
        self.assertEqual([r[0] for r in vectors], ['image_0', 'image_1', 'image_3', 'image_4'])
        self.assertTrue(all(len(r) == DIMENSIONS + 1 for r in vectors))
        factory.assert_called_once_with('auto')
        for row in vectors:
            self.assertAlmostEqual(math.sqrt(sum(float(v) ** 2 for v in row[1:])), 1.0, places=6)
        metadata = json.loads(Path(str(self.output) + '.meta.json').read_text())
        self.assertEqual(metadata['dimensions'], 512)
        self.assertEqual(metadata['model'], 'ViT-B-32')
        self.assertEqual(metadata['rows'], 4)
        self.assertEqual(metadata['vectors_sha256'], hashlib.sha256(self.output.read_bytes()).hexdigest())

    def test_normalization_rejects_invalid_embeddings_and_casts_float32(self):
        values = normalize_embedding([float(i + 1) for i in range(DIMENSIONS)])
        self.assertEqual(len(values), DIMENSIONS)
        self.assertEqual(values[0], struct.unpack('<f', struct.pack('<f', values[0]))[0])
        for vector in ([0.0] * DIMENSIONS, [1.0] * 3, [float('nan')] * DIMENSIONS, [float('inf')] * DIMENSIONS):
            with self.subTest(first=vector[0]), self.assertRaises(ValueError):
                normalize_embedding(vector)

    def test_resume_after_interruption_no_duplicates_and_no_model_for_completed_run(self):
        self.write_manifest()
        encoder = FakeEncoder()
        encode = encoder.encode_batch
        def interrupt(images):
            if encoder.calls == 1:
                raise KeyboardInterrupt()
            return encode(images)
        encoder.encode_batch = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.embed(batch_size=2, encoder_factory=lambda device: encoder)
        self.assertFalse(self.output.exists())
        with sqlite3.connect(str(self.output) + '.checkpoint.sqlite3') as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0], 2)
        summary = self.embed(batch_size=2)
        self.assertEqual(summary['resumed'], 2)
        self.assertEqual(summary['embedded'], 3)
        self.assertEqual(len({r[0] for r in self.vectors()}), 5)
        before = self.output.read_bytes()
        factory = MagicMock(side_effect=AssertionError('model should not load'))
        self.embed(encoder_factory=factory)
        factory.assert_not_called()
        self.assertEqual(before, self.output.read_bytes())

    def test_resume_can_extend_limit_and_retry_failed_images(self):
        self.write_manifest(corrupt=1)
        self.embed(limit=2)
        self.assertEqual(len(self.vectors()), 1)
        (self.root / 'image 1.jpg').write_bytes(b'repaired')
        summary = self.embed(limit=5)
        self.assertEqual(summary['resumed'], 1)
        self.assertEqual(len(self.vectors()), 5)
        self.assertEqual([r[0] for r in self.vectors()], [f'image_{i}' for i in range(5)])

    def test_resume_refuses_changed_manifest_or_already_embedded_image(self):
        self.write_manifest()
        self.embed(limit=2)
        (self.root / 'image 0.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed or disappeared'):
            self.embed()
        self.manifest.write_text(self.manifest.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            self.embed()

    def test_resume_refuses_changed_dependencies(self):
        self.write_manifest()
        self.embed(limit=1)
        encoder = FakeEncoder()
        encoder.signature = {'test_encoder': '2'}
        with self.assertRaisesRegex(ValueError, 'dependency versions changed'):
            self.embed(limit=2, encoder_factory=lambda device: encoder)

    def test_refuse_unverified_existing_output_and_duplicate_manifest_ids(self):
        rows = self.write_manifest(count=2)
        self.output.write_text('unverified')
        with self.assertRaisesRegex(ValueError, 'no checkpoint'):
            self.embed()
        self.assertEqual(self.output.read_text(), 'unverified')
        self.output.unlink()
        rows[1]['image_id'] = rows[0]['image_id']
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, 'Duplicate manifest'):
            self.embed()
        self.assertFalse(self.output.exists())

    def test_invalid_model_batch_is_not_checkpointed_or_published(self):
        self.write_manifest(count=2)
        encoder = FakeEncoder()
        encoder.encode_batch = lambda images: [[float('nan')] * DIMENSIONS] * len(images)
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.embed(encoder_factory=lambda device: encoder)
        with sqlite3.connect(str(self.output) + '.checkpoint.sqlite3') as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0], 0)
        self.assertFalse(self.output.exists())

    def test_failed_extension_preserves_published_csv_and_rejects_wrong_dimensions(self):
        self.write_manifest()
        self.embed(limit=2)
        before = self.output.read_bytes()
        encoder = FakeEncoder()
        encoder.encode_batch = lambda images: [[1.0, 2.0] for _ in images]
        with self.assertRaisesRegex(ValueError, '512'):
            self.embed(limit=5, encoder_factory=lambda device: encoder)
        self.assertEqual(self.output.read_bytes(), before)
        summary = self.embed(limit=5)
        self.assertEqual(summary['resumed'], 2)
        self.assertEqual(summary['exported'], 5)

    def test_limit_does_not_parse_unselected_manifest_record(self):
        self.write_manifest(count=1)
        with self.manifest.open('a') as stream:
            stream.write('invalid-manifest-row\n')
        self.embed(limit=1)
        self.assertEqual(len(self.vectors()), 1)

    def test_checkpoint_vectors_are_revalidated_before_publication(self):
        self.write_manifest(count=1)
        self.embed()
        before = self.output.read_bytes()
        with sqlite3.connect(str(self.output) + '.checkpoint.sqlite3') as db:
            db.execute("UPDATE embeddings SET vector='nan'")
        with self.assertRaisesRegex(ValueError, '512'):
            self.embed()
        self.assertEqual(self.output.read_bytes(), before)

    def test_single_writer_lock(self):
        self.write_manifest()
        with output_lock(self.output), self.assertRaisesRegex(ValueError, 'Another embedding process'):
            self.embed()

    def test_result_resolution_product_deduplication_and_unknown_id(self):
        self.write_manifest()
        neighbors = [dict(rank=i + 1, image_id=f'image_{i}', squared_l2=float(i), l2=math.sqrt(i)) for i in range(5)]
        matches = resolve_results(neighbors, self.manifest, 3, deduplicate=True)
        self.assertEqual([m['product_id'] for m in matches], ['product_0', 'product_1', 'product_2'])
        self.assertEqual([m['candidate_rank'] for m in matches], [1, 3, 5])
        self.assertEqual([m['rank'] for m in matches], [1, 2, 3])
        self.assertTrue(Path(matches[0]['image_path']).is_absolute())
        with self.assertRaisesRegex(ValueError, 'not in the manifest'):
            resolve_results([dict(rank=1, image_id='unknown')], self.manifest, 1)

    def test_cli_arguments_response_contract_and_actionable_errors(self):
        unit = normalize_embedding([1.0] * DIMENSIONS)
        runner = MagicMock(return_value=SimpleNamespace(stdout='rank,id,squared_l2,l2\n1,image_0,0.25,0.5\n', stderr='search_us=2'))
        rows, _ = run_ann(self.root / 'ann cli', self.root / 'index;literal.ivf', unit, 7, 2, runner)
        self.assertEqual(rows[0]['image_id'], 'image_0')
        command = runner.call_args.args[0]
        self.assertIsInstance(command, list)
        self.assertIn(str((self.root / 'index;literal.ivf').resolve()), command)
        self.assertNotIn('shell', runner.call_args.kwargs)
        self.assertEqual(command[-4:], ['--topk', '7', '--nprobe', '2'])
        runner.return_value.stdout = 'unexpected\n'
        with self.assertRaisesRegex(ValueError, 'Unexpected ANN output'):
            run_ann('cli', 'index', unit, 1, 1, runner)
        runner.side_effect = subprocess.CalledProcessError(1, ['cli'], stderr='Query dimensionality mismatch')
        with self.assertRaisesRegex(ValueError, 'dimensionality mismatch'):
            run_ann('cli', 'index', unit, 1, 1, runner)

    def test_search_loads_once_and_requests_extra_candidates(self):
        self.write_manifest()
        index, cli = self.root / 'test.ivf', self.root / 'ann_cli'
        index.touch()
        cli.touch()
        runner = MagicMock(return_value=SimpleNamespace(stdout='rank,id,squared_l2,l2\n1,image_0,0,0\n2,image_1,1,1\n', stderr=''))
        encoder = FakeEncoder()
        factory = MagicMock(return_value=encoder)
        result = search_image(self.root / 'image 0.jpg', index, self.manifest, cli, topk=2,
                              deduplicate=True, candidate_multiplier=7, encoder_factory=factory, runner=runner)
        factory.assert_called_once_with('auto')
        self.assertEqual(encoder.calls, 1)
        self.assertEqual(result['candidates_requested'], 14)
        self.assertEqual(result['returned'], 1)
        self.assertEqual(result['matches'][0]['title'], 'Product 0, multilingual č')

    @unittest.skipUnless(shutil.which('clang++') or shutil.which('c++'), 'C++17 compiler unavailable')
    def test_real_cpp_engine_accepts_generated_512d_csv_and_resolves_matches(self):
        self.write_manifest()
        self.embed(batch_size=2)
        cli = self.root / 'ann_cli'
        compiler = shutil.which('clang++') or shutil.which('c++')
        subprocess.run([compiler, '-std=c++17', '-O2', '-I' + str(ROOT / 'include'),
                        str(ROOT / 'src/ivf_flat.cpp'), str(ROOT / 'src/main.cpp'), '-o', str(cli)], check=True, capture_output=True)
        index = self.root / 'test.ivf'
        subprocess.run([str(cli), 'build', '--input', str(self.output), '--output', str(index),
                        '--clusters', '2', '--iterations', '20', '--train-size', '0'], check=True, capture_output=True)
        result = search_image(self.root / 'image 0.jpg', index, self.manifest, cli, topk=3, nprobe=2,
                              encoder_factory=FakeEncoder)
        self.assertEqual(result['matches'][0]['image_id'], 'image_0')
        self.assertAlmostEqual(result['matches'][0]['squared_l2'], 0.0, places=10)
        self.assertEqual(result['returned'], 3)


class EncoderTest(unittest.TestCase):
    def test_device_auto_mps_cpu_cuda_and_explicit_unavailable(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
        self.assertEqual(device_for_run('auto', torch), 'mps')
        torch.backends.mps.is_available = lambda: False
        self.assertEqual(device_for_run('auto', torch), 'cpu')
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            device_for_run('mps', torch)
        torch.cuda.is_available = lambda: True
        self.assertEqual(device_for_run('auto', torch), 'cuda')
        self.assertEqual(device_for_run('cpu', torch), 'cpu')

    def test_model_eval_preprocessing_inference_float32_and_shared_normalization(self):
        torch, open_clip, pil = MagicMock(), MagicMock(), MagicMock()
        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = False
        model, preprocess = MagicMock(), MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        open_clip.create_model_and_transforms.return_value = (model, None, preprocess)
        features = MagicMock()
        features.shape = (1, DIMENSIONS)
        features.float.return_value = features
        features.cpu.return_value.tolist.return_value = [[2.0] * DIMENSIONS]
        model.encode_image.return_value = features
        with patch.dict(sys.modules, {'torch': torch, 'open_clip': open_clip, 'PIL': pil}), patch('clip_embeddings.version', return_value='mock'):
            encoder = ClipEncoder('cpu')
            prepared = encoder.prepare_image('fixture.jpg')
            vector = encoder.encode_batch([prepared])[0]
        open_clip.create_model_and_transforms.assert_called_once_with('ViT-B-32', pretrained='laion2b_s34b_b79k', precision='fp32')
        model.eval.assert_called_once()
        torch.inference_mode.assert_called_once()
        features.float.assert_called_once()
        pil.ImageOps.exif_transpose.assert_called_once()
        preprocess.assert_called_once()
        self.assertEqual(vector, normalize_embedding([2.0] * DIMENSIONS))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable: real tensor inference test not executed')
    def test_real_torch_tensor_inference_without_model_weights(self):
        import torch
        encoder = ClipEncoder.__new__(ClipEncoder)
        encoder.torch, encoder.device = torch, 'cpu'
        def encode(batch):
            self.assertFalse(torch.is_grad_enabled())
            return torch.arange(1, DIMENSIONS + 1, dtype=torch.float16).repeat(len(batch), 1)
        encoder.model = SimpleNamespace(encode_image=encode)
        vectors = encoder.encode_batch([torch.zeros(3, 224, 224), torch.ones(3, 224, 224)])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors[0], normalize_embedding([float(i + 1) for i in range(DIMENSIONS)]))
        encoder.model = SimpleNamespace(encode_image=lambda batch: torch.zeros(1, 2))
        with self.assertRaisesRegex(ValueError, 'output shape'):
            encoder.encode_batch([torch.zeros(3, 224, 224)])

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'Pillow unavailable: real JPEG decode test not executed')
    def test_real_pillow_corrupt_image_and_rgb_conversion(self):
        from PIL import Image, ImageOps
        encoder = ClipEncoder.__new__(ClipEncoder)
        encoder.Image, encoder.ImageOps = Image, ImageOps
        encoder.preprocess = lambda image: (image.mode, image.size)
        with tempfile.TemporaryDirectory() as directory:
            good, bad = Path(directory) / 'good.png', Path(directory) / 'bad.jpg'
            Image.new('L', (3, 4)).save(good)
            bad.write_bytes(b'not a JPEG')
            self.assertEqual(encoder.prepare_image(good), ('RGB', (3, 4)))
            with self.assertRaises(ImageReadError):
                encoder.prepare_image(bad)


if __name__ == '__main__':
    unittest.main()
