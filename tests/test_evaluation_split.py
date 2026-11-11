"""Reproducible held-out splits using small local fixtures; no full ABO processing."""
from contextlib import redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import create_evaluation_split as split


class EvaluationSplitTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='evaluation-split-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / 'vectors-full.csv'
        self.train = self.root / 'vectors-eval-train.csv'
        self.queries = self.root / 'queries-eval.csv'
        self.metadata = Path(str(self.queries) + '.meta.json')
        self.records = self.make_records(30)
        self.source.write_bytes(b''.join(self.records))
        redirect = redirect_stderr(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    @staticmethod
    def make_records(count):
        spellings = [b'0.123456789', b'-0.0', b'+1.0e-03', b'1.00000E+0', b' .5 ', b'\t-2.5e-4\t']
        values = spellings + [b'0'] * (512 - len(spellings))
        return [f'image_{i:05d},'.encode() + b','.join(values) + (b'\r\n' if i % 2 else b'\n') for i in range(count)]

    def create(self, query_count=5, seed=42, **kwargs):
        return split.create_split(self.source, kwargs.pop('train_output', self.train),
                                  kwargs.pop('queries_output', self.queries), query_count, seed, **kwargs)

    @staticmethod
    def read_rows(path):
        with path.open('rb') as stream:
            return list(stream)

    def assert_no_outputs(self):
        for path in (self.train, self.queries, self.metadata):
            self.assertFalse(os.path.lexists(path), str(path))
        self.assertFalse(list(self.root.glob('.*.tmp')))
        self.assertFalse(list(self.root.glob('evaluation-split-*')))

    def test_exact_row_bytes_counts_unique_ids_disjointness_and_source_preservation(self):
        # Last record deliberately has no newline; do not alter its representation.
        self.records[-1] = self.records[-1].rstrip(b'\r\n')
        original = b''.join(self.records)
        self.source.write_bytes(original)
        report = self.create()
        train_rows, query_rows = self.read_rows(self.train), self.read_rows(self.queries)
        self.assertEqual((report['input_vectors'], report['train_vectors'], report['query_vectors']), (30, 25, 5))
        self.assertEqual(report['overlap'], 0)
        self.assertEqual(self.source.read_bytes(), original)
        source_rows = {row.split(b',', 1)[0]: row for row in self.records}
        train_ids = {row.split(b',', 1)[0] for row in train_rows}
        query_ids = {row.split(b',', 1)[0] for row in query_rows}
        self.assertEqual(len(train_ids), 25)
        self.assertEqual(len(query_ids), 5)
        self.assertFalse(train_ids & query_ids)
        self.assertEqual(train_ids | query_ids, set(source_rows))
        for rows in (train_rows, query_rows):
            for row in rows:
                self.assertEqual(row, source_rows[row.split(b',', 1)[0]])
            self.assertEqual(rows, [row for row in self.records if row in rows], 'Source ordering must be preserved')
        metadata = json.loads(self.metadata.read_text())
        self.assertEqual(set(metadata['query_ids']), {identifier.decode() for identifier in query_ids})
        self.assertEqual(metadata['input_sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(metadata['train_sha256'], hashlib.sha256(self.train.read_bytes()).hexdigest())
        self.assertEqual(metadata['queries_sha256'], hashlib.sha256(self.queries.read_bytes()).hexdigest())
        self.assertEqual(metadata['dimensions'], 512)
        self.assertEqual(metadata['seed'], 42)

    def test_identical_seed_produces_identical_csvs_different_seed_changes_queries(self):
        self.create()
        other_train, other_queries = self.root / 'repeat-train.csv', self.root / 'repeat-queries.csv'
        self.create(train_output=other_train, queries_output=other_queries)
        self.assertEqual(self.train.read_bytes(), other_train.read_bytes())
        self.assertEqual(self.queries.read_bytes(), other_queries.read_bytes())
        other_train, other_queries = self.root / 'other-train.csv', self.root / 'other-queries.csv'
        self.create(seed=43, train_output=other_train, queries_output=other_queries)
        self.assertNotEqual(self.queries.read_bytes(), other_queries.read_bytes())
        self.assertNotEqual(self.read_rows(self.queries), self.records[:5], 'Sampling should not just take the first rows')

    def test_default_query_count_is_500(self):
        self.source.write_bytes(b''.join(self.make_records(503)))
        report = split.create_split(self.source, self.train, self.queries)
        self.assertEqual(report['query_vectors'], 500)
        self.assertEqual(report['train_vectors'], 3)
        self.assertEqual(len(self.read_rows(self.queries)), 500)

    def test_invalid_query_count_leaves_no_outputs(self):
        for count in (0, -1, 30, 31, 1000):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.create(query_count=count)
            self.assert_no_outputs()

    def test_duplicate_ids_are_rejected_even_when_values_differ(self):
        self.records[-1] = self.records[-1].replace(b'image_00029', b'image_00000').replace(b'0.123456789', b'0.987654321')
        self.source.write_bytes(b''.join(self.records))
        with self.assertRaisesRegex(ValueError, 'duplicate image ID'):
            self.create()
        self.assert_no_outputs()

    def test_invalid_csv_ids_dimensions_headers_and_coordinates(self):
        valid = self.records[0]
        bad_records = [
            b'', b'\n', b'q,1,2\n', b'image_id,' + b','.join([b'value'] * 512) + b'\n',
            valid.replace(b'image_00000', b''), valid.replace(b'image_00000', b' bad-id'),
            valid.replace(b'image_00000', b'#comment'), valid.replace(b'image_00000', b'"quoted"'),
            valid.replace(b'image_00000', b'bad\x00id'), valid.replace(b'image_00000', b'\xff'),
            valid.rstrip(b'\r\n') + b',0\n',
        ]
        bad_records.extend(valid.replace(b'0.123456789', token) for token in
                           (b'nan', b'inf', b'1e999', b'1e40', b'1_000', b'1.0suffix', b''))
        for bad in bad_records:
            # Empty input is tested separately; include a malformed row after valid ones here.
            self.source.write_bytes(b''.join(self.records[:3]) + bad if bad else b'')
            with self.subTest(record=bad[:35]), self.assertRaises(ValueError):
                self.create(query_count=1)
            self.assert_no_outputs()

    def test_refuses_existing_train_query_metadata_and_index_destinations(self):
        for path in (self.train, self.queries, self.metadata):
            path.write_bytes(b'existing file')
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'overwrite existing'):
                self.create()
            self.assertEqual(path.read_bytes(), b'existing file')
            path.unlink()
            self.assert_no_outputs()
        index = self.root / 'abo-full.ivf'
        index.write_bytes(b'existing IVF index')
        with self.assertRaisesRegex(ValueError, 'overwrite existing'):
            self.create(train_output=index)
        self.assertEqual(index.read_bytes(), b'existing IVF index')

    def test_refuses_input_output_aliases_and_identical_outputs(self):
        original = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, 'overwrite existing'):
            self.create(train_output=self.source)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            self.create(queries_output=self.train)
        alias = self.root / 'alias.csv'
        alias.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, 'overwrite existing'):
            self.create(train_output=alias)
        alias.unlink()
        os.link(self.source, alias)
        with self.assertRaisesRegex(ValueError, 'overwrite existing'):
            self.create(train_output=alias)
        self.assertEqual(self.source.read_bytes(), original)
        self.assert_no_outputs()

    def test_streams_input_instead_of_reading_entire_file(self):
        original_open = Path.open
        scans = []
        class StreamingFile:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def __iter__(self):
                scans.append(True)
                return iter(self.stream)
            def read(self, *args):
                raise AssertionError('Input must be streamed, not read into memory')
            def readlines(self, *args):
                raise AssertionError('Input must be streamed, not readlines()')
        def open_file(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return StreamingFile(stream) if path == self.source else stream
        with patch.object(Path, 'open', open_file):
            self.create()
        self.assertEqual(len(scans), 2)

    def test_changed_input_between_passes_is_detected(self):
        original_random = split.random.Random
        def edit_source(seed):
            self.source.write_bytes(self.source.read_bytes().replace(b'0.123456789', b'0.987654321', 1))
            return original_random(seed)
        with patch('create_evaluation_split.random.Random', side_effect=edit_source):
            with self.assertRaisesRegex(ValueError, 'Input changed'):
                self.create()
        self.assert_no_outputs()

    def test_output_coordinate_corruption_is_detected_before_publication(self):
        verify = split.verify_output
        def corrupt(path, partition, *args):
            if partition == 'train':
                path.write_bytes(path.read_bytes().replace(b'0.123456789', b'0.987654321', 1))
            return verify(path, partition, *args)
        with patch('create_evaluation_split.verify_output', side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                self.create()
        self.assert_no_outputs()

    def test_output_overlap_is_detected_before_publication(self):
        verify = split.verify_output
        train_row = []
        def overlap(path, partition, *args):
            if partition == 'train':
                with path.open('rb') as stream:
                    train_row.append(next(stream))
            else:
                with path.open('ab') as stream:
                    stream.write(train_row[0])
            return verify(path, partition, *args)
        with patch('create_evaluation_split.verify_output', side_effect=overlap):
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                self.create()
        self.assert_no_outputs()

    def test_publication_race_preserves_concurrent_query_file_and_rolls_back_train(self):
        link = os.link
        def race(temporary, destination):
            if destination == self.queries:
                destination.write_bytes(b'external query file')
            return link(temporary, destination)
        with patch('create_evaluation_split.os.link', side_effect=race), self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(self.queries.read_bytes(), b'external query file')
        self.assertFalse(self.train.exists())
        self.assertFalse(self.metadata.exists())
        self.assertFalse(list(self.root.glob('.*.tmp')))

    def test_custom_output_directories_and_cli_options(self):
        train, queries = self.root / 'index-vectors/train.csv', self.root / 'eval/queries.csv'
        result = subprocess.run([
            sys.executable, str(ROOT / 'scripts/create_evaluation_split.py'),
            '--input', str(self.source), '--train-output', str(train), '--queries-output', str(queries),
            '--query-count', '7', '--seed', '123',
        ], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        metrics = dict(line.split('=', 1) for line in result.stdout.splitlines())
        self.assertEqual(metrics['query_vectors'], '7')
        self.assertEqual(metrics['train_vectors'], '23')
        self.assertEqual(metrics['overlap'], '0')
        self.assertEqual(metrics['seed'], '123')
        self.assertEqual(len(self.read_rows(queries)), 7)
        repeated = subprocess.run([
            sys.executable, str(ROOT / 'scripts/create_evaluation_split.py'),
            '--input', str(self.source), '--train-output', str(train), '--queries-output', str(queries),
            '--query-count', '7',
        ], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn('Refusing to overwrite', repeated.stderr)


if __name__ == '__main__':
    unittest.main()
