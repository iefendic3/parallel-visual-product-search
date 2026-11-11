"""Benchmark CLI integration tests; all repeated searches run inside C++.

Optional ANN_CLI points to an existing executable. Otherwise compile a temporary
CLI once for this suite. No ABO data, image dependencies, or model weights needed.
"""
import csv
import io
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='ann-bench-test-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        configured_cli = os.environ.get('ANN_CLI')
        if configured_cli:
            cls.cli = Path(configured_cli).resolve()
        else:
            compiler = shutil.which('clang++') or shutil.which('c++')
            if not compiler:
                raise unittest.SkipTest('C++17 compiler unavailable; set ANN_CLI to a built CLI')
            cls.cli = cls.directory / 'ann_cli'
            subprocess.run([
                compiler, '-std=c++17', '-O2', '-g', '-fno-omit-frame-pointer',
                '-I' + str(ROOT / 'include'), str(ROOT / 'src/ivf_flat.cpp'),
                str(ROOT / 'src/main.cpp'), '-o', str(cls.cli),
            ], check=True, capture_output=True, text=True, timeout=120)
        cls.index = cls.directory / 'sample.ivf'
        cls.queries = ROOT / 'examples/sample_queries.csv'
        subprocess.run([
            str(cls.cli), 'build', '--input', str(ROOT / 'examples/sample_vectors.csv'),
            '--output', str(cls.index), '--clusters', '3', '--iterations', '25', '--train-size', '0',
        ], check=True, capture_output=True, text=True, timeout=10)

    def bench(self, **overrides):
        options = {'index': self.index, 'queries': self.queries, 'mode': 'ann',
                   'topk': 3, 'nprobe': 1, 'min-duration': 0.015, 'warmup': 7}
        options.update(overrides)
        command = [str(self.cli), 'bench']
        for key, value in options.items():
            if value is not None:
                command.extend(['--' + key, str(value)])
        return subprocess.run(command, capture_output=True, text=True, timeout=10)

    def metrics(self, **overrides):
        completed = self.bench(**overrides)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('Warmup complete; starting measured search phase', completed.stderr)
        lines = [line.split('=', 1) for line in completed.stdout.splitlines()]
        self.assertTrue(all(len(line) == 2 for line in lines))
        result = dict(lines)
        self.assertEqual(len(result), len(lines), 'Metric keys must be unique')
        return result

    def assert_valid_measurements(self, metrics):
        elapsed = float(metrics['measured_seconds'])
        search_seconds = float(metrics['timed_search_seconds'])
        count = int(metrics['searches'])
        self.assertGreaterEqual(elapsed, float(metrics['min_duration_seconds']))
        self.assertGreater(count, int(metrics['query_count']), 'Queries should be cycled repeatedly')
        self.assertGreater(search_seconds, 0)
        self.assertLessEqual(search_seconds, elapsed)
        for key in ('mean_search_us', 'median_search_us', 'p95_search_us', 'queries_per_second'):
            self.assertTrue(math.isfinite(float(metrics[key])), key)
            self.assertGreaterEqual(float(metrics[key]), 0, key)
        self.assertAlmostEqual(float(metrics['mean_search_us']), search_seconds * 1e6 / count, delta=0.0002)
        self.assertAlmostEqual(float(metrics['queries_per_second']), count / elapsed,
                               delta=max(0.01, count / elapsed * 1e-6))
        self.assertGreaterEqual(float(metrics['p95_search_us']), float(metrics['median_search_us']))
        samples, capacity = int(metrics['latency_samples']), int(metrics['latency_sample_capacity'])
        self.assertEqual(samples, min(count, capacity))
        self.assertEqual(metrics['latency_sampling'], 'all' if count <= capacity else 'reservoir')
        self.assertGreater(int(metrics['checksum']), 0)

    def test_ann_configuration_duration_and_metric_formulas(self):
        metrics = self.metrics()
        self.assert_valid_measurements(metrics)
        self.assertEqual(metrics['mode'], 'ann')
        self.assertEqual(metrics['nprobe'], '1')
        self.assertEqual(metrics['warmup'], '7')
        self.assertEqual(metrics['topk'], '3')
        self.assertEqual(metrics['index_vectors'], '12')
        self.assertEqual(metrics['dimensions'], '2')
        self.assertEqual(metrics['clusters'], '3')
        self.assertEqual(metrics['query_count'], '3')
        self.assertEqual(metrics['index'], str(self.index))
        self.assertEqual(metrics['queries'], str(self.queries))

    def test_exact_mode_ignores_cluster_probe_count(self):
        metrics = self.metrics(mode='exact', nprobe=999)
        self.assert_valid_measurements(metrics)
        self.assertEqual(metrics['mode'], 'exact')
        self.assertEqual(metrics['nprobe'], 'not_applicable')

    def test_all_cluster_ann_and_exact_consume_same_warmup_results(self):
        ann = self.metrics(nprobe=3, warmup=11)
        exact = self.metrics(mode='exact', warmup=11)
        repeated = self.metrics(nprobe=3, warmup=11)
        self.assertEqual(ann['warmup_checksum'], exact['warmup_checksum'])
        self.assertEqual(ann['warmup_checksum'], repeated['warmup_checksum'])

    def test_zero_warmup_and_topk_larger_than_database(self):
        metrics = self.metrics(warmup=0, topk=100, nprobe=3)
        self.assert_valid_measurements(metrics)
        self.assertEqual(metrics['warmup'], '0')
        self.assertNotEqual(metrics['checksum'], metrics['warmup_checksum'])

    def test_default_mode_is_ann(self):
        self.assertEqual(self.metrics(mode=None)['mode'], 'ann')

    def test_collection_remains_bounded_during_longer_run(self):
        metrics = self.metrics(**{'min-duration': 0.08})
        self.assert_valid_measurements(metrics)
        self.assertLessEqual(int(metrics['latency_samples']), 65536)

    def test_invalid_benchmark_options(self):
        invalid = [
            {'mode': 'both'}, {'mode': ''}, {'topk': 0}, {'topk': -1}, {'topk': '2x'},
            {'nprobe': 0}, {'nprobe': 4}, {'warmup': -1}, {'warmup': ' -1'},
            {'warmup': '+1'}, {'warmup': '18446744073709551616'},
            {'min-duration': 0}, {'min-duration': -0.01}, {'min-duration': 'nan'},
            {'min-duration': 'inf'}, {'min-duration': '1e999'},
            {'min-duration': 'garbage'}, {'min-duration': '0.01suffix'},
            {'min-duraton': 0.01}, {'index': None}, {'queries': None},
        ]
        for options in invalid:
            with self.subTest(options=options):
                completed = self.bench(**options)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn('Error:', completed.stderr)
                self.assertNotIn('measured_seconds=', completed.stdout)

    def test_empty_mismatched_and_missing_query_files(self):
        for name, content in [('empty.csv', ''), ('wrong.csv', 'q,1,2,3\n')]:
            path = self.directory / name
            path.write_text(content)
            with self.subTest(name=name):
                completed = self.bench(queries=path, warmup=0)
                self.assertNotEqual(completed.returncode, 0)
                self.assertNotIn('starting measured search', completed.stderr)
        self.assertNotEqual(self.bench(queries=self.directory / 'missing.csv').returncode, 0)

    def test_existing_help_search_and_evaluate_contracts(self):
        help_result = subprocess.run([str(self.cli), '--help'], capture_output=True, text=True, check=True)
        self.assertIn('ann_cli bench', help_result.stdout)
        search = subprocess.run([
            str(self.cli), 'search', '--index', str(self.index), '--query', '0.11,0.12',
            '--topk', '3', '--nprobe', '1',
        ], check=True, capture_output=True, text=True)
        rows = list(csv.reader(io.StringIO(search.stdout)))
        self.assertEqual(rows[0], ['rank', 'id', 'squared_l2', 'l2'])
        self.assertEqual(len(rows), 4)
        evaluated = subprocess.run([
            str(self.cli), 'evaluate', '--index', str(self.index), '--queries', str(self.queries),
            '--topk', '3', '--nprobe', '3',
        ], check=True, capture_output=True, text=True)
        metrics = dict(line.split('=', 1) for line in evaluated.stdout.splitlines())
        self.assertEqual(set(metrics), {'queries', 'recall_at_3', 'ann_mean_us', 'exact_mean_us', 'search_speedup_vs_exact'})
        self.assertEqual(metrics['queries'], '3')
        self.assertEqual(float(metrics['recall_at_3']), 1.0)


if __name__ == '__main__':
    unittest.main()
