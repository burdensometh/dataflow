#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tempfile
import numpy as np
import os
import unittest

from tensorpack.dataflow import HDF5Serializer, LMDBSerializer, NumpySerializer, TFRecordSerializer
from tensorpack.dataflow.base import DataFlow
from tensorpack.dataflow.format import IncompleteLMDBError
from tensorpack.dataflow.serialize import LMDBResumeError


def delete_file_if_exists(fn):
    try:
        os.remove(fn)
    except OSError:
        pass


class SeededFakeDataFlow(DataFlow):
    """docstring for SeededFakeDataFlow"""

    def __init__(self, seed=42, size=32):
        super(SeededFakeDataFlow, self).__init__()
        self.seed = seed
        self._size = size
        self.cache = []

    def reset_state(self):
        np.random.seed(self.seed)
        for _ in range(self._size):
            label = np.random.randint(low=0, high=10)
            img = np.random.randn(28, 28, 3)
            self.cache.append([label, img])

    def __len__(self):
        return self._size

    def __iter__(self):
        for dp in self.cache:
            yield dp


class InterruptibleDataFlow(DataFlow):
    """
    A fixed, repeatable, finite source whose content is a deterministic function
    of the index.  It can be told to raise at a given index to simulate an
    interrupted export, and to produce different content while keeping the same
    class and length.
    """

    def __init__(self, size=64, offset=0, fail_at=None):
        self._size = size
        self.offset = offset
        self.fail_at = fail_at

    def __len__(self):
        return self._size

    def __iter__(self):
        for k in range(self._size):
            if self.fail_at is not None and k == self.fail_at:
                raise RuntimeError("simulated interruption at {}".format(k))
            yield [k, np.full((4,), k + self.offset, dtype='float32')]


class SerializerTest(unittest.TestCase):

    def run_write_read_test(self, file, serializer, w_args, w_kwargs, r_args, r_kwargs, error_msg):
        try:
            delete_file_if_exists(file)

            ds_expected = SeededFakeDataFlow()
            serializer.save(ds_expected, file, *w_args, **w_kwargs)
            ds_actual = serializer.load(file, *r_args, **r_kwargs)

            ds_actual.reset_state()
            ds_expected.reset_state()

            for dp_expected, dp_actual in zip(ds_expected.__iter__(), ds_actual.__iter__()):
                self.assertEqual(dp_expected[0], dp_actual[0])
                self.assertTrue(np.allclose(dp_expected[1], dp_actual[1]))
        except ImportError:
            print(error_msg)

    def test_lmdb(self):
        with tempfile.TemporaryDirectory() as f:
            self.run_write_read_test(
                os.path.join(f, 'test.lmdb'),
                LMDBSerializer,
                {}, {},
                {}, {'shuffle': False},
                'Skip test_lmdb, no lmdb available')

    def test_tfrecord(self):
        with tempfile.TemporaryDirectory() as f:
            self.run_write_read_test(
                os.path.join(f, 'test.tfrecord'),
                TFRecordSerializer,
                {}, {},
                {}, {'size': 32},
                'Skip test_tfrecord, no tensorflow available')

    def test_numpy(self):
        with tempfile.TemporaryDirectory() as f:
            self.run_write_read_test(
                os.path.join(f, 'test.npz'),
                NumpySerializer,
                {}, {},
                {}, {'shuffle': False},
                'Skip test_numpy, no numpy available')

    def test_hdf5(self):
        args = [['label', 'image']]
        with tempfile.TemporaryDirectory() as f:
            self.run_write_read_test(
                os.path.join(f, 'test.h5'),
                HDF5Serializer,
                args, {},
                args, {'shuffle': False},
                'Skip test_hdf5, no h5py available')


def _read_all(path):
    df = LMDBSerializer.load(path, shuffle=False)
    df.reset_state()
    return list(df)


class LMDBResumeTest(unittest.TestCase):
    """Tests for the resumable export entry point."""

    def _path(self, f, name='test.lmdb'):
        return os.path.join(f, name)

    def test_full_export(self):
        with tempfile.TemporaryDirectory() as f:
            path = self._path(f)
            n = LMDBSerializer.save_resume(InterruptibleDataFlow(64), path, write_frequency=10)
            self.assertEqual(n, 64)
            self.assertEqual(LMDBSerializer.export_status(path), 'complete')
            dps = _read_all(path)
            self.assertEqual(len(dps), 64)
            self.assertEqual([dp[0] for dp in dps], list(range(64)))
            self.assertTrue(all(np.all(dp[1] == k) for k, dp in enumerate(dps)))

    def test_interrupted_then_resumed(self):
        with tempfile.TemporaryDirectory() as f:
            path = self._path(f)
            with self.assertRaises(RuntimeError):
                LMDBSerializer.save_resume(
                    InterruptibleDataFlow(64, fail_at=35), path, write_frequency=10)

            # Partial data is recognized as incomplete rather than read as if full.
            self.assertEqual(LMDBSerializer.export_status(path), 'incomplete')
            with self.assertRaises(IncompleteLMDBError):
                LMDBSerializer.load(path, shuffle=False)

            n = LMDBSerializer.save_resume(InterruptibleDataFlow(64), path, write_frequency=10)
            self.assertEqual(n, 64)
            self.assertEqual(LMDBSerializer.export_status(path), 'complete')
            dps = _read_all(path)
            self.assertEqual([dp[0] for dp in dps], list(range(64)))
            self.assertEqual(len(dps), 64)      # no duplicates from the retry

    def test_failed_batch_not_counted(self):
        with tempfile.TemporaryDirectory() as f:
            path = self._path(f)
            with self.assertRaises(RuntimeError):
                LMDBSerializer.save_resume(
                    InterruptibleDataFlow(64, fail_at=35), path, write_frequency=10)
            # Batches are committed atomically, so only 30 datapoints made it.
            LMDBSerializer.save_resume(InterruptibleDataFlow(64), path, write_frequency=10)
            self.assertEqual(len(_read_all(path)), 64)

    def test_completed_export_is_idempotent(self):
        with tempfile.TemporaryDirectory() as f:
            path = self._path(f)
            LMDBSerializer.save_resume(InterruptibleDataFlow(32), path, write_frequency=8)
            self.assertEqual(
                LMDBSerializer.save_resume(InterruptibleDataFlow(32), path, write_frequency=8), 32)
            self.assertEqual(len(_read_all(path)), 32)

    def test_map_growth_keeps_current_batch(self):
        with tempfile.TemporaryDirectory() as f:
            path = self._path(f)
            # A tiny map forces growth while a batch is being written.
            n = LMDBSerializer.save_resume(
                InterruptibleDataFlow(64), path, write_frequency=20, map_size=8 * 1024)
            self.assertEqual(n, 64)
            dps = _read_all(path)
            self.assertEqual(len(dps), 64)
            self.assertEqual([dp[0] for dp in dps], list(range(64)))

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            path = self._path(f)
            with self.assertRaises(RuntimeError):
                LMDBSerializer.save_resume(
                    InterruptibleDataFlow(64, fail_at=40), path, write_frequency=10)
            # Same class and length, different content: must not be mixed in.
            with self.assertRaises(LMDBResumeError):
                LMDBSerializer.save_resume(
                    InterruptibleDataFlow(64, offset=1000), path, write_frequency=10)

    def test_changed_source_id_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            path = self._path(f)
            with self.assertRaises(RuntimeError):
                LMDBSerializer.save_resume(
                    InterruptibleDataFlow(64, fail_at=30), path, write_frequency=10,
                    source_id='dataset-v1', export_definition='prep-1')
            with self.assertRaises(LMDBResumeError):
                LMDBSerializer.save_resume(
                    InterruptibleDataFlow(64), path, write_frequency=10,
                    source_id='dataset-v2', export_definition='prep-1')
            with self.assertRaises(LMDBResumeError):
                LMDBSerializer.save_resume(
                    InterruptibleDataFlow(64), path, write_frequency=10,
                    source_id='dataset-v1', export_definition='prep-2')
            n = LMDBSerializer.save_resume(
                InterruptibleDataFlow(64), path, write_frequency=10,
                source_id='dataset-v1', export_definition='prep-1')
            self.assertEqual(n, 64)

    def test_one_shot_still_works(self):
        with tempfile.TemporaryDirectory() as f:
            path = self._path(f)
            LMDBSerializer.save(InterruptibleDataFlow(32), path)
            self.assertIsNone(LMDBSerializer.export_status(path))
            self.assertEqual(len(_read_all(path)), 32)
            with self.assertRaises(AssertionError):
                LMDBSerializer.save(InterruptibleDataFlow(32), path)

    def test_refuses_foreign_database(self):
        import lmdb
        with tempfile.TemporaryDirectory() as f:
            path = self._path(f)
            db = lmdb.open(path, subdir=False, map_size=1 << 20)
            with db.begin(write=True) as txn:
                txn.put(b'00000000', b'not a resumable export')
            db.close()
            with self.assertRaises(LMDBResumeError):
                LMDBSerializer.save_resume(InterruptibleDataFlow(8), path)

    def test_requires_finite_source(self):
        from tensorpack.dataflow import DataFromGenerator

        def gen():
            for k in range(8):
                yield [k]

        with tempfile.TemporaryDirectory() as f:
            path = self._path(f)
            with self.assertRaises(LMDBResumeError):
                LMDBSerializer.save_resume(DataFromGenerator(gen), path)


if __name__ == '__main__':
    unittest.main()
