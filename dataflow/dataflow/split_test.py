#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: split_test.py

import os
import tempfile
import threading
import time
import unittest

import numpy as np

from dataflow.dataflow.base import DataFlow, DataFlowTerminated
from dataflow.dataflow.common import BatchData, MapData
from dataflow.dataflow.split import DataFlowSplitter, split_dataflow


class CountingDataFlow(DataFlow):
    """A one-shot-ish source: records how often it is reset and iterated."""

    def __init__(self, data, delay=0.0):
        self.data = list(data)
        self.delay = delay
        self.reset_count = 0
        self.iter_count = 0

    def reset_state(self):
        self.reset_count += 1

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.iter_count += 1
        for dp in self.data:
            if self.delay:
                time.sleep(self.delay)
            yield dp


class FailingDataFlow(DataFlow):
    def __init__(self, data, fail_after):
        self.data = list(data)
        self.fail_after = fail_after

    def reset_state(self):
        pass

    def __iter__(self):
        for k, dp in enumerate(self.data):
            if k == self.fail_after:
                raise ValueError("source exploded")
            yield dp
        raise ValueError("source exploded at end")


def drain(iterable):
    return [dp for dp in iterable]


class SplitterTest(unittest.TestCase):

    def test_basic_tee_order_and_single_pass(self):
        src = CountingDataFlow([k for k in range(50)])
        splitter = DataFlowSplitter(src, buffer_size=4)
        b0 = splitter.branch('a')
        b1 = splitter.branch('b')
        b0.reset_state()

        out0 = drain(b0)
        out1 = drain(b1)
        self.assertEqual(out0, list(range(50)))
        self.assertEqual(out1, list(range(50)))
        self.assertEqual(src.reset_count, 1)
        self.assertEqual(src.iter_count, 1)
        splitter.close()

    def test_wrapped_by_map_and_batch(self):
        src = CountingDataFlow([[k] for k in range(20)])
        splitter = DataFlowSplitter(src, buffer_size=3)
        mapped = MapData(splitter.branch(), lambda dp: [dp[0] * 10])
        batched = BatchData(splitter.branch(), 4)

        mapped.reset_state()
        self.assertEqual(drain(mapped), [[k * 10] for k in range(20)])
        batches = drain(batched)
        self.assertEqual(len(batches), 5)
        self.assertEqual(np.concatenate([b[0] for b in batches]).tolist(),
                         list(range(20)))
        self.assertEqual(src.iter_count, 1)
        splitter.close()

    def test_backpressure_bounded_memory(self):
        src = CountingDataFlow([k for k in range(30)])
        splitter = DataFlowSplitter(src, buffer_size=5, spill=False)
        fast = splitter.branch('fast')
        slow = splitter.branch('slow')

        got = []
        got_slow = []

        def consume_slow():
            for dp in slow:
                got_slow.append(dp)
                time.sleep(0.002)

        th = threading.Thread(target=consume_slow)
        th.start()
        max_observed = 0
        for dp in fast:
            got.append(dp)
            max_observed = max(max_observed, slow.in_memory)
        th.join()
        self.assertEqual(got, list(range(30)))
        self.assertEqual(got_slow, list(range(30)))
        self.assertLessEqual(max_observed, 5)
        splitter.close()

    def test_spill_when_slow_then_catch_up(self):
        src = CountingDataFlow([k for k in range(60)], delay=0.001)
        splitter = DataFlowSplitter(src, buffer_size=4)
        fast = splitter.branch('fast')
        slow = splitter.branch('slow')

        # Let the fast branch run to completion.  The slow branch is not
        # consumed yet, so it must spill its excess to disk.
        fast_out = drain(fast)
        self.assertEqual(fast_out, list(range(60)))
        self.assertTrue(slow.spilled)
        self.assertGreater(slow.spilled_count, 0)
        self.assertLessEqual(slow.in_memory, 4)
        # A spill file exists while the branch is behind.
        self.assertIsNotNone(slow.spill_path)
        self.assertTrue(os.path.exists(slow.spill_path))

        slow_out = drain(slow)
        self.assertEqual(slow_out, list(range(60)))
        # Once caught up, the disk space is released.
        self.assertIsNone(slow.spill_path)
        splitter.close()

    def test_zero_buffer_spills_everything(self):
        src = CountingDataFlow(list(range(10)))
        splitter = DataFlowSplitter(src, buffer_size=0)
        b = splitter.branch()
        self.assertEqual(drain(b), list(range(10)))
        self.assertEqual(b.in_memory, 0)
        self.assertIsNone(b.spill_path)
        splitter.close()

    def test_close_branch_releases_and_others_continue(self):
        src = CountingDataFlow(list(range(40)), delay=0.001)
        splitter = DataFlowSplitter(src, buffer_size=2)
        kept = splitter.branch('kept')
        dropped = splitter.branch('dropped')

        kept_out = []

        def consume_kept():
            for dp in kept:
                kept_out.append(dp)
                time.sleep(0.001)
                if len(kept_out) == 10 and not dropped.closed:
                    # Close the other branch mid-stream.
                    dropped.close()

        th = threading.Thread(target=consume_kept)
        th.start()
        th.join()
        self.assertTrue(dropped.closed)
        self.assertEqual(dropped.in_memory, 0)
        self.assertIsNone(dropped.spill_path)
        self.assertEqual(dropped.buffered_count, 0)
        # The kept branch still gets every row, in order.
        self.assertEqual(kept_out, list(range(40)))
        splitter.close()

    def test_close_branch_with_spill_frees_files(self):
        src = CountingDataFlow(list(range(30)))
        splitter = DataFlowSplitter(src, buffer_size=2, spill_dir=None)
        fast = splitter.branch('fast')
        slow = splitter.branch('slow')
        drain(fast)
        self.assertTrue(slow.spilled)
        path = slow.spill_path
        self.assertTrue(os.path.exists(path))
        slow.close()
        self.assertFalse(os.path.exists(path))
        self.assertIsNone(slow.spill_path)
        self.assertEqual(slow.in_memory, 0)
        splitter.close()

    def test_upstream_failure_propagates_after_drain(self):
        src = FailingDataFlow([1, 2, 3], fail_after=2)
        splitter = DataFlowSplitter(src, buffer_size=1)
        b0 = splitter.branch()
        b1 = splitter.branch()

        def consume(branch, out):
            try:
                for dp in branch:
                    out.append(dp)
            except ValueError as e:
                out.append(('error', str(e)))

        out0, out1 = [], []
        t0 = threading.Thread(target=consume, args=(b0, out0))
        t1 = threading.Thread(target=consume, args=(b1, out1))
        t0.start()
        t1.start()
        t0.join()
        t1.join()

        self.assertEqual(out0[:2], [1, 2])
        self.assertEqual(out1[:2], [1, 2])
        self.assertEqual(out0[2][0], 'error')
        self.assertEqual(out1[2][0], 'error')
        self.assertIn('source exploded', out0[2][1])
        splitter.close()

    def test_upstream_terminated_propagates_after_drain(self):
        class Terminated(DataFlow):
            def reset_state(self):
                pass

            def __iter__(self):
                yield 1
                yield 2
                raise DataFlowTerminated()

        splitter = DataFlowSplitter(Terminated(), buffer_size=1)
        b0 = splitter.branch()
        b1 = splitter.branch()
        for branch in (b0, b1):
            got = []
            with self.assertRaises(DataFlowTerminated):
                for dp in branch:
                    got.append(dp)
            self.assertEqual(got, [1, 2])
        splitter.close()

    def test_reset_failure_propagates(self):
        class BadReset(DataFlow):
            def reset_state(self):
                raise RuntimeError("cannot reset")

            def __iter__(self):
                yield 1

        splitter = DataFlowSplitter(BadReset())
        b0 = splitter.branch()
        b1 = splitter.branch()
        with self.assertRaises(RuntimeError):
            b0.reset_state()
        with self.assertRaises(RuntimeError):
            drain(b1)
        splitter.close()

    def test_branch_after_start_rejected(self):
        src = CountingDataFlow(range(10))
        splitter = DataFlowSplitter(src, buffer_size=2)
        b = splitter.branch()
        b.reset_state()
        it = iter(b)
        next(it)
        with self.assertRaises(RuntimeError):
            splitter.branch()
        # Consume the rest without leaking the iterator.
        drain(it)
        splitter.close()

    def test_len_and_split_helper(self):
        src = CountingDataFlow(range(7))
        branches = split_dataflow(src, num_branches=2, buffer_size=2)
        self.assertEqual(len(branches), 2)
        self.assertEqual(len(branches[0]), 7)
        self.assertEqual(drain(branches[0]), list(range(7)))
        self.assertEqual(drain(branches[1]), list(range(7)))
        branches[0].splitter.close()

    def test_dict_datapoints(self):
        data = [{'x': k, 'y': np.full(3, k)} for k in range(12)]
        src = CountingDataFlow(data)
        splitter = DataFlowSplitter(src, buffer_size=2)
        a = splitter.branch()
        b = splitter.branch()
        out_a = drain(MapData(a, lambda dp: dp['x']))
        out_b = drain(MapData(b, lambda dp: dp['y'].tolist()))
        self.assertEqual(out_a, list(range(12)))
        self.assertEqual(out_b, [[k] * 3 for k in range(12)])
        splitter.close()

    def test_unpicklable_datapoints_preserve_order(self):
        class Unpicklable(object):
            def __init__(self, v):
                self.v = v

            def __reduce__(self):
                raise TypeError("no pickling for you")

        data = [Unpicklable(k) for k in range(10)]
        src = CountingDataFlow(data)
        splitter = DataFlowSplitter(src, buffer_size=1)
        fast = splitter.branch('fast')
        slow = splitter.branch('slow')
        drain(fast)
        # The slow branch could not spill, but it must still see everything
        # in order.
        values = [dp.v for dp in slow]
        self.assertEqual(values, list(range(10)))
        splitter.close()

    def test_interleaved_unpicklable_keeps_memory_bounded(self):
        class Unpicklable(object):
            def __init__(self, v):
                self.v = v

            def __reduce__(self):
                raise TypeError("no pickling for you")

        # Every third row cannot be pickled.  Spillable rows around them must
        # still go to disk, so memory stays bounded.
        data = []
        for k in range(60):
            if k % 3 == 0:
                data.append(Unpicklable(k))
            else:
                data.append(k)

        src = CountingDataFlow(data)
        splitter = DataFlowSplitter(src, buffer_size=2)
        fast = splitter.branch('fast')
        slow = splitter.branch('slow')
        drain(fast)
        self.assertTrue(slow.spilled)
        self.assertLessEqual(slow.in_memory, 30)
        self.assertGreater(len(slow.spill_paths), 1)
        out = [getattr(dp, 'v', dp) for dp in slow]
        self.assertEqual(out, list(range(60)))
        self.assertEqual(slow.spill_paths, [])
        splitter.close()

    def test_many_uneven_branches_stress(self):
        n = 200
        src = CountingDataFlow([[k] for k in range(n)], delay=0.0002)
        splitter = DataFlowSplitter(src, buffer_size=3)
        # Two slow branches that spill and one fast one that never does.
        fast = splitter.branch('fast')
        slow1 = splitter.branch('slow1')
        slow2 = splitter.branch('slow2')

        results = {}

        def consume(branch, delay):
            out = []
            for dp in branch:
                out.append(dp[0])
                if delay:
                    time.sleep(delay)
            results[branch.name] = out

        threads = [
            threading.Thread(target=consume, args=(slow1, 0.001)),
            threading.Thread(target=consume, args=(slow2, 0.002)),
            threading.Thread(target=consume, args=(fast, 0)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(results['fast'], list(range(n)))
        self.assertEqual(results['slow1'], list(range(n)))
        self.assertEqual(results['slow2'], list(range(n)))
        self.assertEqual(src.iter_count, 1)
        self.assertEqual(src.reset_count, 1)
        self.assertEqual(splitter.produced_count, n)
        splitter.close()

    def test_backpressure_close_unblocks_pump(self):
        src = CountingDataFlow(list(range(100)))
        splitter = DataFlowSplitter(src, buffer_size=2, spill=False)
        kept = splitter.branch('kept')
        stalled = splitter.branch('stalled')

        out = []

        def consume_kept():
            for dp in kept:
                out.append(dp)
                time.sleep(0.0005)

        th = threading.Thread(target=consume_kept)
        th.start()
        # The pump fills the stalled branch and then blocks on it.
        deadline = time.time() + 5
        while splitter.produced_count < 3 and time.time() < deadline:
            time.sleep(0.005)
        self.assertEqual(stalled.in_memory, 2)
        produced_before = splitter.produced_count

        # Closing the stalled branch must let the pump move on.
        stalled.close()
        th.join(10)
        self.assertFalse(th.is_alive())
        self.assertEqual(out, list(range(100)))
        self.assertGreater(splitter.produced_count, produced_before)
        self.assertEqual(src.iter_count, 1)
        splitter.close()

    def test_reset_state_is_idempotent(self):
        src = CountingDataFlow(range(5))
        splitter = DataFlowSplitter(src, buffer_size=2, num_branches=2)
        splitter.reset_state()
        splitter.reset_state()
        for b in splitter.branches:
            b.reset_state()
        self.assertEqual(src.reset_count, 1)
        self.assertEqual(drain(splitter.branches[0]), list(range(5)))
        self.assertEqual(drain(splitter.branches[1]), list(range(5)))
        self.assertEqual(src.reset_count, 1)
        splitter.close()

    def test_byte_quota(self):
        data = [np.zeros(1024, dtype='float32') for _ in range(40)]
        src = CountingDataFlow(data)
        splitter = DataFlowSplitter(
            src, buffer_size=None, max_buffer_bytes=8 * 1024)
        fast = splitter.branch('fast')
        slow = splitter.branch('slow')
        drain(fast)
        self.assertTrue(slow.spilled)
        self.assertLessEqual(slow.in_memory * 4096, 8 * 1024 + 4096)
        self.assertEqual(len(drain(slow)), 40)
        splitter.close()

    def test_close_during_backpressure(self):
        src = CountingDataFlow(list(range(500)))
        splitter = DataFlowSplitter(src, buffer_size=2, spill=False)
        b = splitter.branch()
        it = iter(b)
        next(it)
        # The pump blocks on this branch until it is closed.
        t0 = time.time()
        splitter.close()
        self.assertLess(time.time() - t0, 5)
        self.assertTrue(b.closed)
        # Release the iterator before iterating the branch again.
        del it
        self.assertEqual(drain(b), [])
        self.assertFalse(os.path.isdir(splitter.spill_dir or ''))

    def test_explicit_spill_dir_is_kept_but_emptied(self):
        src = CountingDataFlow(list(range(40)))
        d = tempfile.mkdtemp(prefix='splitter-test-')
        try:
            splitter = DataFlowSplitter(src, buffer_size=1, spill_dir=d)
            fast = splitter.branch('fast')
            slow = splitter.branch('slow')
            drain(fast)
            self.assertTrue(slow.spilled)
            splitter.close()
            self.assertTrue(os.path.isdir(d))
            self.assertEqual(os.listdir(d), [])
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_splitter_close_stops_everything(self):
        src = CountingDataFlow(list(range(100)), delay=0.001)
        splitter = DataFlowSplitter(src, buffer_size=2)
        b0 = splitter.branch()
        b1 = splitter.branch()
        drain(b0)          # run to completion
        splitter.close()
        self.assertTrue(b1.closed)
        self.assertEqual(drain(b1), [])
        self.assertFalse(os.path.isdir(splitter.spill_dir or ''))


if __name__ == '__main__':
    unittest.main()
