# -*- coding: utf-8 -*-
# File: state_test.py

import os
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from .common import BatchData, CacheData, LocallyShuffleData, MapData
from .raw import DataFromList


def _make_datapoints(n=50):
    # a deterministic dataset of [ndarray, scalar] datapoints
    return [[np.asarray([k, k * k, -k], dtype='int32'), k % 7] for k in range(n)]


def _make_chain(lst, shuffle=True, num_reuse=1, buffer_size=10, shuffle_interval=4, batch_size=8):
    ds = DataFromList(lst, shuffle=shuffle)
    ds = MapData(ds, lambda dp: [dp[0] + 1, dp[1] * 2])
    ds = LocallyShuffleData(ds, buffer_size, num_reuse=num_reuse, shuffle_interval=shuffle_interval)
    ds = BatchData(ds, batch_size)
    return ds


def _take_batches(ds, n):
    """ Take n batches from the dataflow. """
    ret = []
    itr = iter(ds)
    for _ in range(n):
        ret.append(next(itr))
    return ret


def _assert_datapoints_equal(testcase, dps1, dps2):
    testcase.assertEqual(len(dps1), len(dps2))
    for d1, d2 in zip(dps1, dps2):
        testcase.assertEqual(len(d1), len(d2))
        for c1, c2 in zip(d1, d2):
            testcase.assertTrue(np.array_equal(c1, c2))


# Rebuilds the chain of _make_chain(lst, shuffle=True, num_reuse=2) in a new
# process, restores the state from a file, and dumps the next batches.
# Takes the state file and the output file from sys.argv.
_RESTORE_SCRIPT = '''
import pickle
import sys

import numpy as np

from dataflow import BatchData, DataFromList, LocallyShuffleData, MapData

state_file, out_file = sys.argv[1], sys.argv[2]
with open(state_file, "rb") as f:
    state = pickle.load(f)

lst = [[np.asarray([k, k * k, -k], dtype="int32"), k % 7] for k in range(50)]
ds = DataFromList(lst, shuffle=True)
ds = MapData(ds, lambda dp: [dp[0] + 1, dp[1] * 2])
ds = LocallyShuffleData(ds, 10, num_reuse=2, shuffle_interval=4)
ds = BatchData(ds, 8)
ds.reset_state()
ds.set_state(state)

batches = []
for b in ds:
    batches.append(b)
    if len(batches) == 6:
        break
with open(out_file, "wb") as f:
    pickle.dump(batches, f)
'''


class ChainStateTest(unittest.TestCase):

    def test_save_restore_chain(self):
        # a chain saved at a batch boundary and restored into a new dataflow
        # produces exactly the same stream as the uninterrupted one
        for shuffle in (True, False):
            for num_reuse in (1, 2, 3):
                with self.subTest(shuffle=shuffle, num_reuse=num_reuse):
                    ds = _make_chain(_make_datapoints(), shuffle=shuffle, num_reuse=num_reuse)
                    ds.reset_state()
                    _take_batches(ds, 7)
                    state = ds.get_state()
                    # the state must be picklable, so that it can be
                    # transferred to a new process
                    state = pickle.loads(pickle.dumps(state))
                    expected = _take_batches(ds, 11)

                    # a new dataflow with the same data and options, like in a new process
                    ds2 = _make_chain(_make_datapoints(), shuffle=shuffle, num_reuse=num_reuse)
                    ds2.reset_state()
                    ds2.set_state(state)
                    actual = _take_batches(ds2, 11)
                    _assert_datapoints_equal(self, expected, actual)

    def test_restore_then_save_again(self):
        # a restored dataflow can be saved and restored again
        ds = _make_chain(_make_datapoints(), num_reuse=2)
        ds.reset_state()
        _take_batches(ds, 4)
        state = ds.get_state()
        expected = _take_batches(ds, 9)

        ds2 = _make_chain(_make_datapoints(), num_reuse=2)
        ds2.reset_state()
        ds2.set_state(state)
        _assert_datapoints_equal(self, expected[:4], _take_batches(ds2, 4))
        state2 = pickle.loads(pickle.dumps(ds2.get_state()))
        expected2 = _take_batches(ds2, 5)
        _assert_datapoints_equal(self, expected[4:], expected2)

        ds3 = _make_chain(_make_datapoints(), num_reuse=2)
        ds3.reset_state()
        ds3.set_state(state2)
        _assert_datapoints_equal(self, expected2, _take_batches(ds3, 5))

    def test_datafromlist_mid_pass(self):
        lst = _make_datapoints(10)
        ds = DataFromList(lst, shuffle=True)
        ds.reset_state()
        itr = iter(ds)
        for _ in range(4):
            next(itr)
        state = pickle.loads(pickle.dumps(ds.get_state()))
        rest = list(itr)    # the remaining 6 datapoints of this pass
        pass2 = list(ds)    # the next pass

        ds2 = DataFromList(lst, shuffle=True)
        ds2.reset_state()
        ds2.set_state(state)
        _assert_datapoints_equal(self, rest, list(ds2))
        _assert_datapoints_equal(self, pass2, list(ds2))

    def test_datafromlist_pass_boundary(self):
        # state saved exactly at the end of a pass
        lst = _make_datapoints(10)
        ds = DataFromList(lst, shuffle=True)
        ds.reset_state()
        list(ds)    # one full pass
        self.assertEqual(ds.get_state()['state']['pos'], 10)
        state = pickle.loads(pickle.dumps(ds.get_state()))
        pass2 = list(ds)

        ds2 = DataFromList(lst, shuffle=True)
        ds2.reset_state()
        ds2.set_state(state)
        # the pass that was current when the state was saved is already exhausted
        self.assertEqual(list(ds2), [])
        # the following pass matches the uninterrupted run
        _assert_datapoints_equal(self, pass2, list(ds2))

    def test_batchdata_holder_state(self):
        # a state can also be restored in the middle of a batch:
        # unbatched datapoints are kept in the state
        lst = _make_datapoints(10)
        ds = BatchData(DataFromList(lst, shuffle=False), 4)
        ds.reset_state()
        itr = iter(ds)
        next(itr)   # consumes datapoints 0-3
        state = ds.get_state()
        # craft a state as if it was saved in the middle of a batch:
        # datapoints 4 and 5 were consumed from the child and wait in the holder
        state['child']['state']['pos'] = 6
        state['state']['holder'] = [lst[4], lst[5]]

        ds2 = BatchData(DataFromList(lst, shuffle=False), 4)
        ds2.reset_state()
        ds2.set_state(state)
        batch = next(iter(ds2))
        expected = BatchData.aggregate_batch([lst[4], lst[5], lst[6], lst[7]])
        _assert_datapoints_equal(self, [expected], [batch])

    def test_restore_validates_data_and_config(self):
        lst = _make_datapoints()
        ds = _make_chain(lst)
        ds.reset_state()
        _take_batches(ds, 3)
        state = ds.get_state()

        def restored_chain(data=None, **kwargs):
            ds2 = _make_chain(_make_datapoints() if data is None else data, **kwargs)
            ds2.reset_state()
            return ds2

        # same data and same options: accepted
        restored_chain().set_state(state)

        # different options at every layer of the chain: rejected
        for kwargs in ({'batch_size': 4}, {'buffer_size': 20}, {'num_reuse': 2},
                       {'shuffle_interval': 5}, {'shuffle': False}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                restored_chain(**kwargs).set_state(state)

        # different data content: rejected
        bad_data = _make_datapoints()
        bad_data[3][1] += 1
        with self.assertRaises(ValueError):
            restored_chain(data=bad_data).set_state(state)

        # different data size: rejected
        with self.assertRaises(ValueError):
            restored_chain(data=_make_datapoints(60)).set_state(state)

        # a state saved by a different layer of the chain: rejected
        inner_state = ds.ds.get_state()     # the state of the LocallyShuffleData layer
        with self.assertRaises(ValueError):
            restored_chain().set_state(inner_state)

        # a corrupted position in the saved state: rejected
        bad_state = ds.get_state()
        bad_state['child']['child']['child']['state']['pos'] = 10 ** 6
        with self.assertRaises(ValueError):
            restored_chain().set_state(bad_state)

        # a corrupted shuffle buffer in the saved state: rejected
        bad_state = ds.get_state()
        buffer = bad_state['child']['state']['buffer']
        bad_state['child']['state']['buffer'] = buffer + buffer
        with self.assertRaises(ValueError):
            restored_chain().set_state(bad_state)

    def test_reset_state_old_usage(self):
        # the old usage (reset_state once, then iterate) keeps working
        ds = _make_chain(_make_datapoints())
        ds.reset_state()
        batches = _take_batches(ds, 5)
        self.assertEqual(len(batches), 5)
        for b in batches:
            self.assertEqual(b[0].shape, (8, 3))
            self.assertEqual(b[1].shape, (8,))

        # reset_state can be called again to reinitialize the dataflow
        ds.reset_state()
        self.assertEqual(len(_take_batches(ds, 5)), 5)

        # reset_state after set_state discards the restored state
        ds2 = _make_chain(_make_datapoints())
        ds2.reset_state()
        ds2.set_state(ds.get_state())
        ds2.reset_state()
        self.assertEqual(ds2.get_state()['child']['child']['child']['state']['pos'], 0)
        self.assertEqual(len(_take_batches(ds2, 3)), 3)

        # DataFromList without shuffle does not require reset_state (old behavior)
        self.assertEqual(len(list(DataFromList(_make_datapoints(10), shuffle=False))), 10)
        # neither does a BatchData on top of it
        self.assertEqual(len(list(BatchData(DataFromList(_make_datapoints(10), shuffle=False), 4))), 2)

    def test_batchdata_repeated_iteration(self):
        # BatchData over a finite dataflow can be iterated repeatedly, and
        # every pass restarts the underlying dataflow (old behavior)
        lst = _make_datapoints(20)
        ds = BatchData(DataFromList(lst, shuffle=False), 8)
        ds.reset_state()
        for _ in range(2):
            self.assertEqual(len(list(ds)), 2)  # 20 = 8 + 8 + 4 (dropped)

        ds = BatchData(DataFromList(lst, shuffle=False), 8, remainder=True)
        ds.reset_state()
        for _ in range(2):
            batches = list(ds)
            self.assertEqual(len(batches), 3)
            self.assertEqual(batches[-1][0].shape[0], 4)    # the remainder batch

    def test_state_not_implemented(self):
        # dataflows that do not support state snapshots fail loudly
        ds = CacheData(DataFromList(_make_datapoints(10), shuffle=False))
        ds.reset_state()
        with self.assertRaises(NotImplementedError):
            ds.get_state()
        with self.assertRaises(NotImplementedError):
            ds.set_state({})

    def test_restore_in_new_process(self):
        # the scenario from the docs: save in one process, resume in a new one
        ds = _make_chain(_make_datapoints(), shuffle=True, num_reuse=2)
        ds.reset_state()
        _take_batches(ds, 5)
        state = ds.get_state()
        expected = _take_batches(ds, 6)

        with tempfile.TemporaryDirectory() as d:
            state_file = os.path.join(d, 'state.pkl')
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
            out_file = os.path.join(d, 'out.pkl')
            subprocess.check_call(
                [sys.executable, '-I', '-c', _RESTORE_SCRIPT, state_file, out_file],
                cwd=d)
            with open(out_file, 'rb') as f:
                actual = pickle.load(f)
        _assert_datapoints_equal(self, expected, actual)


if __name__ == '__main__':
    unittest.main()
