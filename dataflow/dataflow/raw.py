# -*- coding: utf-8 -*-
# File: raw.py


import copy
import hashlib
import pickle
import numpy as np
import six

from .base import DataFlow, RNGDataFlow

__all__ = ['FakeData', 'DataFromQueue', 'DataFromList', 'DataFromGenerator', 'DataFromIterable']


class _HashFile(object):
    """ A file-like object that hashes everything written to it. """
    def __init__(self):
        self._hash = hashlib.sha256()

    def write(self, data):
        self._hash.update(data)

    def hexdigest(self):
        return self._hash.hexdigest()


def _data_fingerprint(lst):
    """
    A content fingerprint of a list of datapoints. Used to verify that a
    dataflow being restored is backed by the same data as the dataflow that
    saved the state.
    """
    f = _HashFile()
    pickle.Pickler(f, protocol=2).dump(lst)
    return f.hexdigest()


class FakeData(RNGDataFlow):
    """ Generate fake data of given shapes"""

    def __init__(self, shapes, size=1000, random=True, dtype='float32', domain=(0, 1)):
        """
        Args:
            shapes (list): a list of lists/tuples. Shapes of each component.
            size (int): size of this DataFlow.
            random (bool): whether to randomly generate data every iteration.
                Note that merely generating the data could sometimes be time-consuming!
            dtype (str or list): data type as string, or a list of data types.
            domain (tuple or list): (min, max) tuple, or a list of such tuples
        """
        super(FakeData, self).__init__()
        self.shapes = shapes
        self._size = int(size)
        self.random = random
        self.dtype = [dtype] * len(shapes) if isinstance(dtype, six.string_types) else dtype
        self.domain = [domain] * len(shapes) if isinstance(domain, tuple) else domain
        assert len(self.dtype) == len(self.shapes)
        assert len(self.domain) == len(self.domain)

    def __len__(self):
        return self._size

    def __iter__(self):
        if self.random:
            for _ in range(self._size):
                val = []
                for k in range(len(self.shapes)):
                    v = self.rng.rand(*self.shapes[k]) * (self.domain[k][1] - self.domain[k][0]) + self.domain[k][0]
                    val.append(v.astype(self.dtype[k]))
                yield val
        else:
            val = []
            for k in range(len(self.shapes)):
                v = self.rng.rand(*self.shapes[k]) * (self.domain[k][1] - self.domain[k][0]) + self.domain[k][0]
                val.append(v.astype(self.dtype[k]))
            for _ in range(self._size):
                yield copy.copy(val)


class DataFromQueue(DataFlow):
    """ Produce data from a queue """
    def __init__(self, queue):
        """
        Args:
            queue (queue): a queue with ``get()`` method.
        """
        self.queue = queue

    def __iter__(self):
        while True:
            yield self.queue.get()


class DataFromList(RNGDataFlow):
    """ Wrap a list of datapoints to a DataFlow"""

    def __init__(self, lst, shuffle=True):
        """
        Args:
            lst (list): input list. Each element is a datapoint.
            shuffle (bool): shuffle data.

        Note:
            The input list should not be modified after the dataflow is in
            use, otherwise the produced data and the saved state may be
            inconsistent.
        """
        super(DataFromList, self).__init__()
        self.lst = lst
        self.shuffle = shuffle
        # Position where the next pass over the list should start from.
        # It is non-zero only right after set_state() was called, to resume
        # an interrupted pass.
        self._start_pos = 0
        # Number of datapoints already produced in the current pass.
        self._pos = 0
        # RNG state right before the shuffle of the current pass. Together
        # with `_pos` it fully determines the rest of the current pass as
        # well as the RNG evolution of all future passes.
        self._rng_state_before_pass = None
        self._data_hash = None

    def __len__(self):
        return len(self.lst)

    def reset_state(self):
        super(DataFromList, self).reset_state()
        self._start_pos = 0
        self._pos = 0
        self._rng_state_before_pass = self.rng.get_state()

    def __iter__(self):
        # Every call starts a new pass over the whole list, optionally
        # resuming an interrupted pass from `_start_pos`.
        start = self._start_pos
        self._start_pos = 0
        idxs = np.arange(len(self.lst))
        if self.shuffle:
            # Take a snapshot of the RNG before this pass consumes any
            # randomness, so that get_state() is able to resume this pass.
            self._rng_state_before_pass = self.rng.get_state()
            self.rng.shuffle(idxs)
        self._pos = start
        for k in idxs[start:]:
            self._pos += 1
            yield self.lst[k]

    def get_state(self):
        if self.shuffle and self._rng_state_before_pass is None:
            raise RuntimeError(
                "reset_state() must be called before get_state()!")
        return {
            'class': type(self).__name__,
            'config': {
                'size': len(self.lst),
                'shuffle': self.shuffle,
                'data_hash': self._fingerprint(),
            },
            'state': {
                'pos': self._pos,
                'rng': self._rng_state_before_pass,
            },
        }

    def set_state(self, state):
        self._check_state(state, {
            'size': len(self.lst),
            'shuffle': self.shuffle,
            'data_hash': self._fingerprint(),
        })
        s = state['state']
        pos = s['pos']
        if not 0 <= pos <= len(self.lst):
            raise ValueError(
                "Cannot restore dataflow state: position {} is out of range "
                "for a list of {} datapoints!".format(pos, len(self.lst)))
        rng_state = s['rng']
        if rng_state is None:
            if self.shuffle:
                raise ValueError(
                    "Cannot restore dataflow state: "
                    "the saved state does not contain a RNG state!")
        else:
            if self.rng is None:
                self.rng = np.random.RandomState()
            self.rng.set_state(rng_state)
            self._rng_state_before_pass = rng_state
        self._pos = pos
        # The next pass to produce will resume from this position.
        self._start_pos = pos

    def _fingerprint(self):
        if self._data_hash is None:
            self._data_hash = _data_fingerprint(self.lst)
        return self._data_hash


class DataFromGenerator(DataFlow):
    """
    Wrap a generator to a DataFlow.
    The dataflow will not have length.
    """
    def __init__(self, gen):
        """
        Args:
            gen: iterable, or a callable that returns an iterable
        """
        self._gen = gen

    def __iter__(self):
        if not callable(self._gen):
            yield from self._gen
        else:
            yield from self._gen()

    def __len__(self):
        return len(self._gen)


class DataFromIterable(DataFlow):
    """ Wrap an iterable of datapoints to a DataFlow"""
    def __init__(self, iterable):
        """
        Args:
            iterable: an iterable object
        """
        self._itr = iterable
        try:
            self._len = len(iterable)
        except Exception:
            self._len = None

    def __len__(self):
        if self._len is None:
            raise NotImplementedError
        return self._len

    def __iter__(self):
        yield from self._itr
