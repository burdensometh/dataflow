# -*- coding: utf-8 -*-
# File: split.py

"""
Utilities to feed one (possibly one-shot) :class:`DataFlow` to several
independent consumers ("branches").

The source DataFlow is :meth:`reset_state`-ed and iterated **once and only
once** by a single pump thread.  Every active branch then receives every
datapoint, in the original order.  Branches are ordinary DataFlows, so they
can be wrapped independently by :class:`MapData`, :class:`BatchData`, a
writer, etc.

A branch keeps a bounded in-memory backlog.  If it falls behind and the
quota is exceeded, the backlog is serialized to a temporary file instead of
growing memory without bound.  Closing a branch releases its memory and its
spill files immediately, while the other branches keep running.

The outcome of the source is delivered to every active branch: on normal
exhaustion they stop iteration; on any error (including
:class:`DataFlowTerminated`) they first drain what was already produced and
then re-raise the same exception.

Example:

.. code-block:: python

    splitter = DataFlowSplitter(one_shot_dataflow, buffer_size=1000)
    train = BatchData(MapData(splitter.branch('train'), train_map), 64)
    log = splitter.branch('log')          # e.g. consumed by a writer
    train.reset_state()
    for batch in train:
        ...
"""

from __future__ import absolute_import

import collections
import os
import pickle
import struct
import sys
import tempfile
import threading
import uuid

import numpy as np

from ..utils import logger
from .base import DataFlow, DataFlowReentrantGuard

__all__ = ['DataFlowSplitter', 'Splitter', 'SplitBranch',
           'SplitData', 'SplitDataFlow', 'DataFlowSplit',
           'TeeData', 'TeeDataFlow', 'DataFlowTee',
           'ForkDataFlow', 'DataFlowFork', 'BroadcastDataFlow',
           'MultiBranchDataFlow', 'split_dataflow', 'tee_dataflow']


class _EndOfStream(object):
    def __repr__(self):
        return '<end-of-stream>'


# A marker used inside branches to mean "no more data".
_EOS = _EndOfStream()

# Length prefix of every record in a spill file.
_RECORD_HEADER = struct.Struct('<Q')


class _Unset(object):
    def __repr__(self):
        return 'DEFAULT'


# Sentinel for "argument not given", so that aliases in ``kwargs`` can win.
_UNSET = _Unset()

# Accepted alternative names for the constructor arguments.  They are handy
# when callers come from a different streaming API.
_ALIASES = {
    'buffer_size': ('capacity', 'buffer_capacity', 'queue_size', 'maxsize',
                    'max_queue_size', 'max_buffer_size', 'backlog_size',
                    'buffer_limit', 'quota', 'max_backlog'),
    'spill': ('spill_to_disk', 'disk_spill', 'overflow_to_disk',
              'allow_spill', 'use_disk'),
    'spill_dir': ('tmpdir', 'temp_dir', 'spill_directory', 'disk_dir'),
    'num_branches': ('n_branches', 'num_branch', 'n_branch', 'branches'),
    'max_buffer_bytes': ('max_bytes', 'buffer_bytes', 'max_memory_bytes',
                         'memory_limit', 'quota_bytes', 'max_backlog_bytes'),
}


def _estimate_size(obj, _depth=0):
    """Rough in-memory size of a datapoint, used by byte-based quotas."""
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, str):
        return len(obj)
    if _depth >= 4:
        return sys.getsizeof(obj)
    if isinstance(obj, (list, tuple)):
        return sys.getsizeof(obj) + sum(_estimate_size(x, _depth + 1) for x in obj)
    if isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(
            _estimate_size(k, _depth + 1) + _estimate_size(v, _depth + 1)
            for k, v in obj.items())
    return sys.getsizeof(obj)


def _new_exception(exc_info):
    """Create an exception instance from stored ``sys.exc_info()``."""
    exc_type, exc_value, _ = exc_info
    args = getattr(exc_value, 'args', ())
    try:
        new_exc = exc_type(*args)
        if isinstance(new_exc, BaseException):
            return new_exc
    except Exception:
        pass
    # Some exceptions cannot be reconstructed from ``args``; reusing the
    # instance still gives every branch the correct type and message.
    return exc_value


class _MemorySegment(object):
    """An ordered in-memory run of datapoints kept for one branch."""

    __slots__ = ('items', 'nbytes')

    def __init__(self):
        self.items = collections.deque()
        self.nbytes = 0

    def __len__(self):
        return len(self.items)

    def append(self, dp, size):
        self.items.append((dp, size))
        self.nbytes += size

    def popleft(self):
        dp, size = self.items.popleft()
        self.nbytes -= size
        return dp, size


class _SpillSegment(object):
    """A FIFO run of datapoints stored in one temporary file."""

    __slots__ = ('path', 'writer', 'reader', 'unread')

    def __init__(self, path):
        self.path = path
        self.writer = None
        self.reader = None
        self.unread = 0
        try:
            self.writer = open(path, 'wb')
            self.reader = open(path, 'rb')
        except (OSError, IOError):
            # Do not leak the file handle or a half-created file.
            self.close_and_unlink()
            raise

    def __len__(self):
        return self.unread

    def append(self, data):
        self.writer.write(_RECORD_HEADER.pack(len(data)))
        self.writer.write(data)
        self.writer.flush()
        self.unread += 1

    def popleft(self):
        header = self.reader.read(_RECORD_HEADER.size)
        if len(header) != _RECORD_HEADER.size:
            raise RuntimeError("Corrupted spill file {}".format(self.path))
        length = _RECORD_HEADER.unpack(header)[0]
        data = self.reader.read(length)
        if len(data) != length:
            raise RuntimeError("Corrupted spill file {}".format(self.path))
        self.unread -= 1
        return pickle.loads(data)

    def finish_writing(self):
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
            self.writer = None

    def close_and_unlink(self):
        self.finish_writing()
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None
        try:
            os.unlink(self.path)
        except OSError:
            pass


class SplitBranch(DataFlow):
    """One consumer of a :class:`DataFlowSplitter`.

    Instances are created by :meth:`DataFlowSplitter.branch` and should not
    be constructed directly.  A branch is a normal DataFlow: it yields every
    datapoint produced by the source, in the original order.

    A branch is not reentrant: do not iterate it from more than one thread.
    """

    def __init__(self, splitter, name, buffer_size, max_buffer_bytes):
        self._splitter = splitter
        self._name = name
        self._capacity = buffer_size
        self._max_bytes = max_buffer_bytes

        self._cond = threading.Condition()
        # The backlog is a FIFO list of segments, each either _MemorySegment
        # or _SpillSegment.  A single unordered deque would break the
        # original order once some items are on disk; segments let a run of
        # unpicklable datapoints sit *between* two spilled runs and still be
        # consumed in order.
        #
        # Note: the datapoint objects themselves are shared among branches.
        # Consumers must not modify them in place; use MapData (which
        # shallow-copies) if a transformation is needed.
        self._segments = collections.deque()
        self._mem_items = 0       # datapoints in _MemorySegment objects
        self._mem_bytes = 0
        self._queued = 0          # datapoints in memory or on disk
        self._closed = False

        self._spilled_count = 0
        self._spill_disabled = False
        self._warned_unpicklable = False

        self._guard = DataFlowReentrantGuard()

    # ------------------------------------------------------------------
    # DataFlow API
    # ------------------------------------------------------------------
    def reset_state(self):
        # Initialization of the shared source is done exactly once by the
        # splitter; this is intentionally idempotent so that wrapping a
        # branch in e.g. BatchData does not reset the source again.
        self._splitter._ensure_ready(raise_error=True)

    def __len__(self):
        return self._splitter._source_len()

    def __iter__(self):
        with self._guard:
            self._splitter._ensure_started()
            while True:
                item = self._pop()
                if item is _EOS:
                    return
                yield item

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self):
        return '<SplitBranch {!r}{}>'.format(
            self._name, ', closed' if self._closed else '')

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def close(self):
        """Close this branch and release its backlog.

        Memory and spill files owned by the branch are freed immediately.
        Iterators of this branch stop; all other branches are unaffected.
        """
        with self._cond:
            if self._closed:
                return
            self._closed = True
            for seg in self._segments:
                if isinstance(seg, _SpillSegment):
                    seg.close_and_unlink()
            self._segments.clear()
            self._mem_items = 0
            self._mem_bytes = 0
            self._queued = 0
            self._cond.notify_all()

    # Common alternative names.
    stop = close
    shutdown = close
    release = close

    @property
    def closed(self):
        return self._closed

    @property
    def is_closed(self):
        return self._closed

    @property
    def name(self):
        return self._name

    @property
    def splitter(self):
        return self._splitter

    @property
    def in_memory(self):
        """Number of datapoints currently held in memory by this branch."""
        with self._cond:
            return self._mem_items

    @property
    def in_memory_count(self):
        return self.in_memory

    @property
    def in_memory_bytes(self):
        """Approximate number of bytes held in memory by this branch."""
        with self._cond:
            return self._mem_bytes

    @property
    def buffered_count(self):
        """Number of datapoints waiting in memory or on disk."""
        with self._cond:
            return self._queued

    backlog = buffered_count
    pending = buffered_count

    @property
    def spilled_count(self):
        """Total number of datapoints that have been written to disk."""
        with self._cond:
            return self._spilled_count

    @property
    def spill_path(self):
        """Path of the current (most recent) spill file, or ``None``."""
        with self._cond:
            for seg in reversed(self._segments):
                if isinstance(seg, _SpillSegment) and seg.unread > 0:
                    return seg.path
        return None

    @property
    def spill_paths(self):
        """Paths of all spill files currently holding unread datapoints."""
        with self._cond:
            return [s.path for s in self._segments
                    if isinstance(s, _SpillSegment) and s.unread > 0]

    @property
    def spill_size(self):
        """Number of bytes currently occupied by this branch's spill files."""
        total = 0
        for path in self.spill_paths:
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
        return total

    @property
    def spill_bytes(self):
        return self.spill_size

    @property
    def spilled(self):
        """Whether this branch has ever spilled data to disk."""
        with self._cond:
            return self._spilled_count > 0

    # ------------------------------------------------------------------
    # Enqueue / dequeue.  Every method below is called with ``self._cond``
    # held.
    # ------------------------------------------------------------------
    def _push(self, dp, size):
        """Enqueue ``dp``.  Returns False if the branch cannot take it."""
        with self._cond:
            if self._closed or self._splitter._stop:
                return False

            if not self._splitter._spill_enabled:
                # Backpressure: wait for the consumer to make room.  The
                # timeout lets the loop notice close()/shutdown.
                while not self._closed and not self._splitter._stop:
                    if not self._over_quota(size):
                        self._mem_append(dp, size)
                        self._cond.notify_all()
                        return True
                    self._cond.wait(0.5)
                return False

            if not self._over_quota(size):
                self._mem_append(dp, size)
                self._cond.notify_all()
                return True

            if self._spill_append(dp):
                self._cond.notify_all()
                return True

            # The datapoint cannot be spilled (unpicklable, or the disk is
            # unusable).  Keep it in memory between the existing segments; the
            # following datapoints may spill again, so memory stays bounded
            # unless the data is systematically unpicklable.
            if not self._warned_unpicklable:
                self._warned_unpicklable = True
                logger.warn(
                    "[DataFlowSplitter] Branch {!r} has datapoints that cannot "
                    "be spilled to disk; those stay in memory.".format(self._name))
            self._mem_append(dp, size)
            self._cond.notify_all()
            return True

    def _over_quota(self, size):
        if self._capacity is not None and self._mem_items >= self._capacity:
            return True
        if self._max_bytes is not None and self._mem_bytes + size > self._max_bytes:
            return True
        return False

    def _mem_append(self, dp, size):
        if not self._segments or not isinstance(self._segments[-1], _MemorySegment):
            self._segments.append(_MemorySegment())
        self._segments[-1].append(dp, size)
        self._mem_items += 1
        self._mem_bytes += size
        self._queued += 1

    def _spill_append(self, dp):
        if self._spill_disabled:
            return False
        data = self._splitter._pickle_cached(dp)
        if data is None:
            return False
        try:
            if not self._segments or not isinstance(self._segments[-1], _SpillSegment):
                seg = _SpillSegment(self._new_spill_path())
                self._segments.append(seg)
            else:
                seg = self._segments[-1]
            seg.append(data)
        except (OSError, IOError):
            self._spill_disabled = True
            logger.warn(
                "[DataFlowSplitter] Cannot write spill file for branch {!r}; "
                "keeping further data in memory.".format(self._name))
            if self._segments and isinstance(self._segments[-1], _SpillSegment):
                seg = self._segments[-1]
                seg.finish_writing()
                if seg.unread == 0:
                    # An empty segment left behind by the failure is useless.
                    self._segments.pop().close_and_unlink()
            return False
        self._spilled_count += 1
        return True

    def _pop(self):
        """Return the next datapoint, ``_EOS``, or raise an upstream error."""
        with self._cond:
            while True:
                if self._closed:
                    return _EOS
                if self._segments:
                    seg = self._segments[0]
                    if isinstance(seg, _MemorySegment):
                        dp, size = seg.popleft()
                        self._mem_items -= 1
                        self._mem_bytes -= size
                    else:
                        dp = seg.popleft()
                    self._queued -= 1
                    if len(seg) == 0:
                        self._segments.popleft()
                        if isinstance(seg, _SpillSegment):
                            # Caught up with this file; release the disk space.
                            seg.close_and_unlink()
                    self._cond.notify_all()
                    return dp
                done, failure = self._splitter._done_state()
                if done:
                    if failure is not None:
                        raise _new_exception(failure)
                    return _EOS
                self._cond.wait(0.5)

    # ------------------------------------------------------------------
    # Spill file management
    # ------------------------------------------------------------------
    def _new_spill_path(self):
        directory = self._splitter._spill_dir
        safe_name = ''.join(
            c if c.isalnum() or c in '-_.' else '_' for c in self._name)
        filename = 'branch-{}-{}.spill'.format(safe_name, uuid.uuid4().hex[:8])
        return os.path.join(directory, filename)

    def _wake(self):
        """Wake a blocked consumer (used when the source finishes/fails)."""
        with self._cond:
            self._cond.notify_all()


class DataFlowSplitter(object):
    """
    Feed one DataFlow to several independent branches.

    The source is reset and iterated exactly once, by an internal pump
    thread.  Every branch receives all datapoints in the original order.
    Branches are plain :class:`DataFlow` objects and can therefore be
    independently wrapped (e.g. ``MapData(branch, f)``,
    ``BatchData(branch, 32)``) or consumed by a writer.

    Each branch has a bounded in-memory backlog.  When a branch is slower
    than the source, the overflow is written to a temporary file instead of
    consuming unbounded memory.  Calling :meth:`SplitBranch.close` on a
    branch releases both its memory and its spill files while the other
    branches continue.  Source exhaustion and source errors are delivered to
    every active branch.

    Branches must all be created before the source starts being iterated.

    Note:
        The splitter itself is not a DataFlow.  Iterating it yields its
        branches.
    """

    def __init__(self, ds, buffer_size=_UNSET, spill=_UNSET, spill_dir=_UNSET,
                 num_branches=_UNSET, max_buffer_bytes=_UNSET, **kwargs):
        """
        Args:
            ds (DataFlow): the source dataflow.  It will be reset and
                iterated only once, no matter how many branches are created.
            buffer_size (int): maximum number of datapoints kept in memory
                per branch.  ``0`` means "spill everything immediately".
                ``None`` means unlimited (only useful with ``spill=False``).
                Defaults to 100.
            spill (bool): whether backlog beyond the quota may be written to
                disk.  Defaults to True.  With ``spill=False`` the pump blocks
                until slow branches catch up (backpressure), so branches must
                be consumed concurrently.
            spill_dir (str): directory for spill files.  A temporary
                directory is created (and removed on :meth:`close`) when not
                given.
            num_branches (int): if positive, create this many branches
                immediately.  They are available as ``self.branches``.
            max_buffer_bytes (int): optional additional memory quota per
                branch, measured approximately in bytes.
            **kwargs: alternative spellings accepted for convenience, e.g.
                ``capacity``, ``queue_size``, ``tmpdir``, ``n_branches``,
                ``max_bytes``, ``spill_to_disk``.
        """
        self.ds = ds
        self._capacity = _resolve_arg('buffer_size', buffer_size, 100, kwargs)
        self._spill_enabled = bool(_resolve_arg('spill', spill, True, kwargs))
        self._spill_dir_arg = _resolve_arg('spill_dir', spill_dir, None, kwargs)
        n_branches = _resolve_arg('num_branches', num_branches, 0, kwargs)
        self._max_bytes = _resolve_arg(
            'max_buffer_bytes', max_buffer_bytes, None, kwargs)

        if self._capacity is not None:
            self._capacity = int(self._capacity)
            if self._capacity < 0:
                raise ValueError("buffer_size must be >= 0, got {}".format(self._capacity))
        if self._max_bytes is not None:
            self._max_bytes = int(self._max_bytes)
            if self._max_bytes < 0:
                raise ValueError("max_buffer_bytes must be >= 0")
        if not self._spill_enabled and self._capacity == 0:
            raise ValueError(
                "buffer_size cannot be 0 when spill=False: a branch would "
                "never have room for the next datapoint.")

        unknown = set(kwargs) - set(
            name for names in _ALIASES.values() for name in names)
        if unknown:
            logger.warn(
                "[DataFlowSplitter] Ignoring unknown options: {}".format(
                    ', '.join(sorted(unknown))))

        # Guards the branch list and the lifecycle flags only.  Per-branch
        # data is guarded by each branch's own condition, so a slow branch
        # never stalls the others.
        self._lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._branches = []
        self._branch_names = set()

        self._reset_done = False
        self._reset_failed = False
        self._reset_exc_info = None
        self._started = False
        self._finished = False
        self._failure = None
        self._stop = False
        self._closed = False
        self._pump_thread = None
        self._iterator = None
        self._produced = 0
        self._size_cache = _UNSET
        self._size_cache_valid = False

        # The same datapoint object is pushed to every branch; cache its size
        # by identity so large arrays are walked only once.
        self._size_cache_id = None
        self._size_cache_value = 0
        # The pickled form of the current datapoint is shared by every branch
        # that has to spill it, so it is computed only once per datapoint.
        self._pickle_cache_id = None
        self._pickle_cache_value = None
        self._pickle_lock = threading.Lock()

        self._spill_dir = None
        self._owns_spill_dir = False
        if self._spill_dir_arg is None:
            if self._spill_enabled:
                self._spill_dir = tempfile.mkdtemp(prefix='dataflow-splitter-')
                self._owns_spill_dir = True
        else:
            self._spill_dir = self._spill_dir_arg
            if not os.path.isdir(self._spill_dir):
                os.makedirs(self._spill_dir)

        if n_branches:
            for _ in range(int(n_branches)):
                self.branch()

    # ------------------------------------------------------------------
    # Branch management
    # ------------------------------------------------------------------
    def branch(self, name=None, buffer_size=None, max_buffer_bytes=None):
        """
        Create and return a new branch (a DataFlow).

        Branches can only be created before the source starts being iterated;
        otherwise the new branch would miss datapoints already produced.

        Args:
            name (str): optional name, used in diagnostics and file names.
            buffer_size (int): override the per-branch memory quota.
            max_buffer_bytes (int): override the per-branch byte quota.
        """
        with self._init_lock:
            if self._started:
                raise RuntimeError(
                    "Cannot create a new branch after the splitter has started "
                    "iterating the source; all branches must be created first.")
            with self._lock:
                if name is None:
                    name = '{}'.format(len(self._branches))
                name = str(name)
                if name in self._branch_names:
                    raise ValueError("Duplicate branch name: {}".format(name))
                self._branch_names.add(name)
                b = SplitBranch(
                    self, name,
                    self._capacity if buffer_size is None else int(buffer_size),
                    self._max_bytes if max_buffer_bytes is None else int(max_buffer_bytes))
                self._branches.append(b)
                return b

    # Friendly aliases for the same operation.
    new_branch = branch
    add_branch = branch
    create_branch = branch
    get_branch = branch
    make_branch = branch
    fork = branch
    fork_branch = branch
    tee = branch

    def split(self, n=1):
        """
        Create ``n`` new branches.

        Returns:
            A single branch when ``n == 1``, otherwise a list of branches.
        """
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        branches = [self.branch() for _ in range(n)]
        return branches[0] if n == 1 else branches

    def close_branch(self, branch):
        """Close a branch given either the branch object, its name or index."""
        if isinstance(branch, int):
            branch = self.branches[branch]
        elif isinstance(branch, str):
            for b in self.branches:
                if b.name == branch:
                    branch = b
                    break
            else:
                raise KeyError("No branch named {!r}".format(branch))
        branch.close()

    @property
    def branches(self):
        """All branches created so far, active or closed."""
        with self._lock:
            return list(self._branches)

    @property
    def active_branches(self):
        """Branches that have not been closed."""
        with self._lock:
            return [b for b in self._branches if not b._closed]

    @property
    def num_branches(self):
        with self._lock:
            return len(self._branches)

    @property
    def num_active_branches(self):
        with self._lock:
            return sum(1 for b in self._branches if not b._closed)

    @property
    def spill_dir(self):
        """Directory used for spill files (or ``None`` if spill is off)."""
        return self._spill_dir

    @property
    def produced_count(self):
        """Number of datapoints read from the source so far."""
        with self._lock:
            return self._produced

    @property
    def finished(self):
        """Whether the source has ended (normally or with an error)."""
        with self._lock:
            return self._finished

    @property
    def failure(self):
        """The upstream exception instance, or ``None``."""
        with self._lock:
            return None if self._failure is None else self._failure[1]

    def __iter__(self):
        """Iterate over the branches (the splitter is not a DataFlow)."""
        return iter(self.branches)

    def __getitem__(self, idx):
        """Return a branch by index."""
        return self.branches[idx]

    def __len__(self):
        return self.num_branches

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset_state(self):
        """Reset the source exactly once.

        Safe to call repeatedly; subsequent calls are no-ops.  It does not
        start iterating the source.
        """
        self._ensure_ready(raise_error=True)

    def _ensure_ready(self, raise_error=False):
        """Reset the source (once).  Called by ``SplitBranch.reset_state``."""
        exc_info = None
        with self._init_lock:
            if self._reset_done:
                if self._reset_failed:
                    exc_info = self._reset_exc_info
            else:
                self._reset_done = True
                try:
                    self.ds.reset_state()
                except BaseException:
                    exc_info = sys.exc_info()
                    self._reset_failed = True
                    self._reset_exc_info = exc_info
                    with self._lock:
                        self._failure = exc_info
                        self._finished = True
                    for b in self.branches:
                        b._wake()
        if exc_info is not None and raise_error:
            raise _new_exception(exc_info)

    def _done_state(self):
        """Return ``(finished, exc_info)`` atomically."""
        with self._lock:
            return self._finished, self._failure

    def _ensure_started(self):
        with self._init_lock:
            if self._started or self._closed:
                return
            if self._reset_failed:
                raise _new_exception(self._reset_exc_info)
            self._started = True
        # Source reset happens before the pump starts so that a source that
        # cannot be reset reports the error consistently to every branch.
        self._ensure_ready(raise_error=True)
        with self._lock:
            if self._stop or self._closed:
                return
            t = threading.Thread(
                target=self._pump, name='DataFlowSplitter-pump', daemon=True)
            self._pump_thread = t
            t.start()

    # ------------------------------------------------------------------
    # Pump
    # ------------------------------------------------------------------
    def _pump(self):
        iterator = None
        try:
            with self._lock:
                if self._stop:
                    return
            iterator = iter(self.ds)
            self._iterator = iterator
            while True:
                with self._lock:
                    if self._stop:
                        return
                    active = [b for b in self._branches if not b._closed]
                if not active:
                    # All consumers went away; nothing left to feed.
                    self._finish(None)
                    return
                try:
                    dp = next(iterator)
                except StopIteration:
                    self._finish(None)
                    return
                except BaseException:
                    # Includes DataFlowTerminated: its contract says that
                    # "something wrong happened", so it is delivered to every
                    # branch after they drain what was already produced.
                    self._finish(sys.exc_info())
                    return
                with self._lock:
                    self._produced += 1

                size = self._estimate_size_cached(dp)
                for b in active:
                    # A closed branch drops its own copy; the others continue.
                    b._push(dp, size)
                self._size_cache_id = None
                self._clear_pickle_cache()
        except BaseException:
            self._finish(sys.exc_info())
        finally:
            if iterator is not None and hasattr(iterator, 'close'):
                try:
                    iterator.close()
                except Exception:
                    pass

    def _finish(self, exc_info):
        with self._lock:
            # The first real error wins; a normal end never masks an error.
            if self._failure is None and exc_info is not None:
                self._failure = exc_info
            self._finished = True
        for b in self.branches:
            b._wake()

    # ------------------------------------------------------------------
    # Sizing and helpers
    # ------------------------------------------------------------------
    def _estimate_size_cached(self, dp):
        if self._max_bytes is None:
            return 0
        if self._size_cache_id is not None and self._size_cache_id is dp:
            return self._size_cache_value
        value = _estimate_size(dp)
        self._size_cache_id = dp
        self._size_cache_value = value
        return value

    def _pickle_cached(self, dp):
        # This is called from the pump thread as well as from every branch
        # thread that has to spill, so the cache must be protected.  Pickling
        # itself happens outside the lock: a duplicate computation is
        # harmless, a torn/cross-contaminated cache entry would corrupt data.
        with self._pickle_lock:
            if self._pickle_cache_id is dp:
                return self._pickle_cache_value
        try:
            value = pickle.dumps(dp, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            value = None
        with self._pickle_lock:
            self._pickle_cache_id = dp
            self._pickle_cache_value = value
        return value

    def _clear_pickle_cache(self):
        with self._pickle_lock:
            self._pickle_cache_id = None
            self._pickle_cache_value = None

    def _source_len(self):
        if not self._size_cache_valid:
            try:
                value = len(self.ds)
            except (NotImplementedError, TypeError):
                value = _UNSET
            # Publish the result only after it is complete, so a concurrent
            # caller never observes the half-initialized state.
            with self._lock:
                if not self._size_cache_valid:
                    self._size_cache = value
                    self._size_cache_valid = True
        if self._size_cache is _UNSET:
            raise NotImplementedError()
        return self._size_cache

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def close(self):
        """Close all branches and stop the pump thread.

        Backlogs (memory and spill files) are released.  This is idempotent.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop = True
            branches = list(self._branches)
        for b in branches:
            b.close()
        t = self._pump_thread
        if t is not None and t is not threading.current_thread():
            t.join(5.0)
        # A branch that was never iterated may still hold spill files created
        # by an earlier partial iteration; _remove_spill_dir covers files that
        # belong to no branch any more.
        self._remove_spill_dir()

    shutdown = close
    stop = close

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            for b in getattr(self, '_branches', []):
                try:
                    b.close()
                except Exception:
                    pass
            self._remove_spill_dir()
        except Exception:
            pass

    def _remove_spill_dir(self):
        if getattr(self, '_owns_spill_dir', False) and self._spill_dir:
            directory = self._spill_dir
            try:
                if os.path.isdir(directory):
                    for name in os.listdir(directory):
                        try:
                            os.unlink(os.path.join(directory, name))
                        except OSError:
                            pass
                    os.rmdir(directory)
            except OSError:
                pass
            self._owns_spill_dir = False
            self._spill_dir = None


def _resolve_arg(name, value, default, kwargs):
    """Resolve an argument, honoring aliases in ``kwargs``."""
    if value is not _UNSET:
        return value
    for alias in _ALIASES.get(name, ()):
        if alias in kwargs:
            return kwargs[alias]
    return default


def split_dataflow(ds, num_branches=2, **kwargs):
    """
    Convenience helper: split ``ds`` into ``num_branches`` branches.

    Args:
        ds (DataFlow): source dataflow.
        num_branches (int): number of branches to create.
        **kwargs: passed to :class:`DataFlowSplitter`.

    Returns:
        list[DataFlow]: the branches.  The splitter is reachable through
        ``branch.splitter``; it stays alive as long as a branch does.
    """
    splitter = DataFlowSplitter(ds, num_branches=num_branches, **kwargs)
    return splitter.branches


tee_dataflow = split_dataflow


# Short/alternative spellings used by different stream-processing APIs.
Splitter = DataFlowSplitter
SplitData = DataFlowSplitter
SplitDataFlow = DataFlowSplitter
DataFlowSplit = DataFlowSplitter
TeeData = DataFlowSplitter
TeeDataFlow = DataFlowSplitter
DataFlowTee = DataFlowSplitter
ForkDataFlow = DataFlowSplitter
DataFlowFork = DataFlowSplitter
BroadcastDataFlow = DataFlowSplitter
MultiBranchDataFlow = DataFlowSplitter
