# -*- coding: utf-8 -*-
# File: serialize.py

import hashlib
import numpy as np
import os
import platform
import struct
import types
from collections import defaultdict

from ..utils import logger
from ..utils.serialize import dumps, loads
from ..utils.develop import create_dummy_class  # noqa
from ..utils.utils import get_tqdm
from .base import DataFlow, ProxyDataFlow
from .common import FixedSizeData, MapData
from .format import HDF5Data, LMDBData, IncompleteLMDBError, LMDB_KEYS_KEY, LMDB_META_KEY, read_lmdb_meta
from .raw import DataFromGenerator, DataFromList

__all__ = ['LMDBSerializer', 'LMDBResumeError', 'IncompleteLMDBError',
           'NumpySerializer', 'TFRecordSerializer', 'HDF5Serializer']

# Bump this if the on-disk layout of a resumable export changes.
_LMDB_RESUME_VERSION = 1
_LMDB_KEY_FORMAT = u'{:08}'


class LMDBResumeError(ValueError):
    """
    Raised when a resumable LMDB export cannot be continued safely, e.g. when
    it is not a resumable export, or when the source / export definition does
    not match the one that created the existing data.
    """
    pass


def _reset_df_and_get_size(df):
    df.reset_state()
    try:
        sz = len(df)
    except NotImplementedError:
        sz = 0
    return sz


def _lmdb_key(idx):
    return _LMDB_KEY_FORMAT.format(idx).encode('ascii')


def _default_map_size():
    # It's OK to use super large map_size on Linux, but not on other platforms
    # See: https://github.com/NVIDIA/DIGITS/issues/206
    return 1099511627776 * 2 if platform.system() == 'Linux' else 128 * 10**6


# ---------------------------------------------------------------------------
# Source / export identity
#
# A resumable export can be continued only if the input is the same one that
# produced the already-committed prefix.  Three checks are recorded and enforced:
#
# * a structural fingerprint of the DataFlow chain (class names and lengths),
#   which catches a changed export pipeline before reading any data;
# * a rolling digest over the serialized datapoints of the committed prefix.
#   On resume the source is replayed from the beginning and the digest is
#   recomputed and compared, which catches a source whose content changed even
#   though its class and length stayed the same.  It uses constant memory.
# * optional caller-supplied ``source_id`` / ``export_definition`` for inputs
#   the automatic checks cannot see through (e.g. files changed in place).
# ---------------------------------------------------------------------------
_FINGERPRINT_MAX_DEPTH = 6
_FINGERPRINT_MAX_ITEMS = 64
_FINGERPRINT_MAX_ARRAY_BYTES = 1 << 16


def _type_name(obj):
    t = type(obj)
    return '{}.{}'.format(getattr(t, '__module__', ''), getattr(t, '__qualname__', t.__name__))


def _function_name(func):
    module = getattr(func, '__module__', None)
    qualname = getattr(func, '__qualname__', None)
    if qualname is None:
        return _type_name(func)
    return '{}.{}'.format(module, qualname)


def _fingerprint_code(code, depth):
    consts = []
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            consts.append(_fingerprint_code(c, depth + 1))
        else:
            consts.append(_fingerprint_value(c, depth + 1))
    return ['code', code.co_code, consts, code.co_names, code.co_varnames,
            code.co_argcount, code.co_kwonlyargcount]


def _fingerprint_value(v, depth=0):
    """
    Describe a configuration value.  Only values that are stable across
    instances of the same pipeline are described; arbitrary object state is
    reduced to its type.
    """
    if depth > _FINGERPRINT_MAX_DEPTH:
        return ['max-depth']
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        return ['prim', type(v).__name__, v]
    if isinstance(v, np.ndarray):
        digest = None
        if v.size and v.nbytes <= _FINGERPRINT_MAX_ARRAY_BYTES:
            digest = hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
        return ['ndarray', str(v.dtype), list(v.shape), digest]
    if isinstance(v, np.generic):
        return ['npscalar', str(v.dtype), v.item()]
    if isinstance(v, dict):
        items = sorted(
            ([_fingerprint_value(k, depth + 1), _fingerprint_value(val, depth + 1)]
             for k, val in v.items()), key=repr)
        return ['dict', len(v), items[:_FINGERPRINT_MAX_ITEMS]]
    if isinstance(v, (list, tuple)):
        if len(v) > _FINGERPRINT_MAX_ITEMS:
            return [type(v).__name__, len(v)]
        return [type(v).__name__, len(v), [_fingerprint_value(x, depth + 1) for x in v]]
    if isinstance(v, types.CodeType):
        return _fingerprint_code(v, depth)
    if callable(v):
        code = getattr(v, '__code__', None)
        return ['callable', _function_name(v),
                _fingerprint_code(code, depth + 1) if code is not None else None,
                _fingerprint_value(getattr(v, '__defaults__', None), depth + 1),
                _fingerprint_value(getattr(v, '__kwdefaults__', None), depth + 1)]
    return ['object', _type_name(v)]


def _fingerprint_datasource(df):
    """
    Compute a stable fingerprint of a DataFlow definition: the class chain, the
    length of each stage, and the code of its mappers.  Stable across instances
    of the same pipeline.  Used as a cheap first check when resuming; the
    authoritative check is the prefix digest below.
    """
    chain = []
    node = df
    seen = set()
    while id(node) not in seen:
        seen.add(id(node))
        chain.append(node)
        if isinstance(node, ProxyDataFlow):
            node = node.ds
        else:
            break

    described = []
    for node in chain:
        callables = []
        try:
            items = sorted(vars(node).items())
        except TypeError:
            items = []
        ds = getattr(node, 'ds', None)
        for name, value in items:
            if name.startswith('__') or not callable(value) or value is ds:
                continue
            callables.append([name, _fingerprint_value(value)])
        try:
            length = int(len(node))
        except (NotImplementedError, TypeError):
            length = None
        described.append([_type_name(node), length, callables])
    payload = repr(described).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _digest_value(value):
    """Digest a caller-supplied identity, or None if it was not supplied."""
    if value is None:
        return None
    payload = repr(_fingerprint_value(value)).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


class _PrefixDigest(object):
    """
    Rolling digest over the serialized datapoints of a committed prefix.  Each
    blob is length-prefixed so that concatenated payloads cannot collide by
    shifting bytes between datapoints.  Uses constant memory.
    """
    def __init__(self):
        self._h = hashlib.sha256()

    def update(self, blob):
        self._h.update(struct.pack('<Q', len(blob)))
        self._h.update(blob)

    def hexdigest(self):
        return self._h.hexdigest()


def _open_resumable_lmdb(path, isdir, map_size):
    return lmdb.open(path, subdir=isdir, map_size=map_size, readonly=False,
                     create=True, meminit=False, map_async=True)


def _internal_entry_count(txn):
    return sum(1 for k in (LMDB_KEYS_KEY, LMDB_META_KEY) if txn.get(k) is not None)


def _data_entry_count(db):
    """Number of datapoints visible to a fresh read transaction."""
    with db.begin() as txn:
        return txn.stat()['entries'] - _internal_entry_count(txn)


def _grow_map(db):
    """
    Double the map size.  No write transaction may be open in this process
    when this is called.
    """
    new_size = db.info()['map_size'] * 2
    logger.info("Doubling LMDB map_size to {:.2f}MB".format(new_size / 10**6))
    db.set_mapsize(new_size)


def _ensure_map_size(db, wanted):
    """
    Make sure the map is at least ``wanted`` bytes.  ``lmdb.open`` keeps the
    size of an existing file even if a smaller one is requested, so a resume
    must grow explicitly when needed.  No write transaction may be open.
    """
    if wanted is not None and db.info()['map_size'] < wanted:
        db.set_mapsize(wanted)


def _check_finite_source(df, size):
    if size is None:
        raise LMDBResumeError(
            "LMDBSerializer.save_resume() needs a DataFlow with a finite, known length "
            "so that progress and completeness can be recorded. {} does not implement "
            "__len__().".format(_type_name(df)))


def _validate_resume_meta(db, meta, path, source_fp, source_digest, export_digest, size):
    """
    Check that the existing export can be continued with the given source.
    Returns the committed progress.

    ``meta`` is only trusted after the number of datapoints in the database is
    found to match it: data and progress are committed in the same transaction,
    so a mismatch means the database was written by something else.
    """
    def mismatch(what, old, new):
        raise LMDBResumeError(
            "Cannot resume {}: {} of the existing export differs from the current "
            "one (recorded {}, got {}). Refusing to mix data from different inputs."
            .format(path, what, old, new))

    if meta.get('version') != _LMDB_RESUME_VERSION:
        raise LMDBResumeError(
            "Cannot resume {}: resumable export format version {} is not supported "
            "(expected {}).".format(path, meta.get('version'), _LMDB_RESUME_VERSION))

    if meta.get('key_format') != _LMDB_KEY_FORMAT:
        mismatch('key format', meta.get('key_format'), _LMDB_KEY_FORMAT)

    if meta.get('source_fingerprint') != source_fp:
        mismatch('the DataFlow definition (source identity)', meta.get('source_fingerprint'), source_fp)

    if meta.get('source_id') != source_digest:
        mismatch('the source_id', meta.get('source_id'), source_digest)

    if meta.get('export_definition') != export_digest:
        mismatch('the export_definition', meta.get('export_definition'), export_digest)

    if meta.get('size') is not None and int(meta['size']) != int(size):
        mismatch('the source length', meta.get('size'), size)

    recorded = int(meta.get('progress', 0))
    actual = _data_entry_count(db)
    if actual != recorded:
        raise LMDBResumeError(
            "Cannot resume {}: the resume progress says {} datapoints but the database "
            "contains {}. It was probably modified by someone else.".format(
                path, recorded, actual))

    if meta.get('status') == 'complete' and actual != int(size):
        raise LMDBResumeError(
            "Cannot resume {}: it is marked complete with {} datapoints but the source "
            "has {}.".format(path, actual, size))
    return recorded


def _write_resumable(df, db, path, progress, size, source_fp, source_digest,
                     export_digest, write_frequency):
    """
    Write datapoints starting at ``progress``, committing data and progress
    together.  Returns the total number of datapoints in the finished export.

    The source is replayed from the beginning.  Its first ``progress``
    datapoints are re-serialized into a rolling digest and compared with the
    digest recorded in the metadata, so a changed input is detected before any
    new row is appended, using constant memory.
    """
    expected_digest = None
    if progress:
        with db.begin() as txn:
            expected_digest = loads(txn.get(LMDB_META_KEY)).get('prefix_digest')
    digest = _PrefixDigest()
    pbar = get_tqdm(total=size, initial=progress)
    resume_at = progress     # datapoints before this index are already stored
    committed = progress     # progress that will be recorded by the next commit
    batch = []
    idx = -1
    prefix_verified = not resume_at

    def verify_prefix():
        """Compare the replayed committed prefix with the recorded digest."""
        if digest.hexdigest() != expected_digest:
            raise LMDBResumeError(
                "Cannot resume {}: the source produces different datapoints in the "
                "already-committed prefix. The input changed; refusing to mix data "
                "from different inputs.".format(path))

    try:
        for idx, dp in enumerate(df):
            blob = dumps(dp)
            if idx < resume_at:
                digest.update(blob)
                continue
            if idx >= size:
                # A fixed source must not produce more than its declared length.
                # Stop before writing anything beyond it, leaving the export
                # incomplete rather than silently malformed.
                raise LMDBResumeError(
                    "{} yields more than its declared length of {}. "
                    "LMDBSerializer.save_resume() requires a fixed, repeatable, "
                    "finite source; the export is left incomplete.".format(
                        _type_name(df), size))
            if not prefix_verified:
                verify_prefix()
                prefix_verified = True
            digest.update(blob)
            # Materialize the whole batch in memory before writing it, so that
            # a failure while generating a later datapoint leaves no
            # half-written batch behind.
            batch.append((_lmdb_key(idx), blob))
            if len(batch) >= write_frequency:
                committed = _commit_batch(db, batch, committed, size, source_fp,
                                          source_digest, export_digest, digest.hexdigest())
                pbar.update(len(batch))
                batch = []
        if not prefix_verified:
            if idx + 1 < resume_at:
                raise LMDBResumeError(
                    "Cannot resume {}: the source ended before the committed prefix at "
                    "index {} was reached.".format(path, resume_at))
            # The source ends exactly at the committed prefix: the previous run
            # crashed after committing the last datapoint but before writing the
            # final marker.  The prefix digest still applies.
            verify_prefix()
        if batch:
            committed = _commit_batch(db, batch, committed, size, source_fp,
                                      source_digest, export_digest, digest.hexdigest())
            pbar.update(len(batch))
            batch = []

        total = idx + 1 if idx >= 0 else 0
        if total != size:
            raise LMDBResumeError(
                "{} produced {} datapoints but reports a length of {}. "
                "LMDBSerializer.save_resume() requires a fixed, repeatable, finite "
                "source whose length matches what it yields; the export is left "
                "incomplete.".format(_type_name(df), total, size))
        _finish_export(db, total, size, source_fp, source_digest, export_digest,
                       digest.hexdigest())
        return total
    finally:
        pbar.close()


def _put_or_grow(db, batch, meta):
    """
    Write one batch and its metadata in a single transaction, growing the map
    and replaying the whole batch if it does not fit.  Aborting the
    transaction rolls back every row written so far in this batch, and the
    replay writes them all again -- so no row of the current batch is lost and
    a failed batch never becomes progress.
    """
    while True:
        txn = None
        try:
            txn = db.begin(write=True)
            for key, blob in batch:
                txn.put(key, blob)
            for key, blob in meta:
                txn.put(key, blob)
            txn.commit()
            return
        except lmdb.MapFullError:
            if txn is not None:
                try:
                    txn.abort()
                except Exception:
                    pass
            _grow_map(db)
        except BaseException:
            if txn is not None:
                try:
                    txn.abort()
                except Exception:
                    pass
            raise


def _make_meta(status, progress, size, source_fp, source_digest, export_definition,
               prefix_digest):
    return {
        'version': _LMDB_RESUME_VERSION,
        'status': status,
        'progress': progress,
        'size': size,
        'key_format': _LMDB_KEY_FORMAT,
        'source_fingerprint': source_fp,
        'source_id': source_digest,
        'export_definition': export_definition,
        'prefix_digest': prefix_digest,
    }


def _commit_batch(db, batch, progress, size, source_fp, source_digest, export_definition,
                  prefix_digest):
    if not batch:
        return progress
    new_progress = progress + len(batch)
    meta = _make_meta('in_progress', new_progress, size,
                      source_fp, source_digest, export_definition, prefix_digest)
    _put_or_grow(db, batch, [(LMDB_META_KEY, dumps(meta))])
    return new_progress


def _finish_export(db, total, size, source_fp, source_digest, export_definition,
                   prefix_digest):
    keys = [_lmdb_key(k) for k in range(total)]
    meta = _make_meta('complete', total, size, source_fp, source_digest,
                      export_definition, prefix_digest)
    _put_or_grow(db, [(LMDB_KEYS_KEY, dumps(keys))],
                 [(LMDB_META_KEY, dumps(meta))])


class LMDBSerializer():
    """
    Serialize a Dataflow to a lmdb database, where the keys are indices and values
    are serialized datapoints.

    You will need to ``pip install lmdb`` to use it.

    Example:

    .. code-block:: python

        LMDBSerializer.save(my_df, "output.lmdb")

        new_df = LMDBSerializer.load("output.lmdb", shuffle=True)

    ``save`` writes everything in one shot and refuses an existing path.  For a
    large, fixed, repeatable source, :meth:`save_resume` can continue an
    interrupted export instead of starting over:

    .. code-block:: python

        LMDBSerializer.save_resume(my_df, "output.lmdb", write_frequency=5000)
    """
    @staticmethod
    def save(df, path, write_frequency=5000):
        """
        Serialize a DataFlow to a new LMDB database in one shot.

        This is the original, non-resumable entry point: it refuses to touch an
        existing database.  If an interrupted export needs to be continued, use
        :meth:`save_resume` instead.

        Args:
            df (DataFlow): the DataFlow to serialize.
            path (str): output path. Either a directory or an lmdb file.
            write_frequency (int): the frequency to write back data to disk.
                A smaller value reduces memory usage.
        """
        assert isinstance(df, DataFlow), type(df)
        isdir = os.path.isdir(path)
        if isdir:
            assert not os.path.isfile(os.path.join(path, 'data.mdb')), "LMDB file exists!"
        else:
            assert not os.path.isfile(path), "LMDB file {} exists!".format(path)
        map_size = _default_map_size()
        db = lmdb.open(path, subdir=isdir,
                       map_size=map_size, readonly=False,
                       meminit=False, map_async=True)    # need sync() at the end
        size = _reset_df_and_get_size(df)

        # put data into lmdb, and doubling the size if full.
        # Ref: https://github.com/NVIDIA/DIGITS/pull/209/files
        def put_or_grow(txn, key, value):
            try:
                txn.put(key, value)
                return txn
            except lmdb.MapFullError:
                pass
            txn.abort()
            curr_size = db.info()['map_size']
            new_size = curr_size * 2
            logger.info("Doubling LMDB map_size to {:.2f}GB".format(new_size / 10**9))
            db.set_mapsize(new_size)
            txn = db.begin(write=True)
            txn = put_or_grow(txn, key, value)
            return txn

        with get_tqdm(total=size) as pbar:
            idx = -1

            # LMDB transaction is not exception-safe!
            # although it has a context manager interface
            txn = db.begin(write=True)
            for idx, dp in enumerate(df):
                txn = put_or_grow(txn, _lmdb_key(idx), dumps(dp))
                pbar.update()
                if (idx + 1) % write_frequency == 0:
                    txn.commit()
                    txn = db.begin(write=True)
            txn.commit()

            keys = [_lmdb_key(k) for k in range(idx + 1)]
            with db.begin(write=True) as txn:
                txn = put_or_grow(txn, LMDB_KEYS_KEY, dumps(keys))

            logger.info("Flushing database ...")
            db.sync()
        db.close()

    @staticmethod
    def export_status(path):
        """
        Inspect an LMDB export without reading any datapoint.

        Args:
            path (str): a directory or a file.

        Returns:
            str or None: ``'complete'`` if the export can be read, ``'incomplete'``
            if it is a resumable export that has not been finished yet, or None
            if it is not a resumable export (e.g. produced by :meth:`save`).

        Raises:
            IOError: if the database does not exist.
        """
        if not os.path.exists(path):
            raise IOError("LMDB export {} does not exist!".format(path))
        env = lmdb.open(path, subdir=os.path.isdir(path), readonly=True, lock=False,
                        create=False, map_size=_default_map_size(), max_readers=100)
        try:
            with env.begin() as txn:
                meta = read_lmdb_meta(txn)
        finally:
            env.close()
        if meta is None:
            return None
        return 'complete' if meta.get('status') == 'complete' else 'incomplete'

    @staticmethod
    def save_resume(df, path, write_frequency=5000, source_id=None,
                    export_definition=None, map_size=None):
        """
        Serialize a DataFlow to a LMDB database, and continue a previously
        interrupted export if one exists at ``path``.

        This entry point is meant for **fixed, repeatable, finite** sources:
        calling ``df.reset_state()`` and iterating it again must produce the
        same datapoints in the same order.  On every call the export is
        continued from the last committed batch, so an interrupted process can
        simply be started again with the same arguments.

        Guarantees:

        * Data and resume progress are committed in the same LMDB transaction.
          The progress stored in ``__meta__`` never runs ahead of the data.
        * A batch is committed all-or-nothing.  If generating or writing a
          batch fails, that batch is discarded and does not count as progress;
          the next call re-reads it from the source.  Because keys are derived
          from the datapoint index, re-writing the same index cannot duplicate
          rows.
        * When the map is full, the whole current batch is replayed into the
          enlarged map; rows written earlier in the same batch are not lost.
        * On resume, the source identity is checked before any new row is
          appended: the DataFlow definition, its length, and the optional
          ``source_id`` / ``export_definition`` must match, and the committed
          prefix is re-generated and compared through a rolling digest.  A
          mismatch raises :class:`LMDBResumeError` instead of mixing data from
          different inputs.
        * The database is readable through the existing :class:`LMDBData` /
          :meth:`load` chain only after it's finished.  Reading an unfinished
          export raises :class:`IncompleteLMDBError`; :meth:`export_status`
          reports it without reading any datapoint.

        Args:
            df (DataFlow): the DataFlow to serialize.  It should be a fixed and
                repeatable finite source.
            path (str): output path. Either a directory or an lmdb file.
            write_frequency (int): number of datapoints per committed batch.
                A smaller value loses less work on interruption, at the cost of
                more transactions.  The current batch is buffered in memory, so
                this also bounds memory use.
            source_id: optional, any picklable identity of the input (e.g. a
                dataset name, version, or checksum string).  It is hashed and
                recorded, and must match when resuming.  Use it when the
                automatic description of ``df`` cannot capture changes that
                matter to you.
            export_definition: optional, any picklable value describing how the
                data is exported (e.g. a preprocessing version).  It is hashed,
                recorded and must match when resuming.
            map_size (int): initial LMDB map size in bytes.  None uses a very
                large value on Linux.  The map is doubled automatically when
                full.

        Returns:
            int: total number of datapoints in the completed export.
        """
        assert isinstance(df, DataFlow), type(df)
        if not isinstance(write_frequency, int) or isinstance(write_frequency, bool) \
                or write_frequency <= 0:
            raise ValueError("write_frequency must be a positive integer!")

        isdir = os.path.isdir(path)
        initial_map_size = map_size if map_size is not None else _default_map_size()

        # The source is needed for length / identity before we can resume.
        df.reset_state()
        try:
            size = int(len(df))
        except (NotImplementedError, TypeError):
            size = None
        _check_finite_source(df, size)

        source_fp = _fingerprint_datasource(df)
        source_digest = _digest_value(source_id)
        export_digest = _digest_value(export_definition)

        db = _open_resumable_lmdb(path, isdir, initial_map_size)
        try:
            _ensure_map_size(db, initial_map_size)
            with db.begin() as txn:
                meta = read_lmdb_meta(txn)

            if meta is None:
                if _data_entry_count(db) != 0:
                    raise LMDBResumeError(
                        "{} already contains data but is not a resumable export. "
                        "Refusing to overwrite it; use LMDBSerializer.save() only on an empty "
                        "path.".format(path))
                progress = 0
                _put_or_grow(db, [], [(LMDB_META_KEY, dumps(
                    _make_meta('in_progress', 0, size, source_fp,
                               source_digest, export_digest, _PrefixDigest().hexdigest())))])
            else:
                progress = _validate_resume_meta(
                    db, meta, path, source_fp, source_digest, export_digest, size)
                if meta.get('status') == 'complete':
                    logger.info("{} is already a complete export with {} datapoints.".format(
                        path, progress))
                    return progress

            total = _write_resumable(df, db, path, progress, size,
                                     source_fp, source_digest, export_digest, write_frequency)
            logger.info("Flushing database ...")
            db.sync()
            return total
        finally:
            db.close()

    @staticmethod
    def load(path, shuffle=True):
        """
        Note:
            If you found deserialization being the bottleneck, you can use :class:`LMDBData` as the reader
            and run deserialization as a mapper in parallel.

        Raises:
            IncompleteLMDBError: if the export was created by :meth:`save_resume`
                but has not been finished yet.
        """
        df = LMDBData(path, shuffle=shuffle)
        return MapData(df, LMDBSerializer._deserialize_lmdb)

    @staticmethod
    def _deserialize_lmdb(dp):
        return loads(dp[1])


class NumpySerializer():
    """
    Serialize the entire dataflow to a npz dict.
    Note that this would have to store the entire dataflow in memory,
    and is also >10x slower than LMDB/TFRecord serializers.
    """

    @staticmethod
    def save(df, path):
        """
        Args:
            df (DataFlow): the DataFlow to serialize.
            path (str): output npz file.
        """
        buffer = []
        size = _reset_df_and_get_size(df)
        with get_tqdm(total=size) as pbar:
            for dp in df:
                buffer.append(dp)
                pbar.update()
        np.savez_compressed(path, buffer=np.asarray(buffer, dtype=np.object))

    @staticmethod
    def load(path, shuffle=True):
        # allow_pickle defaults to False since numpy 1.16.3
        # (https://www.numpy.org/devdocs/release.html#unpickling-while-loading-requires-explicit-opt-in)
        buffer = np.load(path, allow_pickle=True)['buffer']
        return DataFromList(buffer, shuffle=shuffle)


class TFRecordSerializer():
    """
    Serialize datapoints to bytes (by tensorpack's default serializer) and write to a TFRecord file.

    Note that TFRecord does not support random access and is in fact not very performant.
    It's better to use :class:`LMDBSerializer`.
    """
    @staticmethod
    def save(df, path):
        """
        Args:
            df (DataFlow): the DataFlow to serialize.
            path (str): output tfrecord file.
        """
        size = _reset_df_and_get_size(df)
        with tf.python_io.TFRecordWriter(path) as writer, get_tqdm(total=size) as pbar:
            for dp in df:
                writer.write(dumps(dp))
                pbar.update()

    @staticmethod
    def load(path, size=None):
        """
        Args:
            size (int): total number of records. If not provided, the returned dataflow will have no `__len__()`.
                It's needed because this metadata is not stored in the TFRecord file.
        """
        gen = tf.python_io.tf_record_iterator(path)
        ds = DataFromGenerator(gen)
        ds = MapData(ds, loads)
        if size is not None:
            ds = FixedSizeData(ds, size)
        return ds


class HDF5Serializer():
    """
    Write datapoints to a HDF5 file.

    Note that HDF5 files are in fact not very performant and currently do not support lazy loading.
    It's better to use :class:`LMDBSerializer`.
    """
    @staticmethod
    def save(df, path, data_paths):
        """
        Args:
            df (DataFlow): the DataFlow to serialize.
            path (str): output hdf5 file.
            data_paths (list[str]): list of h5 paths. It should have the same
                length as each datapoint, and each path should correspond to one
                component of the datapoint.
        """
        size = _reset_df_and_get_size(df)
        buffer = defaultdict(list)

        with get_tqdm(total=size) as pbar:
            for dp in df:
                assert len(dp) == len(data_paths), "Datapoint has {} components!".format(len(dp))
                for k, el in zip(data_paths, dp):
                    buffer[k].append(el)
                pbar.update()

        with h5py.File(path, 'w') as hf, get_tqdm(total=len(data_paths)) as pbar:
            for data_path in data_paths:
                hf.create_dataset(data_path, data=buffer[data_path])
                pbar.update()

    @staticmethod
    def load(path, data_paths, shuffle=True):
        """
        Args:
            data_paths (list): list of h5 paths to be zipped.
        """
        return HDF5Data(path, data_paths, shuffle)


try:
    import lmdb
except ImportError:
    LMDBSerializer = create_dummy_class('LMDBSerializer', 'lmdb')   # noqa

try:
    import tensorflow as tf
except ImportError:
    TFRecordSerializer = create_dummy_class('TFRecordSerializer', 'tensorflow')   # noqa

try:
    import h5py
except ImportError:
    HDF5Serializer = create_dummy_class('HDF5Serializer', 'h5py')   # noqa


if __name__ == '__main__':
    from .raw import FakeData
    import time
    ds = FakeData([[300, 300, 3], [1]], 1000)

    print(time.time())
    TFRecordSerializer.save(ds, 'out.tfrecords')
    print(time.time())
    df = TFRecordSerializer.load('out.tfrecords', size=1000)
    df.reset_state()
    for idx, dp in enumerate(df):
        pass
    print("TF Finished, ", idx)
    print(time.time())

    LMDBSerializer.save(ds, 'out.lmdb')
    print(time.time())
    df = LMDBSerializer.load('out.lmdb')
    df.reset_state()
    for idx, dp in enumerate(df):
        pass
    print("LMDB Finished, ", idx)
    print(time.time())

    NumpySerializer.save(ds, 'out.npz')
    print(time.time())
    df = NumpySerializer.load('out.npz')
    df.reset_state()
    for idx, dp in enumerate(df):
        pass
    print("Numpy Finished, ", idx)
    print(time.time())

    paths = ['p1', 'p2']
    HDF5Serializer.save(ds, 'out.h5', paths)
    print(time.time())
    df = HDF5Serializer.load('out.h5', paths)
    df.reset_state()
    for idx, dp in enumerate(df):
        pass
    print("HDF5 Finished, ", idx)
    print(time.time())
