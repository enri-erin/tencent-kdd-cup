"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1
SEQ_TEMPORAL_DENSE_DIM = 8
SEQ_TEMPORAL_ID_DIM = 10
SEQ_LOCAL_TIME_OFFSET_SECONDS = 8 * 3600
SEQ_REQUEST_GAP_SECONDS = 30
SEQ_SESSION_GAP_SECONDS = 1800
SEQ_VISIT_GAP_SECONDS = 86400
SEQ_SESSION_POS_NORM_MAX = 32.0
SEQ_TIME_NORM_MAX_SECONDS = 31536000.0
SEQ_FOURIER_PERIODS = (3600.0, 86400.0, 604800.0)
COARSE_SEQ_FEATS_PER_DOMAIN = 4
COARSE_GLOBAL_SEQ_FEATS = 4
COARSE_GLOBAL_STATIC_FEATS = 0
TARGET_SEQ_FEATS_PER_SEQUENCE = 8
TARGET_SEQ_RECENT_WINDOW = 5
TARGET_SEQ_TIME_WEIGHT_TAU_BUCKETS = 4.0
COARSE_RECENT_7D_SECONDS = 604800
COARSE_FRESHNESS_TAU_SECONDS = 604800.0

# Based on the demo_1000 analysis, these side-info fields either exhibited
# near-random behavior or a clearly reversed correlation when matched directly
# against the target item's sparse ids. They are therefore excluded from the
# target-sequence relation branch by default.
TARGET_SEQ_FIELD_BLACKLIST: Dict[str, set[int]] = {
    'seq_a': {40, 41, 46},
    'seq_b': {70, 75, 77},
    'seq_c': {32, 35},
    'seq_d': {18, 21, 25},
}


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        use_coarse_features: bool = True,
        use_target_seq_features: bool = True,
        use_temporal_sequence_features: bool = True,
        use_iat_sequence_branch: bool = False,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
            use_coarse_features: whether to construct the coarse feature branch.
            use_target_seq_features: whether to construct target-sequence
                relation features.
            use_temporal_sequence_features: whether to construct the newer
                multi-granularity temporal sequence features, including both
                discrete temporal ids and continuous temporal dense values.
            use_iat_sequence_branch: whether to expose the extra per-instance
                metadata required by the IAT branch.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.use_coarse_features = use_coarse_features
        self.use_target_seq_features = use_target_seq_features
        self.use_temporal_sequence_features = use_temporal_sequence_features
        self.use_iat_sequence_branch = use_iat_sequence_branch
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_temporal_ids = {}
        self._buf_seq_temporal = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_temporal_ids[domain] = np.zeros(
                (B, max_len, SEQ_TEMPORAL_ID_DIM),
                dtype=np.int64,
            )
            self._buf_seq_temporal[domain] = np.zeros(
                (B, max_len, SEQ_TEMPORAL_DENSE_DIM),
                dtype=np.float32,
            )
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)
        self.coarse_seq_per_domain_dim = (
            COARSE_SEQ_FEATS_PER_DOMAIN if self.use_coarse_features else 0
        )
        self.coarse_global_seq_dim = (
            COARSE_GLOBAL_SEQ_FEATS if self.use_coarse_features else 0
        )
        self.coarse_global_static_dim = (
            COARSE_GLOBAL_STATIC_FEATS if self.use_coarse_features else 0
        )
        self.coarse_feat_dim = (
            len(self.seq_domains) * self.coarse_seq_per_domain_dim
            + self.coarse_global_seq_dim
            + self.coarse_global_static_dim
        )
        self._buf_coarse = np.zeros((B, self.coarse_feat_dim), dtype=np.float32)
        if self.use_target_seq_features:
            self.target_seq_sideinfo_counts = [
                len(self.target_sideinfo_fids[domain]) for domain in self.seq_domains
            ]
            self.target_seq_feat_dim = (
                sum(self.target_seq_sideinfo_counts) * TARGET_SEQ_FEATS_PER_SEQUENCE
            )
        else:
            self.target_seq_sideinfo_counts = [0 for _ in self.seq_domains]
            self.target_seq_feat_dim = 0
        self._buf_target_seq = np.zeros((B, self.target_seq_feat_dim), dtype=np.float32)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self.target_sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            blacklist = TARGET_SEQ_FIELD_BLACKLIST.get(domain, set())
            self.target_sideinfo_fids[domain] = [
                fid for fid in sideinfo if fid not in blacklist
            ]
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    def _clean_raw_seq_values(
        self,
        values: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> "npt.NDArray[np.int64]":
        """Clean one untruncated raw sparse-id slice for rough feature use."""
        cleaned = np.asarray(values, dtype=np.int64).copy()
        if cleaned.size == 0:
            return cleaned
        cleaned[cleaned <= 0] = 0
        if vocab_size > 0:
            cleaned[cleaned >= vocab_size] = 0
        else:
            cleaned[:] = 0
        return cleaned

    def _bucketize_raw_timestamps(
        self,
        seq_timestamps: "npt.NDArray[np.int64]",
        current_ts: int,
    ) -> "npt.NDArray[np.int64]":
        """Convert one untruncated raw timestamp slice to time buckets."""
        raw_ts = np.asarray(seq_timestamps, dtype=np.int64).copy()
        if raw_ts.size == 0:
            return raw_ts
        raw_ts[raw_ts <= 0] = 0
        time_diff = np.maximum(current_ts - raw_ts, 0)
        raw_buckets = np.clip(
            np.searchsorted(BUCKET_BOUNDARIES, time_diff),
            0,
            len(BUCKET_BOUNDARIES) - 1,
        )
        buckets = raw_buckets.astype(np.int64) + 1
        buckets[raw_ts == 0] = 0
        return buckets

    def _build_temporal_features(
        self,
        seq_timestamps: "npt.NDArray[np.int64]",
        current_ts: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.float32]"]:
        """Build discrete temporal ids + continuous temporal values.

        Discrete ids layout (10 dims):
        0: hour_id
        1: weekday_id
        2: weekend_id
        3: inter-event gap bucket id
        4-6: request/session/visit position ids
        7-9: request/session/visit boundary ids

        Dense layout (8 dims):
        0: age_log_norm
        1: inter-event gap_log_norm
        2-7: Fourier sin/cos for hour/day/week scales
        """
        raw_ts = np.asarray(seq_timestamps, dtype=np.int64).copy()
        if raw_ts.size == 0:
            return (
                np.zeros((0, SEQ_TEMPORAL_ID_DIM), dtype=np.int64),
                np.zeros((0, SEQ_TEMPORAL_DENSE_DIM), dtype=np.float32),
            )

        raw_ts[raw_ts <= 0] = 0
        temporal_ids = np.zeros((raw_ts.shape[0], SEQ_TEMPORAL_ID_DIM), dtype=np.int64)
        feats = np.zeros((raw_ts.shape[0], SEQ_TEMPORAL_DENSE_DIM), dtype=np.float32)
        valid_mask = raw_ts > 0
        if not valid_mask.any():
            return temporal_ids, feats

        valid_ts = raw_ts[valid_mask]
        local_ts = valid_ts + SEQ_LOCAL_TIME_OFFSET_SECONDS
        day_index = local_ts // 86400
        hour = ((local_ts % 86400) // 3600).astype(np.float32)
        weekday = ((day_index + 3) % 7).astype(np.float32)

        hour_ids = hour.astype(np.int64) + 1
        weekday_ids = weekday.astype(np.int64) + 1
        weekend_ids = (weekday >= 5.0).astype(np.int64) + 1
        temporal_ids[valid_mask, 0] = hour_ids
        temporal_ids[valid_mask, 1] = weekday_ids
        temporal_ids[valid_mask, 2] = weekend_ids

        age_seconds = np.maximum(current_ts - valid_ts, 0).astype(np.float32)
        feats[valid_mask, 0] = (
            np.log1p(age_seconds) / np.log1p(SEQ_TIME_NORM_MAX_SECONDS)
        )

        gaps = np.zeros(raw_ts.shape[0], dtype=np.int64)
        prev_valid = raw_ts[:-1] > 0
        curr_valid = raw_ts[1:] > 0
        valid_pairs = prev_valid & curr_valid
        if valid_pairs.any():
            pair_gaps = np.maximum(raw_ts[:-1][valid_pairs] - raw_ts[1:][valid_pairs], 0)
            gaps[1:][valid_pairs] = pair_gaps

        gap_log_norm = (
            np.log1p(gaps.astype(np.float32)) / np.log1p(SEQ_TIME_NORM_MAX_SECONDS)
        )
        nonzero_gap = gaps > 0
        if nonzero_gap.any():
            gap_bucket = np.clip(
                np.searchsorted(BUCKET_BOUNDARIES, gaps[nonzero_gap]),
                0,
                len(BUCKET_BOUNDARIES) - 1,
            ).astype(np.float32) + 1.0
            temporal_ids[nonzero_gap, 3] = gap_bucket.astype(np.int64)
        feats[:, 1] = gap_log_norm * valid_mask.astype(np.float32)

        request_pos = 0
        session_pos = 0
        visit_pos = 0
        prev_ts = 0
        for pos, ts_val in enumerate(raw_ts.tolist()):
            if ts_val <= 0:
                request_pos = 0
                session_pos = 0
                visit_pos = 0
                prev_ts = 0
                continue

            if prev_ts <= 0:
                new_request = True
                new_session = True
                new_visit = True
                request_pos = 0
                session_pos = 0
                visit_pos = 0
            else:
                gap = max(prev_ts - ts_val, 0)
                new_visit = gap > SEQ_VISIT_GAP_SECONDS
                new_session = new_visit or gap > SEQ_SESSION_GAP_SECONDS
                new_request = new_session or gap > SEQ_REQUEST_GAP_SECONDS

                if new_visit:
                    request_pos = 0
                    session_pos = 0
                    visit_pos = 0
                elif new_session:
                    request_pos = 0
                    session_pos = 0
                    visit_pos += 1
                elif new_request:
                    request_pos = 0
                    session_pos += 1
                    visit_pos += 1
                else:
                    request_pos += 1
                    session_pos += 1
                    visit_pos += 1

            temporal_ids[pos, 4] = min(request_pos + 1, int(SEQ_SESSION_POS_NORM_MAX))
            temporal_ids[pos, 5] = min(session_pos + 1, int(SEQ_SESSION_POS_NORM_MAX))
            temporal_ids[pos, 6] = min(visit_pos + 1, int(SEQ_SESSION_POS_NORM_MAX))
            temporal_ids[pos, 7] = int(new_request) + 1
            temporal_ids[pos, 8] = int(new_session) + 1
            temporal_ids[pos, 9] = int(new_visit) + 1
            prev_ts = ts_val

        local_ts_float = local_ts.astype(np.float32)
        for period_idx, period in enumerate(SEQ_FOURIER_PERIODS):
            angle = 2.0 * np.pi * (local_ts_float / period)
            base = 2 + period_idx * 2
            feats[valid_mask, base] = np.sin(angle)
            feats[valid_mask, base + 1] = np.cos(angle)

        return temporal_ids, feats

    def _compute_repeat_ratio(
        self,
        raw_side_sequences: List["npt.NDArray[np.int64]"],
        raw_len: int,
    ) -> float:
        """Measure how repetitive the active step signatures are in one domain."""
        if raw_len <= 1 or not raw_side_sequences:
            return 0.0

        sig_matrix = np.zeros((len(raw_side_sequences), raw_len), dtype=np.int64)
        active_mask = np.zeros(raw_len, dtype=bool)
        for field_idx, seq_vals in enumerate(raw_side_sequences):
            use_len = min(raw_len, seq_vals.shape[0])
            if use_len <= 0:
                continue
            sig_matrix[field_idx, :use_len] = seq_vals[:use_len]
            active_mask[:use_len] |= seq_vals[:use_len] > 0

        active_steps = int(active_mask.sum())
        if active_steps <= 1:
            return 0.0

        active_signatures = np.ascontiguousarray(sig_matrix[:, active_mask].T)
        unique_active = np.unique(active_signatures, axis=0).shape[0]
        return 1.0 - float(unique_active) / float(active_steps)

    def _compute_target_seq_row_features(
        self,
        item_vals: "npt.NDArray[np.int64]",
        seq_vals: "npt.NDArray[np.int64]",
        time_buckets: "npt.NDArray[np.int64]",
    ) -> "npt.NDArray[np.float32]":
        """Compute 8 multi-window target-seq heat features.

        Uses time-bucketed multi-window statistics to capture how often the
        target item's sideinfo IDs appear in the user's historical sequence
        across different time horizons.

        Dims 0-3: positive multi-window heat (higher = stronger target signal)
          0: 1h window target match heat
          1: 6h window target match heat
          2: 1d window target match heat
          3: 7d/30d blended long-term heat (0.7*heat_7d + 0.3*heat_30d)

        Dims 4-7: negative / cooling / stale signals (higher = weaker signal)
          4: 1 - 1d heat  (recent non-match / coldness)
          5: max(long_term - recent, 0)  (interest fading)
          6: stale_hotness  (30d has hits but 1d / 6h are weak)
          7: recency_penalty  (time since last match, 1.0 if never matched)

        All features are clipped to [0, 1].  No label or global-future
        information is used.  When time_buckets are unavailable a
        position-based fallback keeps the pipeline running without error.
        """
        feats = np.zeros(TARGET_SEQ_FEATS_PER_SEQUENCE, dtype=np.float32)
        if item_vals.size == 0 or seq_vals.size == 0:
            return feats

        # --- Align time_bucket length with seq_vals ---
        tb = np.asarray(time_buckets, dtype=np.int64)
        if tb.shape[0] < seq_vals.shape[0]:
            _padded = np.zeros(seq_vals.shape[0], dtype=np.int64)
            _padded[:tb.shape[0]] = tb
            tb = _padded
        elif tb.shape[0] > seq_vals.shape[0]:
            tb = tb[:seq_vals.shape[0]]

        # --- Convert bucket ids to approximate wall-clock seconds ---
        time_seconds = np.zeros(tb.shape[0], dtype=np.float32)
        valid_tb = tb > 0
        use_position_fallback = True
        if valid_tb.any():
            bucket_idx = np.clip(
                tb[valid_tb] - 1, 0, len(BUCKET_BOUNDARIES) - 1,
            )
            time_seconds[valid_tb] = BUCKET_BOUNDARIES[bucket_idx].astype(np.float32)
            use_position_fallback = False

        if use_position_fallback:
            # Assume ~1 h per position when no real timestamps exist.
            time_seconds = (
                np.arange(seq_vals.shape[0], dtype=np.float32) * 3600.0
            )

        # --- Match: which positions contain any target-item feature value ---
        matches = np.isin(seq_vals, item_vals)
        if use_position_fallback:
            active = seq_vals > 0
        else:
            # Positions whose time_bucket is invalid (<=0) carry no reliable
            # temporal information; exclude them from all window statistics.
            active = (seq_vals > 0) & valid_tb
        effective = matches & active

        # --- Per-window heat (log1p-normalised hit rate) ---
        _WIN_SEC = [
            ('1h', 3600.0),
            ('6h', 21600.0),
            ('1d', 86400.0),
            ('7d', 604800.0),
            ('30d', 2592000.0),
        ]
        heat: Dict[str, float] = {}
        for _name, _sec in _WIN_SEC:
            in_win = active & (time_seconds <= _sec)
            n_active = max(int(in_win.sum()), 1)
            n_hit = int((effective & in_win).sum())
            heat[_name] = float(
                np.log1p(float(n_hit)) / np.log1p(float(n_active))
            )

        # --- Positive signals ---
        feats[0] = heat['1h']
        feats[1] = heat['6h']
        feats[2] = heat['1d']
        feats[3] = 0.7 * heat['7d'] + 0.3 * heat['30d']

        # --- Negative / cooling signals ---
        feats[4] = 1.0 - heat['1d']

        _long_minus_recent = heat['30d'] - heat['1d']
        feats[5] = max(_long_minus_recent, 0.0)

        # Stale: 30 d has signal but 1 d AND 6 h are faint.
        if heat['30d'] > 0.01 and heat['6h'] < 0.5:
            feats[6] = heat['30d'] * (1.0 - heat['6h'])
        # else stays 0.0

        # Recency penalty: normalised seconds since last match.
        matched_times = time_seconds[effective]
        if matched_times.size > 0:
            feats[7] = min(matched_times.min() / 2592000.0, 1.0)
        else:
            # Never matched: maximum penalty.  1.0 is the simplest stable
            # default and means "no personal history for this target".
            feats[7] = 1.0

        return np.clip(feats, 0.0, 1.0)

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }
        if self.use_iat_sequence_branch:
            item_ids = batch.column(self._col_idx['item_id']).to_numpy().astype(np.int64)
            result['item_id'] = torch.from_numpy(item_ids)

        coarse = self._buf_coarse[:B]
        coarse[:] = 0.0
        target_seq = self._buf_target_seq[:B]
        target_seq[:] = 0.0
        need_coarse = self.use_coarse_features
        need_target = self.use_target_seq_features
        active_step_counts = [] if need_coarse else None
        recent_domain_hits = [] if need_coarse else None
        raw_total_lengths = np.zeros(B, dtype=np.float32) if need_coarse else None
        item_nonzero_vals = [item_int[i][item_int[i] > 0] for i in range(B)] if need_target else None
        target_cursor = 0

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for domain_idx, domain in enumerate(self.seq_domains):
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            raw_domain_lengths = np.zeros(B, dtype=np.int64)
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                offs = col.offsets.to_numpy()
                vals = col.values.to_numpy()
                col_data.append((offs, vals, vs, ci))
                raw_len = (offs[1:B + 1] - offs[:B]).astype(np.int64, copy=False)
                raw_domain_lengths = np.maximum(raw_domain_lengths, raw_len)

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            temporal_ids = self._buf_seq_temporal_ids[domain][:B]
            temporal_ids[:] = 0
            temporal_dense = self._buf_seq_temporal[domain][:B]
            temporal_dense[:] = 0.0
            ts_offs = None
            ts_vals = None
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                raw_ts_lens = (ts_offs[1:B + 1] - ts_offs[:B]).astype(np.int64, copy=False)
                raw_domain_lengths = np.maximum(raw_domain_lengths, raw_ts_lens)
                # Pad timestamps into shape (B, max_len).
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    if self.use_temporal_sequence_features:
                        cur_ids, cur_dense = self._build_temporal_features(
                            ts_vals[s:s + ul],
                            int(timestamps[i]),
                        )
                        temporal_ids[i, :ul, :] = cur_ids
                        temporal_dense[i, :ul, :] = cur_dense

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())
            if self.use_temporal_sequence_features:
                result[f'{domain}_temporal_ids'] = torch.from_numpy(temporal_ids.copy())
                result[f'{domain}_temporal_dense'] = torch.from_numpy(temporal_dense.copy())

            if not need_coarse and not need_target:
                continue

            raw_lengths_f = raw_domain_lengths.astype(np.float32) if need_coarse else None
            raw_time_buckets = [] if need_target else None
            if need_coarse:
                overflow_ratio = np.maximum(raw_lengths_f - float(max_len), 0.0) / float(max(max_len, 1))
                recent_7d_ratio = np.zeros(B, dtype=np.float32)
                freshness_score = np.zeros(B, dtype=np.float32)
                repeat_ratio = np.zeros(B, dtype=np.float32)
                active_step_count = np.zeros(B, dtype=np.float32)
                recent_domain_hit = np.zeros(B, dtype=np.float32)
            for i in range(B):
                domain_raw_len = int(raw_domain_lengths[i])
                raw_side_sequences = [] if need_coarse else None
                if need_coarse:
                    for offs, vals, vs, _ in col_data:
                        s = int(offs[i])
                        e = int(offs[i + 1])
                        raw_seq = self._clean_raw_seq_values(vals[s:e], vs)
                        raw_side_sequences.append(raw_seq)
                    if domain_raw_len > 0:
                        active_mask = np.zeros(domain_raw_len, dtype=bool)
                        for raw_seq in raw_side_sequences:
                            use_len = min(domain_raw_len, raw_seq.shape[0])
                            if use_len <= 0:
                                continue
                            active_mask[:use_len] |= raw_seq[:use_len] > 0
                        active_step_count[i] = float(active_mask.sum())
                        repeat_ratio[i] = self._compute_repeat_ratio(
                            raw_side_sequences,
                            domain_raw_len,
                        )
                if need_target:
                    raw_tb = np.zeros(domain_raw_len, dtype=np.int64)
                    if ts_offs is not None and ts_vals is not None:
                        s = int(ts_offs[i])
                        e = int(ts_offs[i + 1])
                        raw_buckets = self._bucketize_raw_timestamps(
                            ts_vals[s:e],
                            int(timestamps[i]),
                        )
                        use_len = min(domain_raw_len, raw_buckets.shape[0])
                        raw_tb[:use_len] = raw_buckets[:use_len]
                    raw_time_buckets.append(raw_tb)
                if not need_coarse or ts_offs is None or ts_vals is None:
                    continue
                raw_ts = np.asarray(
                    ts_vals[int(ts_offs[i]):int(ts_offs[i + 1])],
                    dtype=np.int64,
                ).copy()
                raw_ts[raw_ts <= 0] = 0
                valid_ts = raw_ts[raw_ts > 0]
                if valid_ts.size == 0:
                    continue
                time_diff = np.maximum(int(timestamps[i]) - valid_ts, 0)
                recent_7d_ratio[i] = float((time_diff <= COARSE_RECENT_7D_SECONDS).mean())
                freshness_score[i] = float(
                    np.exp(-time_diff.astype(np.float32) / COARSE_FRESHNESS_TAU_SECONDS).mean()
                )
                if active_step_count[i] > 0 and recent_7d_ratio[i] > 0.0:
                    recent_domain_hit[i] = 1.0

            if need_coarse:
                base = domain_idx * COARSE_SEQ_FEATS_PER_DOMAIN
                coarse[:, base + 0] = overflow_ratio
                coarse[:, base + 1] = recent_7d_ratio
                coarse[:, base + 2] = freshness_score
                coarse[:, base + 3] = repeat_ratio

            if need_coarse:
                active_step_counts.append(active_step_count)
                recent_domain_hits.append(recent_domain_hit)
                raw_total_lengths += raw_lengths_f

            if need_target:
                col_data_by_fid = {
                    fid: (offs, vals, vs)
                    for fid, (offs, vals, vs, _) in zip(self.sideinfo_fids[domain], col_data)
                }
                for sideinfo_idx, fid in enumerate(self.target_sideinfo_fids[domain]):
                    offs, vals, vs = col_data_by_fid[fid]
                    feat_base = target_cursor + sideinfo_idx * TARGET_SEQ_FEATS_PER_SEQUENCE
                    for i in range(B):
                        s = int(offs[i])
                        e = int(offs[i + 1])
                        if e <= s:
                            continue
                        raw_seq = self._clean_raw_seq_values(vals[s:e], vs)
                        if raw_seq.size == 0:
                            continue
                        target_seq[i, feat_base:feat_base + TARGET_SEQ_FEATS_PER_SEQUENCE] = (
                            self._compute_target_seq_row_features(
                                item_nonzero_vals[i],
                                raw_seq,
                                raw_time_buckets[i],
                            )
                        )
                target_cursor += (
                    len(self.target_sideinfo_fids[domain]) * TARGET_SEQ_FEATS_PER_SEQUENCE
                )

        if need_coarse and active_step_counts:
            total_max_len = float(sum(max(self._seq_maxlen[d], 1) for d in self.seq_domains))
            active_steps_arr = np.stack(active_step_counts, axis=1)
            recent_domain_arr = np.stack(recent_domain_hits, axis=1)
            domain_balance = np.zeros(B, dtype=np.float32)
            if len(self.seq_domains) > 1:
                active_den = active_steps_arr.sum(axis=1, keepdims=True)
                active_shares = np.divide(
                    active_steps_arr,
                    active_den,
                    out=np.zeros_like(active_steps_arr),
                    where=active_den > 0,
                )
                share_logs = np.zeros_like(active_shares)
                np.log(
                    active_shares,
                    out=share_logs,
                    where=active_shares > 0,
                )
                domain_balance = (
                    -(active_shares * share_logs).sum(axis=1)
                    / float(np.log(len(self.seq_domains)))
                ).astype(np.float32, copy=False)
            global_base = len(self.seq_domains) * COARSE_SEQ_FEATS_PER_DOMAIN
            coarse[:, global_base + 0] = np.maximum(
                raw_total_lengths - total_max_len,
                0.0,
            ) / total_max_len
            coarse[:, global_base + 1] = (active_steps_arr > 0).mean(axis=1)
            coarse[:, global_base + 2] = domain_balance
            coarse[:, global_base + 3] = recent_domain_arr.mean(axis=1)

        result['coarse_feats'] = torch.from_numpy(coarse.copy())
        result['target_seq_feats'] = torch.from_numpy(target_seq.copy())
        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    use_coarse_features: bool = True,
    use_target_seq_features: bool = True,
    use_temporal_sequence_features: bool = True,
    use_iat_sequence_branch: bool = False,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    The validation split is taken as the last ``valid_ratio`` fraction of Row
    Groups (in the file order returned by ``glob``).

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    # train_ratio: use only the first N% of the training Row Groups.
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
        use_coarse_features=use_coarse_features,
        use_target_seq_features=use_target_seq_features,
        use_temporal_sequence_features=use_temporal_sequence_features,
        use_iat_sequence_branch=use_iat_sequence_branch,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
        use_coarse_features=use_coarse_features,
        use_target_seq_features=use_target_seq_features,
        use_temporal_sequence_features=use_temporal_sequence_features,
        use_iat_sequence_branch=use_iat_sequence_branch,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, train_dataset
