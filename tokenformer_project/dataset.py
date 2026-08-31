"""TokenFormer parquet dataset.

This module keeps the same high-level project structure as the current codebase
(`train.py`, `dataset.py`, `model.py`, `trainer.py`, `infer.py`, `utils.py`,
`run.sh`) while reshaping the raw recommendation data into the entities needed
by TokenFormer:

- F: non-sequential static feature fields
- T: chronological behavior tokens
- V: target-side feature tokens

Compared with the HyFormer project, this dataset intentionally does not build
coarse features, target-sequence handcrafted features, or query branches. It
only exposes the raw ingredients needed to construct the unified token stream
described in the TokenFormer paper.
"""

from __future__ import annotations

import gc
import glob
import json
import logging
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, IterableDataset

torch.multiprocessing.set_sharing_strategy("file_system")


class FeatureSchema:
    """Schema helper that records ``(feature_id, offset, length)`` entries."""

    def __init__(self) -> None:
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": self.entries, "total_dim": self.total_dim}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSchema":
        schema = cls()
        for fid, offset, length in data["entries"]:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = int(data["total_dim"])
        return schema


class TokenFormerParquetDataset(IterableDataset):
    """Reads raw multi-column parquet and exposes TokenFormer-ready tensors."""

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
    ) -> None:
        super().__init__()
        if os.path.isdir(parquet_path):
            files = sorted(glob.glob(os.path.join(parquet_path, "*.parquet")))
            if not files:
                raise FileNotFoundError(f"No .parquet files found in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.buffer_batches = int(buffer_batches)
        self.clip_vocab = bool(clip_vocab)
        self.is_training = bool(is_training)
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        self._rg_list: List[Tuple[str, int, int]] = []
        for file_path in self._parquet_files:
            pf = pq.ParquetFile(file_path)
            for rg_idx in range(pf.metadata.num_row_groups):
                self._rg_list.append(
                    (file_path, rg_idx, int(pf.metadata.row_group(rg_idx).num_rows))
                )
        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]
        self.num_rows = sum(row_count for _, _, row_count in self._rg_list)

        self._load_schema(schema_path, seq_max_lens or {})

        pf = pq.ParquetFile(self._parquet_files[0])
        self._col_idx = {name: idx for idx, name in enumerate(pf.schema_arrow.names)}

        self._user_int_plan: List[Tuple[int, int, int, int]] = []
        offset = 0
        for fid, vocab_size, length in self._user_int_cols:
            self._user_int_plan.append(
                (self._col_idx[f"user_int_feats_{fid}"], int(length), offset, int(vocab_size))
            )
            offset += int(length)

        self._item_int_plan: List[Tuple[int, int, int, int]] = []
        offset = 0
        for fid, vocab_size, length in self._item_int_cols:
            self._item_int_plan.append(
                (self._col_idx[f"item_int_feats_{fid}"], int(length), offset, int(vocab_size))
            )
            offset += int(length)

        self._user_dense_plan: List[Tuple[int, int, int]] = []
        offset = 0
        for fid, length in self._user_dense_cols:
            self._user_dense_plan.append(
                (self._col_idx[f"user_dense_feats_{fid}"], int(length), offset)
            )
            offset += int(length)

        self._seq_plan: Dict[str, Tuple[List[Tuple[int, int, int]], Optional[int]]] = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            ts_fid = self.ts_fids[domain]
            side_plan: List[Tuple[int, int, int]] = []
            for slot, fid in enumerate(self.sideinfo_fids[domain]):
                side_plan.append(
                    (
                        self._col_idx[f"{prefix}_{fid}"],
                        slot,
                        int(self.seq_vocab_sizes[domain][fid]),
                    )
                )
            ts_ci = self._col_idx.get(f"{prefix}_{ts_fid}") if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            "TokenFormerParquetDataset: rows=%s files=%s batch_size=%s shuffle=%s",
            self.num_rows,
            len(self._parquet_files),
            self.batch_size,
            self.shuffle,
        )

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        with open(schema_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._user_int_cols: List[List[int]] = raw["user_int"]
        self.user_int_schema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vocab_size, length in self._user_int_cols:
            self.user_int_schema.add(int(fid), int(length))
            self.user_int_vocab_sizes.extend([int(vocab_size)] * int(length))

        self._item_int_cols: List[List[int]] = raw["item_int"]
        self.item_int_schema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vocab_size, length in self._item_int_cols:
            self.item_int_schema.add(int(fid), int(length))
            self.item_int_vocab_sizes.extend([int(vocab_size)] * int(length))

        self._user_dense_cols: List[List[int]] = raw["user_dense"]
        self.user_dense_schema = FeatureSchema()
        for fid, length in self._user_dense_cols:
            self.user_dense_schema.add(int(fid), int(length))

        self.item_dense_schema = FeatureSchema()

        self._seq_cfg: Dict[str, Dict[str, Any]] = raw["seq"]
        self.seq_domains = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = str(cfg["prefix"])
            ts_fid = cfg.get("ts_fid")
            self.ts_fids[domain] = int(ts_fid) if ts_fid is not None else None
            all_fids = [int(fid) for fid, _ in cfg["features"]]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {
                int(fid): int(vocab_size) for fid, vocab_size in cfg["features"]
            }
            sideinfo = [fid for fid in all_fids if fid != self.ts_fids[domain]]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]
            self._seq_maxlen[domain] = int(seq_max_lens.get(domain, 256))

    def __len__(self) -> int:
        return sum((count + self.batch_size - 1) // self.batch_size for _, _, count in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [
                rg for idx, rg in enumerate(rg_list) if idx % worker_info.num_workers == worker_info.id
            ]

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

    def _flush_buffer(self, buffer: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for key in buffer[0].keys():
            if isinstance(buffer[0][key], torch.Tensor):
                merged[key] = torch.cat([b[key] for b in buffer], dim=0)
            else:
                non_tensor_keys[key] = buffer[0][key]
        total_rows = merged["label"].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for start in range(0, total_rows, self.batch_size):
            end = min(start + self.batch_size, total_rows)
            sliced: Dict[str, Any] = {k: v[rand_idx[start:end]] for k, v in merged.items()}
            sliced.update(non_tensor_keys)
            yield sliced
        buffer.clear()

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: np.ndarray,
        vocab_size: int,
    ) -> None:
        oob_mask = arr >= vocab_size
        if not np.any(oob_mask):
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        count = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            stats = self._oob_stats[key]
            stats["count"] += count
            stats["max"] = max(stats["max"], mx)
            stats["min_oob"] = min(stats["min_oob"], mn)
        else:
            self._oob_stats[key] = {
                "count": count,
                "max": mx,
                "min_oob": mn,
                "vocab": int(vocab_size),
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx} has values outside [0, {vocab_size})."
            )

    def _pad_varlen_int_column(
        self,
        arrow_col: pa.Array,
        max_len: int,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()
        padded = np.zeros((batch_size, max_len), dtype=np.int64)
        lengths = np.zeros(batch_size, dtype=np.int64)
        for row_idx in range(batch_size):
            start = int(offsets[row_idx])
            end = int(offsets[row_idx + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[row_idx, :use_len] = values[start:start + use_len]
            lengths[row_idx] = use_len
        padded[padded <= 0] = 0
        return padded, lengths

    def _pad_varlen_float_column(
        self,
        arrow_col: pa.Array,
        max_dim: int,
        batch_size: int,
    ) -> np.ndarray:
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()
        padded = np.zeros((batch_size, max_dim), dtype=np.float32)
        for row_idx in range(batch_size):
            start = int(offsets[row_idx])
            end = int(offsets[row_idx + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[row_idx, :use_len] = values[start:start + use_len]
        return padded

    def _pad_seq_timestamp_column(
        self,
        arrow_col: pa.Array,
        max_len: int,
        batch_size: int,
    ) -> np.ndarray:
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()
        padded = np.zeros((batch_size, max_len), dtype=np.int64)
        for row_idx in range(batch_size):
            start = int(offsets[row_idx])
            end = int(offsets[row_idx + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[row_idx, :use_len] = values[start:start + use_len]
        padded[padded <= 0] = 0
        return padded

    def _convert_batch(self, batch: pa.RecordBatch) -> Dict[str, Any]:
        batch_size = batch.num_rows
        timestamps = batch.column(self._col_idx["timestamp"]).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (
                batch.column(self._col_idx["label_type"])
                .fill_null(0)
                .to_numpy(zero_copy_only=False)
                .astype(np.int64)
                == 2
            ).astype(np.int64)
        else:
            labels = np.zeros(batch_size, dtype=np.int64)

        user_int = np.zeros((batch_size, self.user_int_schema.total_dim), dtype=np.int64)
        for col_idx, dim, offset, vocab_size in self._user_int_plan:
            col = batch.column(col_idx)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vocab_size > 0:
                    self._record_oob("user_int", col_idx, arr, vocab_size)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, batch_size)
                if vocab_size > 0:
                    self._record_oob("user_int", col_idx, padded, vocab_size)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        item_int = np.zeros((batch_size, self.item_int_schema.total_dim), dtype=np.int64)
        for col_idx, dim, offset, vocab_size in self._item_int_plan:
            col = batch.column(col_idx)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vocab_size > 0:
                    self._record_oob("item_int", col_idx, arr, vocab_size)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, batch_size)
                if vocab_size > 0:
                    self._record_oob("item_int", col_idx, padded, vocab_size)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        user_dense = np.zeros((batch_size, self.user_dense_schema.total_dim), dtype=np.float32)
        for col_idx, dim, offset in self._user_dense_plan:
            user_dense[:, offset:offset + dim] = self._pad_varlen_float_column(
                batch.column(col_idx), dim, batch_size
            )

        result: Dict[str, Any] = {
            "user_int_feats": torch.from_numpy(user_int),
            "user_dense_feats": torch.from_numpy(user_dense),
            "item_int_feats": torch.from_numpy(item_int),
            "item_dense_feats": torch.zeros(batch_size, 0, dtype=torch.float32),
            "label": torch.from_numpy(labels),
            "timestamp": torch.from_numpy(timestamps),
            "_seq_domains": self.seq_domains,
        }

        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]
            num_fields = len(side_plan)
            seq_out = np.zeros((batch_size, num_fields, max_len), dtype=np.int64)
            seq_lens = np.zeros(batch_size, dtype=np.int64)

            for slot, (col_idx, _, vocab_size) in enumerate(side_plan):
                padded, lengths = self._pad_varlen_int_column(
                    batch.column(col_idx), max_len, batch_size
                )
                if vocab_size > 0:
                    self._record_oob(f"seq_{domain}", col_idx, padded, vocab_size)
                else:
                    padded[:] = 0
                seq_out[:, slot, :] = padded
                seq_lens = np.maximum(seq_lens, lengths)

            if ts_ci is not None:
                seq_raw_ts = self._pad_seq_timestamp_column(
                    batch.column(ts_ci), max_len, batch_size
                )
            else:
                seq_raw_ts = np.zeros((batch_size, max_len), dtype=np.int64)

            result[domain] = torch.from_numpy(seq_out)
            result[f"{domain}_len"] = torch.from_numpy(seq_lens)
            result[f"{domain}_raw_ts"] = torch.from_numpy(seq_raw_ts)

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
) -> Tuple[DataLoader, DataLoader, TokenFormerParquetDataset]:
    """Build train/valid loaders from raw parquet files."""

    random.seed(seed)

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

    rg_info: List[Tuple[str, int, int]] = []
    for file_path in parquet_files:
        pf = pq.ParquetFile(file_path)
        for rg_idx in range(pf.metadata.num_row_groups):
            rg_info.append((file_path, rg_idx, int(pf.metadata.row_group(rg_idx).num_rows)))

    total_rgs = len(rg_info)
    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info("train_ratio=%s -> using %s training row groups", train_ratio, n_train_rgs)

    train_rows = sum(row_count for _, _, row_count in rg_info[:n_train_rgs])
    valid_rows = sum(row_count for _, _, row_count in rg_info[n_train_rgs:])
    logging.info(
        "Row-group split: train=%s (%s rows) valid=%s (%s rows)",
        n_train_rgs,
        train_rows,
        total_rgs - n_train_rgs,
        valid_rows,
    )

    train_dataset = TokenFormerParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
        is_training=True,
    )
    valid_dataset = TokenFormerParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
        is_training=False,
    )

    use_cuda = torch.cuda.is_available()
    train_loader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=use_cuda,
        **train_loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=use_cuda,
    )

    logging.info(
        "TokenFormer loaders ready: train_rows=%s valid_rows=%s batch_size=%s",
        train_rows,
        valid_rows,
        batch_size,
    )
    return train_loader, valid_loader, train_dataset
