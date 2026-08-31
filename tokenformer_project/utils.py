import os
import random
import copy
import logging
import time
from datetime import timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogFormatter:
    """Custom ``logging.Formatter`` that prefixes every record with the
    wall-clock timestamp and the elapsed wall-clock time since this
    formatter instance was constructed.

    The prefix format is ``"<locale-date> <locale-time> - H:MM:SS"``, which
    is convenient for tracking long-running training runs where both the
    absolute time and the time-since-start are useful.

    Multi-line messages are re-indented so that continuation lines align
    with the beginning of the message (not the prefix).
    """

    def __init__(self) -> None:
        # Anchor used to compute the elapsed-time part of the log prefix.
        # Can be reset at runtime via ``create_logger(...).reset_time()``.
        self.start_time: float = time.time()

    def format(self, record: logging.LogRecord) -> str:
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s" % (
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        # Indent continuation lines so they line up with the message body,
        # not with the timestamp prefix.
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(filepath: str) -> logging.Logger:
    """Create and configure the root logger for a training/inference run.

    The returned logger has two handlers attached:

    * A ``FileHandler`` bound to ``filepath`` (opened in write mode,
      truncating any previous content) that records ``DEBUG``-level and
      above messages for post-mortem inspection.
    * A ``StreamHandler`` to stderr that only echoes ``INFO``-level and
      above messages, keeping the console output concise.

    Both handlers share a ``LogFormatter`` so the console and the log file
    stay in sync. Any pre-existing handlers on the root logger are removed
    to avoid duplicate lines when this function is called multiple times.

    Args:
        filepath: Destination path of the log file. Opened in ``"w"`` mode,
            so previous contents are overwritten.

    Returns:
        The root ``logging.Logger`` instance. The returned object is
        augmented with a ``reset_time()`` attribute that resets the
        elapsed-time clock used by the log prefix. This is useful when the
        "interesting" phase of a run starts well after process launch
        (e.g. after schema building and data loading).
    """
    log_formatter = LogFormatter()

    file_handler = logging.FileHandler(filepath, "w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Allow callers to reset the elapsed-time clock shown in the log prefix.
    def reset_time() -> None:
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time  # type: ignore[attr-defined]

    return logger


class EarlyStopping:
    """Early-stop training when the validation metric plateaus.

    The tracker assumes a *higher-is-better* metric (typical for AUC or
    accuracy). A candidate ``score`` is considered an improvement iff
    ``score > best_score + delta``; otherwise the internal ``counter`` is
    incremented and training is requested to stop once
    ``counter >= patience``.

    On every improvement the current ``model.state_dict()`` is both
    deep-copied in memory (``self.best_model``) and persisted to disk at
    ``checkpoint_path``. The most recent *improving* score is cached in
    ``self.best_saved_score`` so callers can skip redundant IO.

    Attributes:
        checkpoint_path: Destination path for the best ``state_dict``.
        patience: Number of non-improving calls tolerated before
            ``early_stop`` is flipped to ``True``.
        verbose: If ``True``, emit an ``INFO`` line whenever a checkpoint
            is written.
        counter: Number of consecutive non-improving calls seen so far.
        best_score: Best score observed; ``None`` until the first call.
        early_stop: Set to ``True`` once ``counter >= patience``.
        delta: Minimum absolute improvement required to reset ``counter``.
        best_model: In-memory deep copy of the best ``state_dict``.
        best_saved_score: Score associated with the last checkpoint
            actually written to disk.
        best_extra_metrics: Optional auxiliary metrics captured at the
            best-score step (e.g. logloss, other AUCs).
        label: Short prefix (e.g. ``"val"``) prepended to log lines to
            disambiguate multiple trackers running in parallel.
    """

    def __init__(
        self,
        checkpoint_path: str,
        label: str = "",
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0,
    ) -> None:
        self.checkpoint_path: str = checkpoint_path
        self.patience: int = patience
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.delta: float = delta
        self.best_model: Optional[Dict[str, torch.Tensor]] = None
        self.best_saved_score: float = 0.0
        self.best_extra_metrics: Optional[Dict[str, Any]] = None
        self.label: str = label
        if self.label != "":
            self.label += " "

    def _is_not_improved(self, score: float) -> bool:
        """Return ``True`` iff ``score`` fails to beat ``best_score + delta``.

        Used as the gating condition for incrementing the patience counter.
        ``best_score`` must have been seeded by a prior ``__call__``.
        """
        assert self.best_score is not None, "call __call__ first to seed best_score"
        if score > self.best_score + self.delta:
            return False
        return True

    def __call__(
        self,
        score: float,
        model: nn.Module,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Feed a new validation score into the tracker.

        Three branches, in order:

        1. First call (``best_score is None``): seed the tracker, persist a
           checkpoint, and cache the model weights.
        2. Not improved: increment ``counter`` and log the progress; flip
           ``early_stop`` once ``counter >= patience``.
        3. Improved: reset ``counter`` to ``0``, update ``best_score`` and
           ``best_extra_metrics``, refresh the in-memory ``best_model``,
           and write a new checkpoint to disk.

        Args:
            score: Scalar validation metric (higher is better, e.g. AUC).
            model: Model whose ``state_dict`` is snapshotted on
                improvement. Only the parameters are saved, not the
                optimizer state.
            extra_metrics: Optional dict of auxiliary metrics recorded at
                the same step, e.g.
                ``{"best_val_AUC": ..., "best_val_logloss": ...}``. Stored
                verbatim as ``self.best_extra_metrics``; not interpreted
                by ``EarlyStopping`` itself.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_extra_metrics = extra_metrics
            self.best_saved_score = 0.0
            self.save_checkpoint(score, model)
            self.best_model = copy.deepcopy(model.state_dict())
        elif self._is_not_improved(score):
            self.counter += 1
            logging.info(f'{self.label}earlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            logging.info(f'{self.label}earlyStopping counter reset!')
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_extra_metrics = extra_metrics
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score: float, model: nn.Module) -> None:
        """Persist ``model.state_dict()`` to ``self.checkpoint_path``.

        Creates any missing parent directories, writes atomically via
        ``torch.save``, and records ``score`` as ``self.best_saved_score``
        so subsequent callers can detect "no new improvement since last
        save" without re-reading the checkpoint file.

        Args:
            score: Validation score associated with the weights being
                saved. Exposed to callers via ``best_saved_score`` after
                the write completes.
            model: Model whose parameters are being snapshotted. Only
                ``state_dict()`` is written; optimizer and scheduler state
                are explicitly *not* included.
        """
        if self.verbose:
            logging.info('Validation score increased. Saving model ...')
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
        self.best_saved_score = score


def set_seed(seed: int) -> None:
    """Seed every RNG that can influence training reproducibility.

    Seeds ``random``, the ``PYTHONHASHSEED`` env var, NumPy, the CPU
    PyTorch generator and all CUDA generators, then forces cuDNN into
    deterministic mode.

    Note that full bitwise determinism on GPU also requires disabling
    cuDNN auto-tuning (``torch.backends.cudnn.benchmark = False``) and may
    come with a non-trivial throughput cost; this helper intentionally
    only toggles ``deterministic`` to preserve speed for common use cases.

    Args:
        seed: Non-negative integer seed shared by all RNGs listed above.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: (N,) raw logits (before sigmoid).
        targets: (N,) binary labels {0, 1}.
        alpha: positive-class weight in (0, 1). When positives dominate,
            use alpha < 0.5 to downweight the positive class.
        gamma: focusing parameter. gamma=0 degenerates to standard BCE;
            gamma=2 is the standard value.
        reduction: 'mean' | 'sum' | 'none'.
    """
    p = torch.sigmoid(logits)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * bce_loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


class IATHistoryStore:
    """GPU-backed per-user InsEmb store used by the IAT branch."""

    def __init__(
        self,
        history_len: int,
        ins_emb_dim: int,
        bucket_boundaries: np.ndarray,
        task_hash_size: int = 65536,
        device: str = 'cpu',
        initial_capacity: int = 4096,
    ) -> None:
        self.history_len = history_len
        self.ins_emb_dim = ins_emb_dim
        self.device = torch.device(device)
        self.bucket_boundaries = torch.as_tensor(
            np.asarray(bucket_boundaries, dtype=np.int64),
            dtype=torch.long,
            device=self.device,
        )
        self.task_hash_size = task_hash_size
        self.capacity = max(int(initial_capacity), 1)
        self._slot_by_uid: Dict[Any, int] = {}
        self._next_slot = 0
        self._hist_ins_emb = torch.zeros(
            self.capacity,
            self.history_len,
            self.ins_emb_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self._hist_labels = torch.zeros(
            self.capacity,
            self.history_len,
            dtype=torch.long,
            device=self.device,
        )
        self._hist_timestamps = torch.zeros(
            self.capacity,
            self.history_len,
            dtype=torch.long,
            device=self.device,
        )
        self._hist_task_ids = torch.zeros(
            self.capacity,
            self.history_len,
            dtype=torch.long,
            device=self.device,
        )

    def clear(self) -> None:
        self._slot_by_uid.clear()
        self._next_slot = 0

    def clone(self) -> "IATHistoryStore":
        cloned = IATHistoryStore(
            history_len=self.history_len,
            ins_emb_dim=self.ins_emb_dim,
            bucket_boundaries=self.bucket_boundaries.detach().cpu().numpy(),
            task_hash_size=self.task_hash_size,
            device=str(self.device),
            initial_capacity=self.capacity,
        )
        cloned._slot_by_uid = dict(self._slot_by_uid)
        cloned._next_slot = self._next_slot
        if self._next_slot > 0:
            sl = slice(0, self._next_slot)
            cloned._hist_ins_emb[sl].copy_(self._hist_ins_emb[sl])
            cloned._hist_labels[sl].copy_(self._hist_labels[sl])
            cloned._hist_timestamps[sl].copy_(self._hist_timestamps[sl])
            cloned._hist_task_ids[sl].copy_(self._hist_task_ids[sl])
        return cloned

    @staticmethod
    def _normalize_uid(uid: Any) -> Any:
        if isinstance(uid, (int, np.integer)):
            return int(uid)
        if isinstance(uid, str):
            return uid
        try:
            return int(uid)
        except Exception:
            return str(uid)

    def _ensure_capacity(self, target_slots: int) -> None:
        if target_slots <= self.capacity:
            return
        new_capacity = self.capacity
        while new_capacity < target_slots:
            new_capacity *= 2

        def _grow(old: torch.Tensor) -> torch.Tensor:
            new = torch.zeros(
                (new_capacity, *old.shape[1:]),
                dtype=old.dtype,
                device=old.device,
            )
            if self._next_slot > 0:
                new[:self._next_slot].copy_(old[:self._next_slot])
            return new

        self._hist_ins_emb = _grow(self._hist_ins_emb)
        self._hist_labels = _grow(self._hist_labels)
        self._hist_timestamps = _grow(self._hist_timestamps)
        self._hist_task_ids = _grow(self._hist_task_ids)
        self.capacity = new_capacity

    def _hash_task_id(self, task_id: int) -> int:
        if task_id <= 0:
            return 0
        return int(task_id % self.task_hash_size) + 1

    def _get_or_create_slot(self, uid: Any) -> int:
        key = self._normalize_uid(uid)
        slot = self._slot_by_uid.get(key)
        if slot is not None:
            return slot
        slot = self._next_slot
        self._ensure_capacity(slot + 1)
        self._slot_by_uid[key] = slot
        self._next_slot += 1
        self._hist_ins_emb[slot].zero_()
        self._hist_labels[slot].zero_()
        self._hist_timestamps[slot].zero_()
        self._hist_task_ids[slot].zero_()
        return slot

    def build_batch(
        self,
        user_ids: List[Any],
        timestamps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B = len(user_ids)
        hist_ins_emb = torch.zeros(
            B, self.history_len, self.ins_emb_dim,
            dtype=torch.float32, device=self.device,
        )
        hist_labels = torch.zeros(B, self.history_len, dtype=torch.long, device=self.device)
        hist_time_buckets = torch.zeros(B, self.history_len, dtype=torch.long, device=self.device)
        hist_task_ids = torch.zeros(B, self.history_len, dtype=torch.long, device=self.device)
        hist_mask = torch.ones(B, self.history_len, dtype=torch.bool, device=self.device)

        if B == 0 or self._next_slot == 0:
            return {
                'iat_hist_ins_emb': hist_ins_emb,
                'iat_hist_labels': hist_labels,
                'iat_hist_time_buckets': hist_time_buckets,
                'iat_hist_task_ids': hist_task_ids,
                'iat_hist_mask': hist_mask,
            }

        slot_indices = [self._slot_by_uid.get(self._normalize_uid(uid), -1) for uid in user_ids]
        existing_rows = [i for i, slot in enumerate(slot_indices) if slot >= 0]
        if existing_rows:
            row_tensor = torch.as_tensor(existing_rows, dtype=torch.long, device=self.device)
            slot_tensor = torch.as_tensor(
                [slot_indices[i] for i in existing_rows],
                dtype=torch.long,
                device=self.device,
            )
            current_ts = timestamps.to(self.device, non_blocking=True)[row_tensor]
            gathered_ts = self._hist_timestamps.index_select(0, slot_tensor)
            valid_mask = (gathered_ts > 0) & (gathered_ts < current_ts.unsqueeze(1))
            if valid_mask.any():
                hist_ins_emb[row_tensor] = self._hist_ins_emb.index_select(0, slot_tensor)
                hist_labels[row_tensor] = self._hist_labels.index_select(0, slot_tensor)
                hist_task_ids[row_tensor] = self._hist_task_ids.index_select(0, slot_tensor)
                hist_mask[row_tensor] = ~valid_mask
                deltas = (current_ts.unsqueeze(1) - gathered_ts).clamp_min(0)
                bucketized = torch.bucketize(deltas, self.bucket_boundaries) + 1
                bucketized = torch.minimum(
                    bucketized,
                    torch.full_like(bucketized, self.bucket_boundaries.numel()),
                )
                hist_time_buckets[row_tensor] = bucketized * valid_mask.long()

        return {
            'iat_hist_ins_emb': hist_ins_emb,
            'iat_hist_labels': hist_labels,
            'iat_hist_time_buckets': hist_time_buckets,
            'iat_hist_task_ids': hist_task_ids,
            'iat_hist_mask': hist_mask,
        }

    def update(
        self,
        user_ids: List[Any],
        timestamps: torch.Tensor,
        current_ins_emb: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
    ) -> None:
        ts_list = timestamps.detach().cpu().tolist()
        emb_dev = current_ins_emb.detach().to(self.device, dtype=torch.float32)
        if labels is not None:
            label_dev = labels.detach().to(self.device, dtype=torch.long) + 1
        else:
            label_dev = torch.zeros(len(user_ids), dtype=torch.long, device=self.device)
        if task_ids is not None:
            task_list = task_ids.detach().cpu().tolist()
        else:
            task_list = [0 for _ in user_ids]

        for row_idx, (uid, ts, task_id) in enumerate(zip(user_ids, ts_list, task_list)):
            slot = self._get_or_create_slot(uid)
            ts_int = int(ts)
            current_label = int(label_dev[row_idx].item())
            current_task = self._hash_task_id(int(task_id))

            hist_ts = self._hist_timestamps[slot]
            if hist_ts[-1].item() > 0 and ts_int <= hist_ts[-1].item():
                continue
            insert_pos = 0
            if hist_ts[0].item() == 0 or hist_ts[0].item() <= ts_int:
                insert_pos = 0
            else:
                greater = torch.nonzero(hist_ts > ts_int, as_tuple=False)
                insert_pos = int(greater.shape[0])
                insert_pos = min(insert_pos, self.history_len - 1)

            if insert_pos == 0:
                self._hist_ins_emb[slot, 1:] = self._hist_ins_emb[slot, :-1].clone()
                self._hist_labels[slot, 1:] = self._hist_labels[slot, :-1].clone()
                self._hist_timestamps[slot, 1:] = self._hist_timestamps[slot, :-1].clone()
                self._hist_task_ids[slot, 1:] = self._hist_task_ids[slot, :-1].clone()
            else:
                self._hist_ins_emb[slot, insert_pos + 1:] = self._hist_ins_emb[slot, insert_pos:-1].clone()
                self._hist_labels[slot, insert_pos + 1:] = self._hist_labels[slot, insert_pos:-1].clone()
                self._hist_timestamps[slot, insert_pos + 1:] = self._hist_timestamps[slot, insert_pos:-1].clone()
                self._hist_task_ids[slot, insert_pos + 1:] = self._hist_task_ids[slot, insert_pos:-1].clone()

            self._hist_ins_emb[slot, insert_pos] = emb_dev[row_idx]
            self._hist_labels[slot, insert_pos] = current_label
            self._hist_timestamps[slot, insert_pos] = ts_int
            self._hist_task_ids[slot, insert_pos] = current_task
