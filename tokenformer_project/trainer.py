"""TokenFormer trainer."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ModelInput


class TokenFormerTrainer:
    """AUC-monitored trainer for the TokenFormer project."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        weight_decay: float,
        num_epochs: int,
        patience: int,
        device: str,
        save_dir: str,
        schema_path: Optional[str] = None,
        train_config: Optional[Dict[str, Any]] = None,
        eval_every_n_steps: int = 0,
        use_amp: bool = False,
        grad_clip_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = int(num_epochs)
        self.patience = int(patience)
        self.device = device
        self.save_dir = save_dir
        self.schema_path = schema_path
        self.train_config = train_config or {}
        self.eval_every_n_steps = int(eval_every_n_steps)
        self.use_amp = bool(use_amp and str(device).startswith("cuda"))
        self.grad_clip_norm = float(grad_clip_norm)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )
        self.amp_dtype = (
            torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
        )
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.best_auc = -math.inf
        self.best_logloss = math.inf
        self.bad_eval_count = 0
        self.global_step = 0

        logging.info(
            "TokenFormerTrainer lr=%s weight_decay=%s use_amp=%s amp_dtype=%s",
            lr,
            weight_decay,
            self.use_amp,
            self.amp_dtype,
        )

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    def _make_model_input(self, batch: Dict[str, Any]) -> ModelInput:
        seq_domains = batch["_seq_domains"]
        return ModelInput(
            user_int_feats=batch["user_int_feats"],
            item_int_feats=batch["item_int_feats"],
            user_dense_feats=batch["user_dense_feats"],
            item_dense_feats=batch["item_dense_feats"],
            seq_data={domain: batch[domain] for domain in seq_domains},
            seq_lens={domain: batch[f"{domain}_len"] for domain in seq_domains},
            seq_raw_timestamps={domain: batch[f"{domain}_raw_ts"] for domain in seq_domains},
        )

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, os.path.join(ckpt_dir, "schema.json"))
        if self.train_config:
            with open(os.path.join(ckpt_dir, "train_config.json"), "w", encoding="utf-8") as f:
                json.dump(self.train_config, f, indent=2)

    def _save_best_checkpoint(self) -> None:
        ckpt_dir = os.path.join(self.save_dir, "best_model")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info("Saved best checkpoint to %s", ckpt_dir)

    def _forward_logits(self, batch: Dict[str, Any]) -> torch.Tensor:
        model_input = self._make_model_input(batch)
        with torch.amp.autocast(
            device_type="cuda",
            enabled=self.use_amp,
            dtype=self.amp_dtype,
        ):
            logits = self.model(model_input)
        return logits

    def _train_step(self, batch: Dict[str, Any]) -> float:
        batch = self._batch_to_device(batch)
        labels = batch["label"].long()

        self.optimizer.zero_grad(set_to_none=True)
        logits = self._forward_logits(batch)
        loss = F.cross_entropy(logits, labels)

        if self.use_amp:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm, foreach=False)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm, foreach=False)
            self.optimizer.step()
        return float(loss.detach().item())

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        probs_list = []
        labels_list = []
        for batch in tqdm(self.valid_loader, desc="valid", leave=False):
            batch = self._batch_to_device(batch)
            logits = self._forward_logits(batch)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            probs_list.append(probs.detach().cpu())
            labels_list.append(batch["label"].detach().cpu())

        probs_np = torch.cat(probs_list).numpy()
        labels_np = torch.cat(labels_list).numpy()

        try:
            auc = float(roc_auc_score(labels_np, probs_np))
        except ValueError:
            auc = 0.0
        try:
            ll = float(log_loss(labels_np, np.clip(probs_np, 1e-7, 1.0 - 1e-7)))
        except ValueError:
            ll = float("inf")
        self.model.train()
        return auc, ll

    def _maybe_validate(self, total_step: int, epoch: int) -> bool:
        auc, ll = self.evaluate()
        logging.info("Validation epoch=%s step=%s | AUC=%.6f LogLoss=%.6f", epoch, total_step, auc, ll)
        if auc > self.best_auc:
            self.best_auc = auc
            self.best_logloss = ll
            self.bad_eval_count = 0
            self._save_best_checkpoint()
        else:
            self.bad_eval_count += 1
            logging.info("No improvement. patience=%s/%s", self.bad_eval_count, self.patience)
        return self.bad_eval_count >= self.patience

    def fit(self) -> Dict[str, float]:
        self.model.train()
        for epoch in range(1, self.num_epochs + 1):
            running_loss = 0.0
            num_batches = 0
            progress = tqdm(self.train_loader, desc=f"train-{epoch}", leave=False)
            for batch in progress:
                loss = self._train_step(batch)
                running_loss += loss
                num_batches += 1
                self.global_step += 1
                progress.set_postfix(loss=f"{loss:.4f}")

                if self.eval_every_n_steps > 0 and self.global_step % self.eval_every_n_steps == 0:
                    should_stop = self._maybe_validate(self.global_step, epoch)
                    if should_stop:
                        logging.info("Early stopping triggered at step %s", self.global_step)
                        return {
                            "best_val_auc": self.best_auc,
                            "best_val_logloss": self.best_logloss,
                        }

            avg_loss = running_loss / max(num_batches, 1)
            logging.info("Epoch %s finished | train_loss=%.6f", epoch, avg_loss)
            if self.eval_every_n_steps <= 0:
                should_stop = self._maybe_validate(self.global_step, epoch)
                if should_stop:
                    logging.info("Early stopping triggered after epoch %s", epoch)
                    break

        return {"best_val_auc": self.best_auc, "best_val_logloss": self.best_logloss}
