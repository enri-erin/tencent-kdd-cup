"""TokenFormer inference / offline evaluation entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FeatureSchema, TokenFormerParquetDataset
from model import ModelInput, TokenFormer
from utils import create_logger


def build_sparse_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    specs: List[Tuple[int, int, int]] = []
    for _, offset, length in schema.entries:
        vocab_size = max(per_position_vocab_sizes[offset:offset + length]) if length > 0 else 1
        specs.append((int(vocab_size), int(offset), int(length)))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TokenFormer Inference")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory containing model.pt")
    parser.add_argument("--data_dir", type=str, required=True, help="Parquet directory or parquet file")
    parser.add_argument("--schema_path", type=str, default=None, help="Schema JSON path")
    parser.add_argument("--output_path", type=str, default="predictions.npy")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--evaluate", action="store_true", default=False, help="Compute AUC/LogLoss if labels exist")
    return parser.parse_args()


def load_train_config(ckpt_dir: str) -> Dict[str, object]:
    with open(os.path.join(ckpt_dir, "train_config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def make_model_input(batch: Dict[str, object]) -> ModelInput:
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


def batch_to_device(batch: Dict[str, object], device: str) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def main() -> None:
    args = parse_args()
    model_ckpt_dir = args.ckpt_dir
    if not os.path.exists(os.path.join(model_ckpt_dir, "model.pt")):
        fallback_dir = os.path.join(model_ckpt_dir, "best_model")
        if os.path.exists(os.path.join(fallback_dir, "model.pt")):
            model_ckpt_dir = fallback_dir

    create_logger(os.path.join(model_ckpt_dir, "infer.log"))
    train_cfg = load_train_config(model_ckpt_dir)

    if args.schema_path is None:
        local_schema = os.path.join(model_ckpt_dir, "schema.json")
        if os.path.exists(local_schema):
            args.schema_path = local_schema
        else:
            args.schema_path = str(Path(args.data_dir) / "schema.json")

    dataset = TokenFormerParquetDataset(
        parquet_path=args.data_dir,
        schema_path=args.schema_path,
        batch_size=args.batch_size,
        seq_max_lens={k: int(v) for k, v in dict(train_cfg["seq_max_lens"]).items()},
        shuffle=False,
        buffer_batches=0,
        clip_vocab=True,
        is_training=bool(args.evaluate),
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    user_int_specs = build_sparse_feature_specs(dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_specs = build_sparse_feature_specs(dataset.item_int_schema, dataset.item_int_vocab_sizes)
    user_dense_specs = [(int(fid), int(offset), int(length)) for fid, offset, length in dataset.user_dense_schema.entries]

    model = TokenFormer(
        user_int_feature_specs=user_int_specs,
        user_dense_feature_specs=user_dense_specs,
        item_int_feature_specs=item_int_specs,
        seq_domain_vocab_sizes=dataset.seq_domain_vocab_sizes,
        max_behavior_tokens=int(train_cfg["max_behavior_tokens"]),
        d_model=int(train_cfg["d_model"]),
        emb_dim=int(train_cfg["emb_dim"]),
        num_heads=int(train_cfg["num_heads"]),
        num_layers=int(train_cfg["num_layers"]),
        full_attention_layers=int(train_cfg["full_attention_layers"]),
        sliding_window_sizes=[int(v) for v in train_cfg["sliding_window_sizes"]],
        ffn_mult=int(train_cfg["ffn_mult"]),
        dropout=float(train_cfg["dropout_rate"]),
        rope_base=float(train_cfg["rope_base"]),
        action_dim=2,
    ).to(args.device)
    state = torch.load(os.path.join(model_ckpt_dir, "model.pt"), map_location=args.device)
    model.load_state_dict(state, strict=True)
    model.eval()

    probs_list = []
    labels_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="infer", leave=False):
            batch = batch_to_device(batch, args.device)
            logits = model(make_model_input(batch))
            probs = torch.softmax(logits, dim=-1)[:, 1]
            probs_list.append(probs.detach().cpu())
            if args.evaluate:
                labels_list.append(batch["label"].detach().cpu())

    probs_np = torch.cat(probs_list).numpy()
    np.save(args.output_path, probs_np)
    logging.info("Saved predictions to %s", args.output_path)

    if args.evaluate and labels_list:
        labels_np = torch.cat(labels_list).numpy()
        auc = float(roc_auc_score(labels_np, probs_np))
        ll = float(log_loss(labels_np, np.clip(probs_np, 1e-7, 1.0 - 1e-7)))
        logging.info("Evaluation | AUC=%.6f LogLoss=%.6f", auc, ll)


if __name__ == "__main__":
    main()
