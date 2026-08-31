"""TokenFormer training entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from dataset import FeatureSchema, get_pcvr_data
from model import TokenFormer
from trainer import TokenFormerTrainer
from utils import create_logger, set_seed


def build_sparse_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    specs: List[Tuple[int, int, int]] = []
    for _, offset, length in schema.entries:
        vocab_size = max(per_position_vocab_sizes[offset:offset + length]) if length > 0 else 1
        specs.append((int(vocab_size), int(offset), int(length)))
    return specs


def parse_seq_max_lens(text: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for chunk in text.split(","):
        if not chunk:
            continue
        name, value = chunk.split(":")
        mapping[name.strip()] = int(value)
    return mapping


def parse_int_list(text: str) -> List[int]:
    if not text.strip():
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TokenFormer Training")

    parser.add_argument("--data_dir", type=str, default=None, help="Training parquet directory")
    parser.add_argument("--schema_path", type=str, default=None, help="Schema JSON path")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default=None, help="Log directory")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--buffer_batches", type=int, default=20)
    parser.add_argument("--train_ratio", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--eval_every_n_steps", type=int, default=0)
    parser.add_argument(
        "--seq_max_lens",
        type=str,
        default="seq_a:256,seq_b:256,seq_c:512,seq_d:512",
        help="Per-domain truncation, e.g. seq_a:256,seq_b:128",
    )

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--full_attention_layers", type=int, default=2)
    parser.add_argument(
        "--sliding_window_sizes",
        type=str,
        default="32,16",
        help="Comma-separated window sizes for the sliding layers, e.g. 32,16",
    )
    parser.add_argument("--ffn_mult", type=int, default=4)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--rope_base", type=float, default=10000.0)
    parser.add_argument(
        "--max_behavior_tokens",
        type=int,
        default=0,
        help="0 means auto-sum per-domain seq_max_lens",
    )
    parser.add_argument("--amp", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = os.environ.get("TRAIN_DATA_PATH", args.data_dir)
    args.ckpt_dir = os.environ.get("TRAIN_CKPT_PATH", args.ckpt_dir)
    args.log_dir = os.environ.get("TRAIN_LOG_PATH", args.log_dir)

    project_dir = Path(__file__).resolve().parent
    if args.data_dir is None:
        raise ValueError("data_dir is required (or set TRAIN_DATA_PATH)")
    if args.schema_path is None:
        args.schema_path = str(Path(args.data_dir) / "schema.json")
    if args.ckpt_dir is None:
        args.ckpt_dir = str(project_dir / "checkpoints")
    if args.log_dir is None:
        args.log_dir = str(project_dir / "logs")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    create_logger(os.path.join(args.log_dir, "train.log"))
    set_seed(args.seed)

    seq_max_lens = parse_seq_max_lens(args.seq_max_lens)
    sliding_window_sizes = parse_int_list(args.sliding_window_sizes)

    logging.info("Loading TokenFormer data from %s", args.data_dir)
    train_loader, valid_loader, train_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=args.schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        shuffle_train=True,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
    )

    user_int_specs = build_sparse_feature_specs(
        train_dataset.user_int_schema,
        train_dataset.user_int_vocab_sizes,
    )
    item_int_specs = build_sparse_feature_specs(
        train_dataset.item_int_schema,
        train_dataset.item_int_vocab_sizes,
    )
    user_dense_specs = [(int(fid), int(offset), int(length)) for fid, offset, length in train_dataset.user_dense_schema.entries]

    max_behavior_tokens = int(args.max_behavior_tokens)
    if max_behavior_tokens <= 0:
        max_behavior_tokens = sum(int(v) for v in train_dataset._seq_maxlen.values())

    model = TokenFormer(
        user_int_feature_specs=user_int_specs,
        user_dense_feature_specs=user_dense_specs,
        item_int_feature_specs=item_int_specs,
        seq_domain_vocab_sizes=train_dataset.seq_domain_vocab_sizes,
        max_behavior_tokens=max_behavior_tokens,
        d_model=args.d_model,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        full_attention_layers=args.full_attention_layers,
        sliding_window_sizes=sliding_window_sizes,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout_rate,
        rope_base=args.rope_base,
        action_dim=2,
    ).to(args.device)

    train_config = {
        "data_dir": args.data_dir,
        "schema_path": args.schema_path,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "device": args.device,
        "num_workers": args.num_workers,
        "buffer_batches": args.buffer_batches,
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "eval_every_n_steps": args.eval_every_n_steps,
        "seq_max_lens": seq_max_lens,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "full_attention_layers": args.full_attention_layers,
        "sliding_window_sizes": sliding_window_sizes,
        "ffn_mult": args.ffn_mult,
        "dropout_rate": args.dropout_rate,
        "rope_base": args.rope_base,
        "max_behavior_tokens": max_behavior_tokens,
        "seq_domains": train_dataset.seq_domains,
        "amp": args.amp,
    }

    logging.info("TokenFormer config:\n%s", json.dumps(train_config, indent=2))

    trainer = TokenFormerTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        patience=args.patience,
        device=args.device,
        save_dir=args.ckpt_dir,
        schema_path=args.schema_path,
        train_config=train_config,
        eval_every_n_steps=args.eval_every_n_steps,
        use_amp=args.amp,
    )
    metrics = trainer.fit()
    logging.info("Training finished: %s", metrics)


if __name__ == "__main__":
    main()
