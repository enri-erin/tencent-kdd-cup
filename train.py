"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import (
    FeatureSchema,
    get_pcvr_data,
    NUM_TIME_BUCKETS,
    parse_utc_offset_to_seconds,
)
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')
    parser.add_argument('--temporal_utc_offset', type=str, default='UTC+8',
                        help='UTC offset used by absolute temporal sequence features. '
                             'Examples: UTC+8, UTC+0, UTC-5, UTC+05:30, +08:00')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--token_mixer_type', type=str, default='rankmixer',
                        choices=['rankmixer', 'unimixing', 'unimixing_lite'],
                        help='Query/NS token mixer used inside each MultiSeqHyFormerBlock')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')
    parser.add_argument('--no_aligned_user_dense_tokens',
                        dest='use_aligned_user_dense_tokens',
                        action='store_false',
                        default=True,
                        help='Disable aligned user int+dense tokenization and fall back '
                             'to the baseline single user-dense token')
    parser.add_argument('--user_dense_tokens', type=int, default=1,
                        help='Number of NS tokens produced by the aligned user dense tokenizer')
    parser.add_argument('--no_token_type_embeddings',
                        dest='use_token_type_embeddings',
                        action='store_false',
                        default=True,
                        help='Disable token type embeddings for user/item/dense/sequence sources')
    parser.add_argument('--no_domain_gating',
                        dest='use_domain_gating',
                        action='store_false',
                        default=True,
                        help='Disable final lightweight domain gating over per-domain query tokens')
    parser.add_argument('--use_representation_enhancements', action='store_true', default=False,
                        help='Enable the next-step representation enhancement bundle: '
                             'field-aware tokenization and field gating')
    parser.add_argument('--no_coarse_features',
                        dest='use_coarse_features',
                        action='store_false',
                        default=True,
                        help='Disable the lightweight coarse-ranking feature branch '
                             '(sequence statistics and recency summaries)')
    parser.add_argument('--use_activity_rhythm_features',
                        action='store_true',
                        default=False,
                        help='Append user history cadence/activity features to the '
                             'coarse branch. This changes coarse_feat_dim but adds no tokens.')
    parser.add_argument('--no_target_seq_features',
                        dest='use_target_seq_features',
                        action='store_false',
                        default=True,
                        help='Disable target-item vs behavior-sequence relation features')
    parser.add_argument('--no_temporal_sequence_features',
                        dest='use_temporal_sequence_features',
                        action='store_false',
                        default=True,
                        help='Disable the full temporal sequence feature bundle '
                             '(both discrete temporal ids and continuous temporal dense features), '
                             'keeping only the baseline time_bucket path')
    parser.add_argument('--no_temporal_id_attention_gate',
                        dest='use_temporal_id_attention_gate',
                        action='store_false',
                        default=True,
                        help='Disable the attention-style gate over temporal-id embeddings '
                             'and fall back to simple summation')
    parser.add_argument('--use_context_temporal_features',
                        action='store_true',
                        default=False,
                        help='Add one NS token built from the current sample timestamp '
                             '(hour / weekday / rest-day / holiday ids). '
                             'This changes the token count, so d_model must still be '
                             'divisible by num_queries*num_sequences+num_ns in full RankMixer mode.')
    parser.add_argument('--use_user_context_temporal_gate',
                        action='store_true',
                        default=False,
                        help='Inject current-sample timestamp features into user-side '
                             'NS tokens with a residual gate. This does not add NS tokens.')
    parser.add_argument('--use_din_query_pooling',
                        action='store_true',
                        default=False,
                        help='Replace query-generator sequence mean pooling with '
                             'item-aware DIN attention pooling. This does not add tokens.')
    parser.add_argument('--use_unimixing', action='store_true', default=False,
                        help='Backward-compatible alias for --token_mixer_type unimixing')
    parser.add_argument('--use_unimixing_lite', action='store_true', default=False,
                        help='Backward-compatible alias for --token_mixer_type unimixing_lite')
    parser.add_argument('--unimixing_num_basis', type=int, default=4,
                        help='Number of basis local mixing matrices used by UniMixing-Lite')
    parser.add_argument('--unimixing_global_rank', type=int, default=4,
                        help='Low-rank factorization rank used by UniMixing-Lite global mixing')
    parser.add_argument('--unimixing_block_size', type=int, default=0,
                        help='Flattened UniMixing block size B from the paper '
                             '(0 = auto-pick a divisor of T*d_model, preferring 6/8/4/...)')
    parser.add_argument('--unimixing_temperature', type=float, default=1.0,
                        help='Initial temperature used before Sinkhorn normalization')
    parser.add_argument('--unimixing_temperature_end', type=float, default=1.0,
                        help='Final UniMixing temperature after annealing (same as start = fixed)')
    parser.add_argument('--unimixing_temperature_anneal_steps', type=int, default=0,
                        help='Linear annealing steps from unimixing_temperature to '
                             'unimixing_temperature_end (0 = disabled)')
    parser.add_argument('--unimixing_sinkhorn_iters', type=int, default=3,
                        help='Number of Sinkhorn iterations for UniMixing constraints')
    parser.add_argument('--use_unimixing_siamese_norm', action='store_true', default=False,
                        help='Enable the paper-style SiameseNorm path across stacked UniMixing blocks')
    parser.add_argument('--use_iat_sequence_branch', action='store_true', default=False,
                        help='Enable the two-stage IAT branch: compress the current '
                             'instance into InsEmb and replay per-user historical '
                             'InsEmb for downstream sequence modeling')
    parser.add_argument('--iat_max_tokens', type=int, default=192,
                        help='Maximum number of historical InsEmb tokens fetched per user')
    parser.add_argument('--iat_num_layers', type=int, default=1,
                        help='Number of downstream transformer layers used by the IAT branch')
    parser.add_argument('--iat_ins_emb_dim', type=int, default=64,
                        help='Compressed InsEmb dimension used by the IAT two-stage branch')
    parser.add_argument('--iat_source_loss_weight', type=float, default=1.0,
                        help='Auxiliary loss weight for the IAT source-model branch')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable mixed precision training on CUDA')

    args = parser.parse_args()

    if args.use_unimixing and args.use_unimixing_lite:
        parser.error('--use_unimixing and --use_unimixing_lite cannot be enabled together')
    if args.use_unimixing:
        if args.token_mixer_type != 'rankmixer':
            parser.error('--use_unimixing conflicts with an explicit --token_mixer_type')
        args.token_mixer_type = 'unimixing'
    if args.use_unimixing_lite:
        if args.token_mixer_type != 'rankmixer':
            parser.error('--use_unimixing_lite conflicts with an explicit --token_mixer_type')
        args.token_mixer_type = 'unimixing_lite'
    args.use_unimixing = args.token_mixer_type == 'unimixing'
    args.use_unimixing_lite = args.token_mixer_type == 'unimixing_lite'

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')
    parse_utc_offset_to_seconds(args.temporal_utc_offset)

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    effective_shuffle_train = True
    effective_num_workers = args.num_workers
    effective_buffer_batches = args.buffer_batches
    if args.use_iat_sequence_branch:
        effective_shuffle_train = False
        effective_num_workers = 1 if args.num_workers > 0 else 0
        effective_buffer_batches = 0
        logging.info(
            "IAT enabled: forcing train stream to ordered mode "
            f"(shuffle_train=False, num_workers={effective_num_workers}, buffer_batches=0)"
        )
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=effective_num_workers,
        buffer_batches=effective_buffer_batches,
        shuffle_train=effective_shuffle_train,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        use_coarse_features=args.use_coarse_features,
        use_activity_rhythm_features=args.use_activity_rhythm_features,
        use_target_seq_features=args.use_target_seq_features,
        use_temporal_sequence_features=args.use_temporal_sequence_features,
        use_context_temporal_features=(
            args.use_context_temporal_features
            or args.use_user_context_temporal_gate
        ),
        temporal_utc_offset=args.temporal_utc_offset,
        use_iat_sequence_branch=args.use_iat_sequence_branch,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_int_feature_ids": pcvr_dataset.user_int_schema.feature_ids,
        "item_int_feature_ids": pcvr_dataset.item_int_schema.feature_ids,
        "user_dense_feature_specs": pcvr_dataset.user_dense_schema.entries,
        "item_dense_feature_specs": pcvr_dataset.item_dense_schema.entries,
        "coarse_feat_dim": pcvr_dataset.coarse_feat_dim,
        "coarse_seq_per_domain_dim": pcvr_dataset.coarse_seq_per_domain_dim,
        "coarse_global_seq_dim": pcvr_dataset.coarse_global_seq_dim,
        "coarse_global_static_dim": pcvr_dataset.coarse_global_static_dim,
        "target_seq_feat_dim": pcvr_dataset.target_seq_feat_dim,
        "target_seq_sideinfo_counts": pcvr_dataset.target_seq_sideinfo_counts,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "token_mixer_type": args.token_mixer_type,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "use_aligned_user_dense_tokens": args.use_aligned_user_dense_tokens,
        "user_dense_tokens": args.user_dense_tokens,
        "use_token_type_embeddings": args.use_token_type_embeddings,
        "use_domain_gating": args.use_domain_gating,
        "use_representation_enhancements": args.use_representation_enhancements,
        "use_coarse_features": args.use_coarse_features,
        "use_activity_rhythm_features": args.use_activity_rhythm_features,
        "use_target_seq_features": args.use_target_seq_features,
        "use_temporal_sequence_features": args.use_temporal_sequence_features,
        "use_temporal_id_attention_gate": args.use_temporal_id_attention_gate,
        "use_context_temporal_features": args.use_context_temporal_features,
        "use_user_context_temporal_gate": args.use_user_context_temporal_gate,
        "use_din_query_pooling": args.use_din_query_pooling,
        "use_unimixing_lite": args.token_mixer_type == 'unimixing_lite',
        "unimixing_num_basis": args.unimixing_num_basis,
        "unimixing_global_rank": args.unimixing_global_rank,
        "unimixing_block_size": args.unimixing_block_size,
        "unimixing_temperature": args.unimixing_temperature,
        "unimixing_sinkhorn_iters": args.unimixing_sinkhorn_iters,
        "use_unimixing_siamese_norm": args.use_unimixing_siamese_norm,
        "use_iat_sequence_branch": args.use_iat_sequence_branch,
        "iat_max_tokens": args.iat_max_tokens,
        "iat_num_layers": args.iat_num_layers,
        "iat_ins_emb_dim": args.iat_ins_emb_dim,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    if args.token_mixer_type == 'rankmixer':
        mixer_name = f'rankmixer:{args.rank_mixer_mode}'
    else:
        mixer_name = (
            f"{args.token_mixer_type}:B={args.unimixing_block_size or 'auto'}"
            f":siamese={int(args.use_unimixing_siamese_norm)}"
        )
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, mixer={mixer_name}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        use_amp=args.amp,
        iat_source_loss_weight=args.iat_source_loss_weight,
        unimixing_temperature_end=args.unimixing_temperature_end,
        unimixing_temperature_anneal_steps=args.unimixing_temperature_anneal_steps,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
