#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Pass extra train.py flags directly via "$@".
# Example:
#   bash tokenformer_project/run.sh --amp
#   bash tokenformer_project/run.sh --sliding_window_sizes 64,32

python3 -u "${SCRIPT_DIR}/train.py" \
    --batch_size 128 \
    --lr 1e-3 \
    --weight_decay 1e-2 \
    --num_workers 8 \
    --d_model 128 \
    --emb_dim 64 \
    --num_heads 4 \
    --num_layers 4 \
    --full_attention_layers 2 \
    --sliding_window_sizes 32,16 \
    --seq_max_lens seq_a:256,seq_b:256,seq_c:512,seq_d:512 \
    "$@"
