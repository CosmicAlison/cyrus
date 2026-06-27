#!/usr/bin/env bash

# Fine-tunes all four Surya downstream tasks sequentially.
# Run this ONCE on the AMD MI300X before deploying the inference service.
#
# Prerequisites:
#   - /surya repo cloned and uv sync complete (handled by Dockerfile.rocm)
#   - ROCm visible: rocm-smi should show MI300X
#   - HuggingFace token set: export HF_TOKEN=hf_...
#   - At least 200GB free disk for datasets
#
# Usage (on AMD cloud instance):
#   cd ~/cyrus/surya_service
#   export HF_TOKEN=hf_...
#   bash finetune/run_all.sh 2>&1 | tee finetune/run_all.log
#
# Output: one .pth checkpoint per task, saved to each task's assets/ dir:
#   /surya/downstream_examples/solar_flare_forcasting/assets/solar_flare_weights.pth
#   /surya/downstream_examples/ar_segmentation/assets/ar_segmentation_weights.pth
#   /surya/downstream_examples/euv_spectra_prediction/assets/euv_spectra_weights.pth
#   /surya/downstream_examples/solar_wind_forcasting/assets/solar_wind_weights.pth
#
# NOTE: The pretrained weights are already fine-tuned by NASA-IBM and
# available on HuggingFace (download_data.sh fetches them automatically).
# Only run this script if you want to re-fine-tune on a different data split
# or solar cycle period. For the hackathon, just use download_data.sh.

set -euo pipefail

SURYA_REPO="/surya"
PYTHON="${SURYA_REPO}/.venv/bin/python"
TORCHRUN="${SURYA_REPO}/.venv/bin/torchrun"
LOG_DIR="$(dirname "$0")/logs"
mkdir -p "$LOG_DIR"

echo "══════════════════════════════════════════════════════"
echo "  Cyrus — Surya Downstream Fine-Tuning"
echo "  $(date)"
echo "══════════════════════════════════════════════════════"

# Verify ROCm
echo ""
echo "▶ Checking ROCm..."
$PYTHON -c "import torch; print('CUDA/ROCm:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ── Task 1: Solar Flare Forecasting ──────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Task 1/4: Solar Flare Forecasting"
echo "══════════════════════════════════════════════════════"
cd "${SURYA_REPO}/downstream_examples/solar_flare_forcasting"

echo "▶ Downloading flare dataset and pretrained weights..."
bash download_data.sh

echo "▶ Fine-tuning (LoRA on spectformer)..."
$TORCHRUN \
    --nnodes=1 \
    --nproc_per_node=1 \
    --standalone \
    finetune.py \
    2>&1 | tee "${LOG_DIR}/flare_finetune.log"

echo "✓ Flare fine-tuning complete"

# ── Task 2: Active Region Segmentation ───────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Task 2/4: Active Region Segmentation"
echo "══════════════════════════════════════════════════════"
cd "${SURYA_REPO}/downstream_examples/ar_segmentation"

echo "▶ Downloading AR segmentation dataset..."
bash download_data.sh

echo "▶ Creating AR CSV index..."
$PYTHON create_ar_csv.py

echo "▶ Extracting AR masks..."
cd assets/surya-bench-ar-segmentation
mkdir -p data
if [ -f data.tar.gz ]; then
    tar -xvzf data.tar.gz -C data
else
    echo "WARNING: data.tar.gz not found — skipping extraction"
fi
cd "${SURYA_REPO}/downstream_examples/ar_segmentation"

echo "▶ Fine-tuning (LoRA + UNet head on HelioSpectformer2D)..."
$TORCHRUN \
    --nnodes=1 \
    --nproc_per_node=1 \
    --standalone \
    finetune.py \
    2>&1 | tee "${LOG_DIR}/ar_finetune.log"

echo "✓ AR segmentation fine-tuning complete"

# ── Task 3: EUV Spectra Prediction ───────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Task 3/4: EUV Spectra Prediction"
echo "══════════════════════════════════════════════════════"
cd "${SURYA_REPO}/downstream_examples/euv_spectra_prediction"

echo "▶ Downloading EUV dataset..."
bash download_data.sh

echo "▶ Fine-tuning (HelioSpectformer1D regression head)..."
$TORCHRUN \
    --nnodes=1 \
    --nproc_per_node=1 \
    --standalone \
    finetune.py \
    2>&1 | tee "${LOG_DIR}/euv_finetune.log"

echo "✓ EUV spectra fine-tuning complete"

# ── Task 4: Solar Wind Speed Forecasting ──────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Task 4/4: Solar Wind Speed Forecasting"
echo "══════════════════════════════════════════════════════"
cd "${SURYA_REPO}/downstream_examples/solar_wind_forcasting"

echo "▶ Downloading solar wind dataset..."
bash download_data.sh

echo "▶ Fine-tuning (regression head on spectformer)..."
$TORCHRUN \
    --nnodes=1 \
    --nproc_per_node=1 \
    --standalone \
    finetune.py \
    2>&1 | tee "${LOG_DIR}/wind_finetune.log"

echo "✓ Solar wind fine-tuning complete"

# Summary
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Fine-tuning complete. Checkpoint locations:"
echo "  $(find ${SURYA_REPO}/downstream_examples -name '*.pth' 2>/dev/null | head -20)"
echo "══════════════════════════════════════════════════════"