#!/usr/bin/env bash
set -e

VENV_DIR=".venv"
REQ_FILE="requirements.txt"

echo "===== [0] Setup virtual environment ====="

# # 1. Create venv if not exists
# if [ ! -d "$VENV_DIR" ]; then
#     echo "→ Creating virtual environment..."
#     python3 -m venv $VENV_DIR
# fi

# # 2. Activate venv
# echo "→ Activating virtual environment..."
# source $VENV_DIR/bin/activate

# 3. Install requirements if requirements.txt exists
if [ -f "$REQ_FILE" ]; then
    echo "→ Installing requirements..."
    pip install --upgrade pip
    pip install -r $REQ_FILE
else
    echo "requirements.txt not found, skipping dependency install"
fi

# 4. Load environment variables
if [ -f ".env" ]; then
    echo "→ Loading .env"
    set -a
    source .env
    set +a
fi

# Keep BLAS/OpenMP thread counts bounded to avoid OpenBLAS crashes in t-SNE.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-8}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-8}"

# Keep Hugging Face cache under project dir and disable xet backend to avoid
# permission issues in shared system cache directories.
export HF_HOME="${HF_HOME:-$(pwd)/.hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HF_HOME/xdg}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$XDG_CACHE_HOME"

# echo "===== [1/5] Data exploration ====="
# python data_exploration.py

# echo "===== [2/5] Observe Pretrained Distribution ====="
# python pretrained_distribution.py

# echo "===== [3/5] Finetune Wav2Vec ====="
# python sup_contrastive_wav2vec.py

# echo "===== [4/5] Finetune HuBERT ====="
# python sup_contrasive_hubert.py

echo "===== [5/5] Push models to HuggingFace ====="
python push_to_hugg.py

echo "All steps completed successfully!"
