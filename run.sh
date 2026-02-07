#!/usr/bin/env bash
set -e

VENV_DIR=".venv"
REQ_FILE="requirements.txt"

echo "===== [0] Setup virtual environment ====="

# 1. Create venv if not exists
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# 2. Activate venv
echo "→ Activating virtual environment..."
source $VENV_DIR/bin/activate

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

echo "===== [1/4] Data exploration ====="
python data_exploration.py

echo "===== [2/4] Finetune Wav2Vec ====="
python sup_contrastive_wav2vec.py

echo "===== [3/4] Finetune HuBERT ====="
python sup_contrasive_hubert.py

echo "===== [4/4] Push models to HuggingFace ====="
python push_to_hugg.py

echo "🎉 All steps completed successfully!"
