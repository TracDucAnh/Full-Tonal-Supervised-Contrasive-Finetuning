# This code provides a Multi-Tasks Supervised Finetuing Pipeline for Tonal Representation with Wav2vec and HuBERT.

Clone repo: 

```bash
https://github.com/TracDucAnh/Full-Tonal-Supervised-Contrasive-Finetuning.git
```

## Access to the dataset.

Download the dataset.zip file:

```bash
https://drive.google.com/file/d/16bOMHkwRKnbWNS6uh-1UoHcS9qJiBwQF/view?usp=sharing
```

Unzip the dataset.zip into the code directory:

```bash
YOUR_WORKING_DIR/
├── .venv/
├── dataset/    <- This is the dataset folder.
│   ├── edge-tts/
│   ├──.../
├── .env        <- Put HF_TOKEN here
├── .gitignore
├── data_exploration.py
├── dataset.py
├── pretrained_distribution.py
├── push_to_hugg.py
├── README.md
├── requirements.txt
├── run.sh
├── sup_contrasive_hubert.py
├── sup_contrastive_wav2vec.py
```

Put private HF_TOKEN into .env:

```bash
HF_TOKEN="hf_abcxyz_bla_bla"
```

## Run the pipeline

```bash
chmod +x run.sh
./run.sh
```

To Change the Training Config, go to:

```bash
sup_contrasive_wav2vec.py
sup_contrasive_hubert.py
```

And change the Finetuning config:

```bash
NUM_EPOCHS = 50
BATCH_SIZE = 256
TEST_BATCH_SIZE = 8  # For testing mode
MAX_BATCHES = None  # Set to None for full dataset, or number for testing (e.g., 8)
NUM_WORKERS = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

# Weighted Loss Configuration
USE_WEIGHTED_LOSS = True  # Set to True to use weighted loss for class imbalance
WEIGHT_CALCULATION_METHOD = 'effective_samples'  # Options: 'inverse_freq', 'effective_samples', 'balanced'. Should use effective_samples because of servere imbalance

# Contrastive Learning
TEMPERATURE = 0.07
PROJECTION_DIM = 128
CONTRASTIVE_WEIGHT = 0.5  # Balance between contrastive and classification loss

# Visualization
TSNE_SAMPLES = 10000  # Number of samples for t-SNE visualization

```

After Finished Finetuning. Commit to the experiment branch.