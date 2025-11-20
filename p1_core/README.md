# 🧠 LLMs from Scratch: Core
This module implements the **core components** of Large Language Models (LLMs) entirely from scratch — with minimal dependencies.

At maximum, it uses:
- `torch.nn.Module` and `torch.optim.Optimizer` as base classes
- Low-level PyTorch tensor ops and math functions

Evyrything else - from tokenization to attention and optimization — is **custom implemented**.

## ✅ Covered
- Tokenization (custom BPE with parallel processing)
- Optimizers (SGD, Momentum, Adam, AdamW, Adan, Lion, LARS)
- Training loop
- Configurable Transformer architecture including Attention, Rope, LayerNorm, RMSNorm etc.
- Dataset loaders (TinyStories, OpenWebText)

## ⚙️ Setup 
### 1. Download the dataset (CS336 based)
Run from the root of the project (e.g. `llms-from-scratch`):

```sh
cd ..
mkdir -p data
cd data
```

**for TinyStories:**
``` sh
mkdir TinyStoriesV2-GPT4 && cd TinyStoriesV2-GPT4 && mkdir raw_data && cd raw_data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

**for OpenWebText (CS336 sample):**
``` sh
mkdir OpenWebText && cd OpenWebText && mkdir raw_data && cd raw_data

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz && mv owt_train.txt train.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz && mv owt_valid.txt valid.txt
```

Return to the repo root:
```sh
cd ..
cd ../llms-from-scratch/p1-core
```


### 2. Create conda environment
``` sh
conda env create -f environment.yml
conda activate llms
```

## 🚀 Usage

### 1. Run BPE tokenizer test
``` sh
cd p1_core
python run_tokenizer.py --use-case save_tokens --dataset-name OpenWebText --file-tokenize train.txt # TinyStoriesV2-GPT4
python run_tokenizer.py --use-case save_tokens --dataset-name OpenWebText --load-from 20251110_112141 --file-tokenize valid.txt # TinyStoriesV2-GPT4
```
💡 Use `--num-processes` carefully based on your system’s RAM and CPU cores.

### 2. Train LLM
``` sh
cd.. # run from the root project folder 
uv run -m p1_core.train --config p1_core/configs/train_owt.yaml
```
⚠️ Double-check your `train_owt.yaml` for the correct device, context_length, and dataset path.

## Experiments
Coming soon.

## References
- 📘 [CS224N – NLP with Deep Learning (Stanford)](https://web.stanford.edu/class/cs224n/)
- 📘 [CS336 – Large Language Models (Stanford)](https://stanford-cs336.github.io/)
- 📺 [Karpathy – Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
