# LLMs from Scratch: Core
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
Run from the root of the project (e.g. `NATURAL_LANGUAGE_PROCESSING`):

```sh
cd ..
mkdir -p data
cd data
```

**for TinyStories:**
``` sh
mkdir TinyStoriesV2-GPT4 && cd TinyStoriesV2-GPT4

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

**for OpenWebText (CS336 sample):**
``` sh
mkdir OpenWebText && cd OpenWebText

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```

Return to the repo root:
```sh
cd ..
cd ../NATURAL_LANGUAGE_PROCESSING
```


### 2. Create conda environment
``` sh
conda env create -f environment.yml
conda activate llms
```

## 🚀 Usage

### 1. Run BPE tokenizer test
``` sh
cd llm_core
python run_tokenizer.py --use-case save_tokens --dataset-name TinyStoriesV2-GPT4 --file-tokenize train.txt
```
💡 Use `--num-processes` carefully based on your system’s RAM and CPU cores.

### 2. Train LLM
``` sh
python train.py --config config.yaml
```
⚠️ Double-check your `config.yaml` for the correct device, context_length, and dataset path.

## Experiments
Coming soon.

## References
- 📘 [CS224N – NLP with Deep Learning (Stanford)](https://web.stanford.edu/class/cs224n/)
- 📘 [CS336 – Large Language Models (Stanford)](https://stanford-cs336.github.io/)
- 📺 [Karpathy – Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)