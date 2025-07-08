### Download the TinyStories dataset

``` sh
cd Natural_Language_Processing
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
cd ..
```

### Create conda environment
``` sh
conda env create -f p1_pre_transformer/environment.yml
conda activate llms
```

### Run BPE tokenizer test
``` sh
cd p1_pre_transformer/tokenizers/
python bpe_tokenizer.py --num-processes 1
```
Be careful with `--num-processes` and `--input-path` variables.
