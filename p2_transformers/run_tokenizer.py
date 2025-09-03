from pathlib import Path
import argparse, os
from time import perf_counter
import numpy as np
from tokenizers import BPETokenizer
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPE tokenizer trainer")
    parser.add_argument(
        "--use-case", 
        type=str, 
        required=True, 
        choices=["save_tokens", "save_tokens_iter", "base_test"],
        help="Use case to train"
    )
    parser.add_argument("--prefix-path", type = str, default=str(Path.cwd().parents[2] / "ai_projects" / "data"))
    parser.add_argument("--dataset-name", type = str, default="TinyStoriesV2-GPT4")
    parser.add_argument("--load-from", type = str, default="") # "20250901_223122"
    parser.add_argument("--file-tokenize", type = str, default="valid.txt") # NOTE: it is assumed that it is in '*dataset_name*/raw_data'
    parser.add_argument("--vocab-size", type = int, default = 10_000)
    parser.add_argument("--num-processes", type = int, default = 24)
    parser.add_argument("--special-tokens", nargs = "+", type = str, default = ["<|endoftext|>"], help = "List of special tokens")
    args = parser.parse_args()
    # python run_tokenizer.py --use-case save_tokens --dataset-name OpenWebText --load-from 20250901_223122 --file-tokenize train.txt

    curr_time = datetime.now().time()
    if args.load_from == "":
        print(f"Tokenization started at {curr_time.strftime('%H:%M:%S')}")
    
        input_path = Path(args.prefix_path) / args.dataset_name / "raw_data" / "train.txt"
        bpe = BPETokenizer(input_path = input_path, vocab_size = args.vocab_size, special_tokens = args.special_tokens)
        bpe.train(num_processes = args.num_processes)

        # save results of training
        out_dir = Path(args.prefix_path) / args.dataset_name / datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(out_dir, exist_ok=True)
        bpe.save_files(out_dir)
    else:
        t = perf_counter()
        print(f"Tokenizer uploading started at {curr_time.strftime('%H:%M:%S')}")
        input_path = Path(args.prefix_path) / args.dataset_name / args.load_from
        vocab_path, merges_path = str(input_path / "vocab.pkl"), str(input_path / "merges.pkl")
        bpe = BPETokenizer(input_path = None, vocab_size = None, special_tokens = args.special_tokens)
        bpe.from_files(vocab_path = vocab_path, merges_path = merges_path)
        print(bpe.cur_vsize, bpe.vocab_size, len(bpe.merges))
        print(f"⏱️ Uploaded: time={perf_counter() - t:.2f}s") 

    if args.use_case == "save_tokens": # TODO: make it more readable
        t = perf_counter()
        print(f"'Save tokens' started at {curr_time.strftime('%H:%M:%S')}")
        tokenize_path = Path(args.prefix_path) / args.dataset_name / "raw_data" / args.file_tokenize
        # encode text
        token_ids = bpe.encode(tokenize_path, num_processes = args.num_processes, lazy_out_path = None)
        # pick dtype (uint16 if vocab <= 65536, else uint32)
        token_ids_np = np.asarray(token_ids, dtype=np.uint16 if bpe.vocab_size <=65536 else np.uint32)        
        # save encoded text
        out_dir = Path(args.prefix_path) / args.dataset_name / "tokenized"
        os.makedirs(out_dir, exist_ok=True)
        out_name = args.file_tokenize.replace(".txt", ".npy")
        np.save(out_dir / out_name, token_ids_np)
        print(f"⏱️ Encoded: time={perf_counter() - t:.2f}s") 
    elif args.use_case == "save_tokens_iter":
        pass
    elif args.use_case == "base_test":
        # simple text input
        test_text = "Sasha Likes Capybaras<|endoftext|> capybaras like sasha<|endoftext|>"

        t = perf_counter()
        ids = bpe.encode(test_text)
        print("Encoded IDs:", ids)
        print(f"⏱️ Encoded: time={perf_counter() - t:.2f}s") 

        t = perf_counter()
        decoded = bpe.decode(ids)
        print("Decoded Text:", decoded)
        print(f"⏱️ Decoded: time={perf_counter() - t:.2f}s") 

        print("\n--- Streaming encoding test ---")
        text_iter = iter([
            "Sasha Likes Capybaras<|endoftext|> capybaras li",
            "ke sasha<|endoftext|>"
        ])
        t = perf_counter()
        iter = list(bpe.encode_iterable(text_iter))
        print("Streamed IDs:")
        for streamed_id in iter:
            print(streamed_id)
        print(f"⏱️ Encoded iter: time={perf_counter() - t:.2f}s")