from pathlib import Path
import argparse, os
from time import perf_counter
import numpy as np
from tokenizers import BPETokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPE tokenizer trainer")
    parser.add_argument(
        "--use-case", 
        type=str, 
        required=True, 
        choices=["save_tokens", "save_tokens_iter", "base_test"],
        help="Use case to train"
    )
    parser.add_argument("--input-path", type = str, default=str(Path.cwd().parents[2] / "ai_projects" / "data" / "TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--tokens-path", type = str, default=str(Path.cwd().parents[2] / "ai_projects" / "data" / "TinyStoriesV2-GPT4-valid.txt")) # valid
    parser.add_argument("--output-path", type = str, default=str(Path.cwd().parents[2] / "ai_projects" / "data"))
    parser.add_argument("--vocab-size", type = int, default = 10_000)
    parser.add_argument("--num-processes", type = int, default = 4) # 24)
    parser.add_argument("--special-tokens", nargs = "+", type = str, default = ["<|endoftext|>"], help = "List of special tokens")
    args = parser.parse_args()

    bpe = BPETokenizer(input_path = args.input_path, vocab_size = args.vocab_size, special_tokens = args.special_tokens)
    bpe.train(num_processes = args.num_processes)

    if args.use_case == "save_tokens": # TODO: make it more readable
        # create folder if not exist
        ext = args.tokens_path.split(".")[-1]
        filename = args.input_path.split("/")[-1].replace(f".{ext}", "")
        out_dir = os.path.join(args.output_path, filename)
        os.makedirs(out_dir, exist_ok=True)
        # save vocab and merge
        bpe.save_files(out_dir)

        # lazily load encoded string
        t = perf_counter()
        with open(args.tokens_path, 'r') as f:
            text = f.read()
        token_ids = bpe.encode(text, num_processes = args.num_processes, lazy_out_path = None) # out_dir
        # save without chunking / streaming
        # Pick dtype (uint16 if vocab <= 65536, else uint32)
        token_ids_np = np.asarray(token_ids, dtype=np.uint16 if bpe.vocab_size <=65536 else np.uint32)   # or np.uint32
        suffix = "train" if "train" in args.tokens_path else "valid"
        np.save(os.path.join(out_dir, f"tokens_{suffix}.npy"), token_ids_np)
        print(f"⏱️ Encoded: time={perf_counter() - t:.2f}s") 
        
        # NOTE: to read lazily later: 
        # count = os.path.getsize(bin_path) / (2 if bpe.vocab_size <= 65535 else 4)
        # arr = np.memmap(bin_path, dtype='<u2', mode='r', shape=(count,))
        
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