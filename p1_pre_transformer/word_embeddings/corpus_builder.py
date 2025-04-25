import os
from pathlib import Path
import re
from lxml import etree as ET
from collections import defaultdict, Counter
import spacy
from time import perf_counter
from datetime import datetime
import requests
from multiprocessing import Pool
from threading import Thread
import pickle
# from pympler import asizeof
import gc
import sqlite3
import torch
import tarfile
import io
import argparse
from fast_counter import CoocCounter


class WikiCorpusBuilder:
    def __init__(self, xml_path, spacy_model = "en_core_web_sm", min_tokens = 11):
        self.xml_path = xml_path
        self.spacy_model = spacy_model  
        self.min_tokens = min_tokens      

    def _init_worker(self):
        global _nlp
        # _nlp = spacy.load(self.spacy_model, disable=["parser", "ner", "tagger"])
        _nlp = spacy.blank("en")

    def _tokenize_entry(self, entry):
        global _nlp
        title, text = entry
        # text = text.replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text).strip()
        doc = _nlp(text)
        # tokens = [t.text.lower() for t in doc if not t.is_space and not t.is_punct]
        tokens = [t.orth_.lower() for t in doc if not t.is_space and not t.is_punct]
        # tokens = [t.text.lower() for t in doc if not t.is_space and (t.is_alpha or t.text in {".", ",", "!", "?", "'", '"'})] # NOTE: to be even closer for Glove
        return title, tokens #" ".join(tokens)

    def _parse_wiki_xml(self, xml_path):
        context = ET.iterparse(xml_path, events=("end",), tag="{*}article", encoding="utf-8", recover=True)
        for _, elem in context:
            title = elem.attrib.get("name")
            for child in elem:
                if child.tag.endswith("content"):
                    content = " ".join(child.itertext()).strip() # NOTE: if I understand right, "strip adds just microseconds"
                    if len(content.split()) >= self.min_tokens:
                        yield title, content
            elem.clear()    

    def tokenize_and_save(self, out_dir, chunk_size_save = 50_000, chunk_size_process = 125, num_workers=8):
        self._log_run(out_dir, chunk_size_save, chunk_size_process, num_workers)
        os.makedirs(out_dir, exist_ok=True)
        buffer = []
        chunk_id = 0

        with Pool(num_workers, initializer=self._init_worker) as pool:
            gen = self._parse_wiki_xml(self.xml_path)
            chunk_start = perf_counter()

            for title_tokens in pool.imap_unordered(self._tokenize_entry, gen, chunksize=chunk_size_process):
                buffer.append(title_tokens)
                if len(buffer) >= chunk_size_save:
                    prep_done = perf_counter()
                    self._save_chunk(buffer, out_dir, chunk_id)
                    save_done = perf_counter()
                    with open("log.txt", "a") as f:
                        print(f"⏱️ Chunk {chunk_id}: prep={prep_done - chunk_start:.2f}s, save={save_done - prep_done:.2f}s, total={save_done - chunk_start:.2f}s", file = f)
                    buffer = []
                    chunk_id += 1
                    chunk_start = perf_counter()

        if buffer:
            prep_done = perf_counter()
            self._save_async(buffer, out_dir, chunk_id)
            save_done = perf_counter()
            with open("log.txt", "a") as f:
                print(f"⏱️ Chunk {chunk_id} (final): prep={prep_done - chunk_start:.2f}s, save={save_done - prep_done:.2f}s, total={save_done - chunk_start:.2f}s", file = f)


    def _save_chunk(self, buffer, out_dir, chunk_id):
        out_path = os.path.join(out_dir, f"chunk_{chunk_id}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open("log.txt", "a") as f:
            print(f"✅ Saved {" " * (2 - len(str(chunk_id)))}{out_path} with {len(buffer)} articles", file = f)

    def _save_async(self, buffer, out_dir, chunk_id):
        Thread(target=self._save_chunk, args=(buffer, out_dir, chunk_id)).start()
    
    def _log_run(self, out_dir, chunk_size_save, chunk_size_process, num_workers):
        try:
            res = requests.get("https://ipinfo.io", timeout=3)
            data = res.json()

            city = data.get("city", "Unknown")
            region = data.get("region", "")
            country = data.get("country", "")
            location = f"{city}, {region}, {country}".strip(", ")
            timezone = data.get("timezone", "")
        except Exception as e:
            location = "-" * 10
            timezone = "-" * 10

        with open("log.txt", "a") as f:
            print("\n\n" + "="*80, file = f)
            print(f"🔁  GloVe Tokenization Run - {datetime.now().strftime('%Y-%m-%d %Y-%m-%d | %H:%M:%S')}", file = f)
            print(f"🕰️  Timezone           : {timezone}", file=f)
            print(f"🌍  Location           : {location}", file=f)
            print(f"🗃️  XML input          : {self.xml_path}", file = f)
            print(f"🧠  CPU cores          : {os.cpu_count()}", file = f)
            print(f"⚙️  Workers            : {num_workers}", file = f)
            print(f"📦  Chunk size save    : {chunk_size_save}", file = f)
            print(f"🚛  Chunk size process : {chunk_size_process}", file = f)
            print(f"📎  SpaCy tokenizer    : {self.spacy_model}", file = f)
            print(f"📂  Output folder      : {out_dir}", file = f)
            print("="*80, file = f)

    def build_vocab_and_freqs(self, tokens_chunk_dir, max_vocab_size = 400_000, min_count=5):
        time_start = perf_counter()
        vocab_counter = Counter()
        with open("log.txt", "a") as f:
            print(f"\n\n⏳ [build_vocab] Counting token frequencies for {max_vocab_size} tokens - {datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}", file = f)
            print("="*80, file = f)

        # Step 1: Get corpus and calculate frequencies
        for file_name in os.listdir(tokens_chunk_dir):
            if not (file_name.endswith(".pkl") and file_name.startswith("chunk_")):
                continue

            with open(os.path.join(tokens_chunk_dir, file_name), "rb") as f:
                data = pickle.load(f)

            for _, tokens in data:
                vocab_counter.update(tokens)
        count_end = perf_counter()
        with open("log.txt", "a") as f:        
            print(f"✅ Token counting done in {count_end - time_start:.2f}s", file = f)

        # Step 2: Filter it and sort by frequencies
        filtered = [(word, count) for word, count in vocab_counter.items() if count >= min_count]
        sorted_vocab = sorted(filtered, key = lambda x: x[1], reverse = True)[:max_vocab_size]

        # Step 3: Build mappings
        vocab = {token: idx for idx, (token, _) in enumerate(sorted_vocab)}
        id_to_token = [token for token, _ in sorted_vocab]
        freqs = {idx: count for idx, (_, count) in enumerate(sorted_vocab)}
        filter_end = perf_counter()
        with open("log.txt", "a") as f:
            print(f"✅ Filtering + sorting done in {filter_end - count_end:.2f}s", file = f)

        # Step 4: Save to pickle
        to_save = {
            "vocab": vocab,
            "id_to_token": id_to_token,
            "freqs": freqs
        }
        out_path = os.path.join(tokens_chunk_dir, "vocab_details.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        save_end = perf_counter()
        total_time = perf_counter() - time_start
        with open("log.txt", "a") as f:
            print(f"💾 Saved vocab to {out_path} in {save_end - filter_end:.2f}s", file = f)
            print(f"🏁 Total time: {total_time:.2f}s for {len(vocab)} tokens\n", file = f)

    def load_vocab_and_freqs(self, tokens_chunk_dir):
        vocab_path = os.path.join(tokens_chunk_dir, "vocab_details.pkl")

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"❌ Vocab file not found: {vocab_path}")
        
        with open(vocab_path, "rb") as f:
            data = pickle.load(f)

        vocab = data["vocab"]
        id_to_token = data["id_to_token"]
        freqs = data["freqs"]
        return vocab, id_to_token, freqs

    def build_co_oc_matrix_sqlite(self, files_path, window_size = 5, window_type = "symmetric", importance = True, batch_size = 10_000_000, group_by_size = 1_000_000_000):
        assert window_type in {"symmetric", "left", "right"}, f"❌ Invalid window_type: {window_type}"
        # TODO: Add info about vocab size
        with open("log.txt", "a") as f:
            print(f"\n\n[{datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}] 🧱 Building co-occurrence matrix with window={window_size} | Batching to SQLite {batch_size} rows | Grouping every {group_by_size} rows", file = f)

        # Step 1: Load Vocab and Frequencies
        vocab, _, _ = self.load_vocab_and_freqs(files_path)

        # Step 2: Initialize SQLite
        db_path = db_path = os.path.join(files_path, "cooc_matrix.db")
        conn, cursor = self._init_sqlite(db_path)

        # Step 3: Iterate over chunks
        co_oc_start = perf_counter()
        total_articles = 0
        total_rows = 0
        grouped_by = 1
        X =  CoocCounter()
        file_list = [f for f in os.listdir(files_path) if f.endswith(".pkl") and f.startswith("chunk_")] # each chunk contains 10K articles
        for file_idx, file_name in enumerate(sorted(file_list)):
            chunk_start = perf_counter()
            with open(os.path.join(files_path, file_name), "rb") as f:
                data = pickle.load(f)

            for _, tokens in data:
                token_ids = [vocab[t] for t in tokens if t in vocab] # NOTE: we are actually moving distances but not significantly
                for center_idx, center_id in enumerate(token_ids):
                    for distance in range(1, window_size + 1):
                        weight = 1.0 / distance if importance else 1.0
                        if window_type != "right" and center_idx - distance >= 0:
                            context_id = token_ids[center_idx - distance]
                            # X[(center_id, context_id)] += weight
                            X.update(center_id, context_id, weight)
                        if window_type != "left" and center_idx + distance < len(token_ids):
                            context_id = token_ids[center_idx + distance]
                            # X[(center_id, context_id)] += weight
                            X.update(center_id, context_id, weight)
                total_articles += 1

                if len(X) >= batch_size:
                    total_rows += len(X)
                    rows = [(i_id, j_id, val) for (i_id, j_id), val in X.items()]
                    t_batch = perf_counter()
                    self._write_to_sqlite(rows, batch_size, conn, cursor)
                    t_batch = perf_counter() - t_batch
                    X = CoocCounter() #  defaultdict(float) #
                    del rows
                    gc.collect()

                if total_rows >= grouped_by * group_by_size:
                    t_aggr = perf_counter() 
                    self._aggregate_and_replace_sqlite(cursor, conn)
                    t_aggr = perf_counter() - t_aggr
                    with open("log.txt", "a") as f: 
                        print(f"[{datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}] ✅✅✅ Grouped at {total_rows:,} rows — Time: {t_aggr:.2f}s", file=f)
                    grouped_by += 1

            with open("log.txt", "a") as f:
                sp1 = " " * (2 - len(str(file_idx + 1)))
                sp2 = " " * (12 - len(str(total_rows)))
                print(f"[{datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}] ✅ Processed {sp1}{file_idx+1}/{len(file_list)} chunks | total rows {sp2}{total_rows}; it took {perf_counter() - chunk_start:.2f}s", file = f)

        # Final flush
        if X:
            total_rows += len(X)
            rows = [(i_id, j_id, val) for (i_id, j_id), val in X.items()]
            self._write_to_sqlite(rows, batch_size, conn, cursor)
            del X, rows
            gc.collect()

        # Final aggregation
        t_aggr = perf_counter() 
        self._aggregate_and_replace_sqlite(cursor, conn)
        t_aggr = perf_counter() - t_aggr
        with open("log.txt", "a") as f:
            print(f"[{datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}] ✅✅✅ Final Grouping: {t_aggr:.2f}s", file=f)

        conn.close()
        co_oc_end = perf_counter()
        with open("log.txt", "a") as f:    
            co_oc_end = perf_counter()
            print(f"[{datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}] ✨ Done. Total time: {co_oc_end - co_oc_start:.2f}s", file=f)

    def _init_sqlite(self, db_path):
        # Step 1: Init sqlite
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # ⚡ Speed up SQLite performance for bulk inserts
        cursor.execute("PRAGMA journal_mode = OFF;")
        cursor.execute("PRAGMA synchronous = OFF;")
        cursor.execute("PRAGMA temp_store = 0;")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cooc (
                word_i INTEGER,
                word_j INTEGER,
                value  REAL
            )
        """)
        conn.commit()
        return conn, cursor
    
    def _write_to_sqlite(self, rows, batch_size, conn, cursor):
        cursor.execute("BEGIN TRANSACTION;")
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            cursor.executemany("""
                INSERT INTO cooc (word_i, word_j, value)
                VALUES (?, ?, ?)
            """, batch)
        conn.commit()

    def _aggregate_and_replace_sqlite(self, cursor, conn):
        # Step 1: Create temp table with aggregated values
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cooc_tmp AS
            SELECT word_i, word_j, SUM(value) AS value
            FROM cooc
            GROUP BY word_i, word_j
        """)
        conn.commit()

        # Step 2: Drop the old table
        cursor.execute("DROP TABLE cooc;")
        conn.commit()

        # Step 3: Rename new table
        cursor.execute("ALTER TABLE cooc_tmp RENAME TO cooc;")
        conn.commit()

    def _sqlite_to_chunks(self, files_path, db_file_name = "cooc_matrix.db", prefix = "cooc", chunk_size = 50_000_000, save_type = '.pt', sample_per_record = True):
        """
        Streams co-occurrence data from a SQLite database and saves it in chunks.

        Depending on `save_type`, data is saved either as:
        - `.pt`: two PyTorch tensors (`indices_pairs` of shape (N, 2), and `cooccurrence` of shape (N,))
        - `webdata`: WebDataset-compatible tar shards (either one tensor or per-sample records)

        Args:
            files_path (str): Base directory containing the SQLite DB and output folders.
            db_file_name (str): Name of the SQLite file inside `files_path`.
            prefix (str): Prefix used for naming output files (e.g. 'cooc').
            chunk_size (int): Number of rows to fetch and store per chunk.
            save_type (str): Either '.pt' or 'webdata', to control output format.
        """
        t_start = perf_counter()
        with open("log.txt", "a") as f:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}] 🚀 Streaming SQLite to '{save_type}' format with chunk_size = {chunk_size}", file=f)
        
        # Assert
        valid_types = {".pt", "webdata"}
        if save_type not in valid_types:
            with open("log.txt", "a") as f:
                error_message = f"❌ Invalid save_type: '{save_type}' (expected one of {valid_types})"
                print(f"[{datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}] {error_message}", file=f)
            raise ValueError(error_message)


        # Init connection
        db_path = os.path.join(files_path, db_file_name)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Extract number of rows
        cur.execute("SELECT COUNT(*) FROM cooc")
        total_rows = cur.fetchone()[0]
        with open("log.txt", "a") as f:
            print(f"[{datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}] 📥 Will stream {total_rows:,} rows", file=f)

        # Extract data from SQLite by batches
        seen = 0
        cur.execute("SELECT word_i, word_j, value FROM cooc")
        n_chunks = total_rows // chunk_size + int(total_rows % chunk_size > 0)
        n_log = round(n_chunks / 20)
        for chunk_id, _ in enumerate(range(0, total_rows, chunk_size)):
            rows = cur.fetchmany(chunk_size)

            if save_type == ".pt":
                # Save tensors
                self._write_pt(files_path, prefix, rows, chunk_id)
            elif save_type == "webdata":
                # Save shards in webdata
                self._write_shard(files_path, prefix, rows, chunk_id, sample_per_record = sample_per_record)

            seen += len(rows)

            # cleanup
            del rows
            gc.collect()

            # log just 20 chunk info
            if chunk_id % n_log == 0:
                with open("log.txt", "a") as f:
                    print(f"[{datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}] 💾 Chunk {chunk_id:04d}: saved {" " * (11 - len(str(seen)))}{seen:,}/{total_rows:,} rows", file=f)

        conn.close()

        with open("log.txt", "a") as f:
            print(f"[{datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}] ✅ Finished streaming in {perf_counter() - t_start:.2f}s", file=f)

    def _write_pt(self, files_path, prefix, rows, chunk_id):
        """
        Saves the co-occurrence data as PyTorch .pt tensors.

        Args:
            files_path (str): Base directory where tensors should be saved.
            prefix (str): Prefix for the saved file names (e.g. 'cooc').
            indices_pairs (Tensor): Tensor of shape (N, 2) with (center_id, context_id) pairs.
            cooccurrence (Tensor): Tensor of shape (N,) with co-occurrence values.
            chunk_id (int): ID of the current chunk, used for file naming.
        """
        center_ids, context_ids, values = zip(*rows)
        indices_pairs = torch.tensor([center_ids, context_ids], dtype=torch.long).T
        cooccurrence = torch.tensor(values, dtype=torch.float)

        save_dir = os.path.join(files_path, "torch_tensors")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(indices_pairs, os.path.join(save_dir, f"{prefix}_indices_pairs_{chunk_id:04d}.pt"))
        torch.save(cooccurrence, os.path.join(save_dir, f"{prefix}_values_{chunk_id:04d}.pt"))

        # cleanup , 
        del center_ids, context_ids, values, indices_pairs, cooccurrence


    
    def _write_shard(self, files_path, prefix, rows, shard_id, sample_per_record = True):
        """
        Writes either per-sample or full-tensor to a WebDataset-compatible tar.
        Each record contains:
            - __key__
            - indices.pt (long)
            - values.pt  (float32)
        """
        tar_path = os.path.join(files_path, "webdataset", f"{prefix}_{shard_id:04d}.tar")
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        key_base = f"cooc_{shard_id:05d}"
        
        with tarfile.open(tar_path, "w") as tar:
            if sample_per_record:
                self._write_triplet_to_shard(tar, key_base, rows)
            else:
                self._write_pt_to_shard(tar, key_base, rows)

    def _write_pt_to_shard(self, tar, key, rows):
        """
        Creates and writes a WebDataset-compatible sample to the provided tar archive.
        Format:
        - __key__
        - indices_pairs.pt
        - cooccurrence.pt
        """
        center_ids, context_ids, values = zip(*rows)
        indices_pairs = torch.tensor([center_ids, context_ids], dtype=torch.long).T
        cooccurrence = torch.tensor(values, dtype=torch.float)
        sample = {
            "__key__": key,
            "indices_pairs.pt": indices_pairs,
            "cooccurrence.pt": cooccurrence
        }
        
        for name, data in sample.items():
            if name == "__key__":
                content = key.encode("utf-8")
            else:
                buffer = io.BytesIO()
                torch.save(data, buffer, _use_new_zipfile_serialization=False)
                content = buffer.getvalue()           
            tarinfo = tarfile.TarInfo(name=f"{key}/{name}")
            tarinfo.size = len(content)
            tar.addfile(tarinfo, fileobj=io.BytesIO(content))
        
        # cleanup , 
        del center_ids, context_ids, values, indices_pairs, cooccurrence
    
    def _write_triplet_to_shard(self, tar, key, rows):
        """
        Creates and writes a WebDataset-compatible row of triples to the provided tar archive.
        Format:
        - file_1.txt
        - file_2.txt
        - file_3.txt
        ...

        NOTE: it is useless in this setup, WebData limit from bottom all files by 512 bytes + 512 bytes for header.
        So, each tar of 50M pairs will be 50M x 1024B = 51.2GB
        """
        for i, (center_id, context_id, cooc) in enumerate(rows):
            filename = f"{key}_{i:06d}.txt"
            content = f"{center_id} {context_id} {cooc}\n".encode("utf-8")
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(content)
            tar.addfile(tarinfo, io.BytesIO(content))



class GloveCorpusBuilder:
    def __init__(self, window_size = 5, x_max = 100, alpha = 0.75, max_vocab_size = 400_000, importance = True, output_path = None):
        self.window_size = window_size
        self.x_max = x_max
        self.alpha = alpha
        self.v_size = max_vocab_size
        self.importance = importance # TODO: possibly replace by weight
        self.output_path = output_path
        # init tokenizer
        self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])
        # create internal variables
        self.articles = None
        self.word_to_id = {}
        self.id_to_word = {} # NOTE: or better list?
        self.word_freq = defaultdict(int)
        self.co_oc_matrix = defaultdict(float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and process Wikipedia corpus")
    
    parser.add_argument(
        "--phase", 
        type=str, 
        required=True, 
        choices=["tokenize", "build_vocab", "cooc", "merge_chunks", "merge_sqlite", "to_torch", "to_webdata"],
        help="Pipeline phase to test"
    )
    parser.add_argument("--xml", type=str, default=str(Path.cwd().parents[1] / "data" / "enwiki-20181001-corpus.xml"))
    parser.add_argument("--out-dir", type=str, default="corpus_tokens_wiki2018")
    parser.add_argument("--chunk-size", type=int, default = 50_000_000)
    parser.add_argument("--vocab-size", type=int, default = 400_000)
    
    args = parser.parse_args()

    builder = WikiCorpusBuilder(args.xml)

    if args.phase == "tokenize":
        builder.tokenize_and_save(args.out_dir, chunk_size_save=100_000, chunk_size_process=250, num_workers=8)
    elif args.phase == "build_vocab":
        builder.build_vocab_and_freqs(tokens_chunk_dir=args.out_dir, max_vocab_size=args.vocab_size)
    elif args.phase == "cooc":
        builder.build_co_oc_matrix_sqlite(args.out_dir, batch_size=20_000_000, group_by_size=2_000_000_000)
    elif args.phase == "to_torch":
        builder._sqlite_to_chunks(args.out_dir, chunk_size=args.chunk_size, save_type=".pt", sample_per_record=False)
    elif args.phase == "to_webdata":
        builder._sqlite_to_chunks(args.out_dir, chunk_size=args.chunk_size, save_type="webdata", sample_per_record=False)
    else:
        raise ValueError(f"Unknown phase: {args.phase}")
    