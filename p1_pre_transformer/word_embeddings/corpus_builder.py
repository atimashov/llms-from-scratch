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
            pickle.dump(buffer, f)
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
            print(f"🔁  GloVe Tokenization Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file = f)
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
            print(f"\n\n⏳ [build_vocab] Counting token frequencies - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file = f)
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
            pickle.dump(to_save, f)
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
    
    def build_co_oc_matrix(self, files_path, window_size = 5, x_max = 100, alpha = 0.75, importance = True):
        pass


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
    # TODO: add options to test
    xml_path = Path.cwd().parents[1] / "data" / "enwiki-20181001-corpus.xml"
    out_dir = 'corpus_tokens_wiki2018'
    builder = WikiCorpusBuilder(xml_path)
    # builder.tokenize_and_save(out_dir, chunk_size_save=100_000, chunk_size_process = 250, num_workers=8)
    builder.build_vocab_and_freqs(tokens_chunk_dir = out_dir)