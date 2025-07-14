# backend/indexing/spimi.py

import os
import math
import json
from collections import defaultdict, Counter
from backend.indexing.preprocessor import preprocess

class SPIMIIndexer:
    def __init__(self, output_path="data/Audio/index.json"):
        self.index = defaultdict(list)
        self.doc_norms = {}
        self.output_path = output_path

    def index_documents(self, documents):
        N = len(documents)

        for doc_id, text in documents.items():
            tokens = preprocess(text)
            tf = Counter(tokens)
            norm = 0

            for term, freq in tf.items():
                # IDF usa el DF actual acumulado
                df = len(set(post[0] for post in self.index[term])) if term in self.index else 0
                idf = math.log(N / (1 + df))
                tfidf = freq * idf
                self.index[term].append((str(doc_id), tfidf))
                norm += tfidf ** 2

            self.doc_norms[str(doc_id)] = math.sqrt(norm)

        self._save_index()

    def _save_index(self):
        data = {
            "index": {term: postings for term, postings in self.index.items()},
            "doc_norms": self.doc_norms
        }
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
