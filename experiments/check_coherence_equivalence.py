"""
Check that the memory-fixed calculate_coherence_scores gives the same scores
as the original. Run on the server:  python check_coherence_equivalence.py
Uses 300 bbc-news documents (small, fast). Prints both results and whether
they match.
"""
import gc
from collections import Counter

import numpy as np
from datasets import load_dataset
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer

TOKEN_PATTERN = "(?u)\\b[\\w-]+\\b"


def _vectorizer():
    return CountVectorizer(
        stop_words="english", ngram_range=(2, 3), token_pattern=TOKEN_PATTERN
    )


def old_scores(topics, docs, topn=10):
    v = _vectorizer()
    v.fit(docs)
    analyzer = v.build_analyzer()
    tokens = [analyzer(doc) for doc in docs]
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(t) for t in tokens]
    texts = [[dictionary[i] for i, _ in doc] for doc in corpus]
    out = {}
    for m in ["c_v", "u_mass", "c_uci", "c_npmi"]:
        cm = CoherenceModel(topics=topics, texts=texts, corpus=corpus,
                            dictionary=dictionary, coherence=m, topn=topn)
        out[m] = float(np.mean(cm.get_coherence_per_topic()))
    return out


def new_scores(topics, docs, topn=10):
    analyzer = _vectorizer().build_analyzer()
    dictionary = Dictionary(analyzer(doc) for doc in docs)
    corpus = [dictionary.doc2bow(analyzer(doc)) for doc in docs]
    texts = [[dictionary[i] for i, _ in doc] for doc in corpus]
    gc.collect()
    out = {}
    for m in ["c_v", "u_mass", "c_uci", "c_npmi"]:
        cm = CoherenceModel(topics=topics, texts=texts, corpus=corpus,
                            dictionary=dictionary, coherence=m, topn=topn,
                            processes=1)
        out[m] = float(np.mean(cm.get_coherence_per_topic()))
        del cm
        gc.collect()
    return out


if __name__ == "__main__":
    ds = load_dataset("SetFit/bbc-news", split="train")
    docs = [d for d in ds["text"][:300] if d.strip()]

    # Build 5 test topics from the 50 most frequent n-grams.
    analyzer = _vectorizer().build_analyzer()
    counts = Counter(g for d in docs for g in analyzer(d))
    common = [g for g, _ in counts.most_common(50)]
    topics = [common[i::5] for i in range(5)]

    a = old_scores(topics, docs)
    b = new_scores(topics, docs)
    print("old:", a)
    print("new:", b)
    same = all(np.isclose(a[k], b[k], rtol=0, atol=1e-9, equal_nan=True) for k in a)
    print("IDENTICAL" if same else "DIFFERENT - do not use the fix without checking")
