#!/usr/bin/env python
# coding: utf-8
"""
full_dataset_comparison_v1.py
=============================
v2 comparison shell (datasets, baselines, Optuna, vocab chunks) with v1
GraphTopic and v1 scoring inlined. ngram_range=(2, 3) is fixed.
Outputs: hyperparameter_analysis_results/full_comparison_v1/
Smoke: python full_dataset_comparison_v1.py --smoke
       → hyperparameter_analysis_results/full_comparison_v1_smoke/
"""

from __future__ import annotations

# ── Shared local-environment bootstrap (same as the main script) ─────────────
import importlib.util
import hashlib
import inspect
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
import warnings
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CHUNKING CONFIGURATION
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CHUNKING — v1 fixed document sizes (not vocabulary-capped)
# ---------------------------------------------------------------------------
N_EQUAL_CHUNKS = 5
# Exact sizes from v1/GraphTopic/data/prepare.py
FIXED_DOC_CHUNK_SIZE = {
    "SetFit/20_newsgroups": 3768,
    "SetFit/bbc-news": 2225,
    "TrumpTweet": 11314,
}
SEED = 42

# ---------------------------------------------------------------------------
# SMOKE TEST — tiny chunks / short Optuna; production sizes stay intact
# ---------------------------------------------------------------------------
SMOKE_TEST = False  # overridden by --smoke
SMOKE_DOC_CHUNK_SIZE = 80
SMOKE_MAX_CHUNKS = 1
SMOKE_N_TRIALS = 2
SMOKE_N_STARTUP_TRIALS = 1
SMOKE_EARLY_STOPPING_PATIENCE = 2
SMOKE_BASELINE_EPOCHS = 5

# ---------------------------------------------------------------------------
# HYPERPARAMETER OPTIMIZATION CONFIGURATION
# ---------------------------------------------------------------------------
# False preserves the current behaviour: load and use saved optimal parameters.
ENABLE_HYPERPARAM_OPTIMIZATION = True
HYPERPARAM_N_TRIALS = 30
HYPERPARAM_N_STARTUP_TRIALS = 8
HYPERPARAM_EARLY_STOPPING = True
HYPERPARAM_EARLY_STOPPING_PATIENCE = 12
HYPERPARAM_USE_MULTI_OBJECTIVE = True
HYPERPARAM_MULTI_OBJECTIVE_METRICS = ["c_npmi", "intra_similarity_avg"]
HYPERPARAM_MULTI_OBJECTIVE_WEIGHTS = [0.75, 0.25]

# v1 search grid. ngram_range=(2, 3) is fixed inside v1 KeyBERT and v1 coherence.
OPTIMIZE_HYPERPARAMS: Dict[str, bool] = {
    "TopM": True,
    "TopN": True,
    "overlapping_threshold": True,
}
PARAM_SPACE: Dict = {
    "TopM": {"type": "categorical", "choices": [500, 1000, 1500]},
    "TopN": {"type": "categorical", "choices": [10, 15, 20]},
    "overlapping_threshold": {"type": "categorical", "choices": [0.35, 0.50, 0.75]},
}
# ---------------------------------------------------------------------------
# LOCAL PATHS  — identical to the main script so results land in the same tree
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR
LOCAL_OUTPUT_DIR = WORKSPACE_DIR / "hyperparameter_analysis_results"
GRAPH_TOPIC_ZIP_PATH = WORKSPACE_DIR / "GraphTopic.zip"
GRAPH_TOPIC_DIR = WORKSPACE_DIR / "GraphTopic"
TWEETS_CSV_PATH = WORKSPACE_DIR / "tweets_01-08-2021.csv"

TWEETS_FILE_URL = (
    "https://drive.google.com/file/d/17y5_RQWQrtFkf_JMA058rMOA9r3LJM5d/view?usp=sharing"
)
GRAPH_TOPIC_ZIP_URL = "https://drive.google.com/file/d/1MpdzVIJnZxB6L_NSHX1GZGWVrF7EPHft/view?usp=drive_link"

AUTO_INSTALL_DEPENDENCIES = True
AUTO_DOWNLOAD_NOTEBOOK_ASSETS = True

FULL_COMPARISON_DIR = LOCAL_OUTPUT_DIR / "full_comparison_v1"
FULL_COMPARISON_SMOKE_DIR = LOCAL_OUTPUT_DIR / "full_comparison_v1_smoke"
# Switched to FULL_COMPARISON_SMOKE_DIR when --smoke / SMOKE_TEST is on.
ACTIVE_COMPARISON_DIR = FULL_COMPARISON_DIR


# ============================================================================
# ── MODEL STYLE REGISTRY  ────────────────────────────────────────────────────
# ============================================================================
# Each entry: color, marker, linestyle, linewidth, markersize
# Override any entry below to customise a model's appearance.
# Set model_subset in the plot functions to restrict which models appear.

DEFAULT_MODEL_ORDER = [
    "GraphTopic",
    "LDA",
    "NMF",
    "HDP",
    "LSI",
    "ProdLDA",
    "HyperMiner",
    "BERTopic",
    "ECRTM",
    "FASTopic",
]

DEFAULT_BASELINE_MODELS = DEFAULT_MODEL_ORDER[1:]
DEFAULT_GENSIM_MODELS = ["LDA", "NMF", "HDP", "LSI"]
DEFAULT_TOPMOST_MODELS = ["ProdLDA", "HyperMiner", "ECRTM", "FASTopic"]

DEFAULT_MODEL_STYLES: Dict[str, Dict] = {
    "GraphTopic": dict(
        color="#2166AC",
        marker="o",
        linestyle="-",
        linewidth=2.5,
        markersize=9,
        hatch="",
        label="GraphTopic (Ours)",
    ),
    "LDA": dict(
        color="#B2182B",
        marker="s",
        linestyle="--",
        linewidth=1.9,
        markersize=7,
        hatch="//",
        label="LDA",
    ),
    "NMF": dict(
        color="#EF8A62",
        marker="^",
        linestyle="-.",
        linewidth=1.9,
        markersize=7,
        hatch="\\\\",
        label="NMF",
    ),
    "HDP": dict(
        color="#67A9CF",
        marker="v",
        linestyle=":",
        linewidth=1.9,
        markersize=7,
        hatch="||",
        label="HDP",
    ),
    "LSI": dict(
        color="#FDD54F",
        marker="<",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=1.9,
        markersize=7,
        hatch="..",
        label="LSI",
    ),
    "ProdLDA": dict(
        color="#762A83",
        marker=">",
        linestyle=(0, (5, 2)),
        linewidth=2.0,
        markersize=8,
        hatch="xx",
        label="ProdLDA",
    ),
    "HyperMiner": dict(
        color="#5AAE61",
        marker="P",
        linestyle=(0, (1, 1)),
        linewidth=2.0,
        markersize=8,
        hatch="++",
        label="HyperMiner",
    ),
    "BERTopic": dict(
        color="#D73027",
        marker="X",
        linestyle=(0, (7, 2, 1, 2)),
        linewidth=2.0,
        markersize=8,
        hatch="--",
        label="BERTopic",
    ),
    "ECRTM": dict(
        color="#FC8D59",
        marker="h",
        linestyle=":",
        linewidth=2.0,
        markersize=8,
        hatch="||",
        label="ECRTM",
    ),
    "FASTopic": dict(
        color="#1A9850",
        marker="D",
        linestyle="-.",
        linewidth=2.0,
        markersize=8,
        hatch="\\\\",
        label="FASTopic",
    ),
}

# Auto-assign styles to any model name not listed above
_EXTRA_COLORS = ["#4D4D4D", "#878787", "#BABABA", "#40B4C4", "#E8601C"]
_EXTRA_MARKERS = ["<", ">", "p", "h", "*"]
_EXTRA_LINESTYLES = [(0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 5)), "-.", "--"]
_EXTRA_HATCHES = ["", "//", "\\\\", "xx", "||"]


def get_style(model_name: str, registry: Dict[str, Dict] | None = None) -> Dict:
    """Return style dict for *model_name*, auto-generating if unknown."""
    reg = registry or DEFAULT_MODEL_STYLES
    if model_name in reg:
        return dict(reg[model_name])
    # deterministic auto-assignment
    idx = sum(ord(ch) for ch in model_name) % len(_EXTRA_COLORS)
    return dict(
        color=_EXTRA_COLORS[idx],
        marker=_EXTRA_MARKERS[idx % len(_EXTRA_MARKERS)],
        linestyle=_EXTRA_LINESTYLES[idx % len(_EXTRA_LINESTYLES)],
        linewidth=2.0,
        markersize=8,
        hatch=_EXTRA_HATCHES[idx % len(_EXTRA_HATCHES)],
        label=model_name,
    )


def ordered_models(
    available_models,
    model_subset: Optional[List[str]] = None,
) -> List[str]:
    """Return a stable publication/order-friendly model list."""
    if model_subset is not None:
        return list(model_subset)
    available = [str(m) for m in available_models]
    preferred = [m for m in DEFAULT_MODEL_ORDER if m in available]
    extras = sorted(m for m in available if m not in DEFAULT_MODEL_ORDER)
    return preferred + extras


# ============================================================================
# ── LOCAL ENVIRONMENT BOOTSTRAP  ─────────────────────────────────────────────
# ============================================================================


def _run_python_module(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", *args])


def _module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def install_local_dependencies() -> None:
    if not AUTO_INSTALL_DEPENDENCIES:
        return
    module_requirements = {
        "numpy": "numpy",
        "pandas": "pandas",
        "psutil": "psutil",
        "sentence_transformers": "sentence_transformers",
        "datasets": "datasets>=3.0,<4.0",
        "netgraph": "netgraph",
        "bertopic": "bertopic",
        "keybert": "keybert",
        "topmost": "topmost",
        "gensim": "gensim",
        "optuna": "optuna",
        "gdown": "gdown",
        "spacy": "spacy",
    }
    missing = [
        pkg for mod, pkg in module_requirements.items() if not _module_exists(mod)
    ]
    if missing:
        _run_python_module("pip", "install", *missing)
    _run_python_module("pip", "install", "sentence-transformers[onnx]")
    # datasets>=5 needs huggingface_hub>=1.x; transformers 4.x needs hub<1.0.
    # Keep datasets on 3.x so sentence-transformers can import.
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        _run_python_module("pip", "install", "-U", "datasets>=3.0,<4.0")
    if not _module_exists("pke"):
        _run_python_module("pip", "install", "git+https://github.com/boudinfl/pke.git")
    try:
        import spacy

        spacy.load("en_core_web_sm")
    except Exception:
        _run_python_module("spacy", "download", "en_core_web_sm")


def download_drive_asset(url: str, output_path: Path) -> None:
    if output_path.exists() or not AUTO_DOWNLOAD_NOTEBOOK_ASSETS:
        return
    import re
    import gdown

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # gdown>=5 removed ``fuzzy``; extract the Drive file id instead.
    match = re.search(r"/file/d/([^/]+)", url)
    if match:
        gdown.download(id=match.group(1), output=str(output_path), quiet=False)
    else:
        gdown.download(url, str(output_path), quiet=False)


def extract_zip_if_needed(zip_path: Path, extract_to: Path) -> None:
    if extract_to.exists() or not zip_path.exists():
        return
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to.parent)


def register_project_paths(search_root: Path) -> None:
    candidate_dirs = {WORKSPACE_DIR}
    if search_root.exists():
        candidate_dirs.add(search_root)
        for pattern in ("baseline.py", "scores.py"):
            for match in search_root.rglob(pattern):
                candidate_dirs.add(match.parent)
        for package_name in ("baseline", "scores"):
            for match in search_root.rglob(package_name):
                if match.is_dir() and (match / "__init__.py").exists():
                    candidate_dirs.add(match.parent)
    for path in sorted(candidate_dirs, key=lambda p: len(str(p))):
        path_str = str(path.resolve())
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def prepare_local_environment() -> None:
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    FULL_COMPARISON_SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    install_local_dependencies()
    download_drive_asset(TWEETS_FILE_URL, TWEETS_CSV_PATH)
    download_drive_asset(GRAPH_TOPIC_ZIP_URL, GRAPH_TOPIC_ZIP_PATH)
    extract_zip_if_needed(GRAPH_TOPIC_ZIP_PATH, GRAPH_TOPIC_DIR)
    register_project_paths(GRAPH_TOPIC_DIR)
    import nltk

    for resource in [
        "stopwords",
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]:
        nltk.download(resource, quiet=True)


prepare_local_environment()


# ── Deferred imports (after path registration) ────────────────────────────────
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np
import optuna
import pandas as pd
import psutil
from optuna.samplers import TPESampler
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# v1 metrics are inlined below. Current scores/main.py is not used.
from baseline import (
    get_bertopic_topics,
    get_hdp_topics,
    get_lda_topics,
    get_lsi_topics,
    get_nmf_topics,
    get_topmost_topics,
    preprocess_gensim,
)


# ── FIX: TopMost device mismatch ───────────────────────────────────────────
# TopMost puts the training data on the dataset's `device` (cuda here), but its
# trainers never move the model; the TopMost docs say to call
# `model = model.to(device)` yourself. baseline_models.get_topmost_topics()
# does not, so ProdLDA / HyperMiner / ECRTM failed with
# "mat1 is on cuda:0, different from other tensors on cpu" and returned no
# topics. This patch moves the model to the data's device right before
# training. It lives here (not in run_baseline_on_chunk) so the cache hashes
# stay unchanged and the run resumes from its existing caches.
def _patch_topmost_trainers_device() -> None:
    import torch

    try:
        from topmost.trainers import BasicTrainer, HierarchicalTrainer
    except Exception as exc:  # TopMost missing or different layout
        print(f"  ⚠ TopMost device patch not applied: {exc}")
        return

    def _data_device(dataset):
        data = getattr(dataset, "train_data", None)
        if torch.is_tensor(data):
            return data.device
        try:
            batch = next(iter(dataset.train_dataloader))
            if torch.is_tensor(batch):
                return batch.device
            if isinstance(batch, dict):
                for v in batch.values():
                    if torch.is_tensor(v):
                        return v.device
        except Exception:
            pass
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cls in (BasicTrainer, HierarchicalTrainer):
        if getattr(cls.train, "_device_patched", False):
            continue
        original_train = cls.train

        def train(self, *args, _orig=original_train, **kwargs):
            device = _data_device(self.dataset)
            self.model = self.model.to(device)
            # model.to() only moves registered parameters/buffers. Some TopMost
            # models (HyperMiner via SawETM) also keep plain tensors such as
            # `curvature`, `gam_prior`, `real_min` and a `self.device` string
            # that they were built with (cpu by default). Move those too.
            for module in self.model.modules():
                for attr, value in list(vars(module).items()):
                    if attr.startswith("_"):
                        continue
                    if torch.is_tensor(value) and not isinstance(
                        value, torch.nn.Parameter
                    ):
                        setattr(module, attr, value.to(device))
                    elif attr == "device" and isinstance(value, (str, torch.device)):
                        setattr(
                            module,
                            attr,
                            str(device) if isinstance(value, str) else device,
                        )
            return _orig(self, *args, **kwargs)

        train._device_patched = True
        cls.train = train


_patch_topmost_trainers_device()


# [LOCAL CHANGE 5] Google Drive mount and /content cleanup cells were removed;
# local files now live beside this script:
# shutil.rmtree(str(GRAPH_TOPIC_DIR), ignore_errors=True)
# shutil.rmtree(str(WORKSPACE_DIR / "baseline"), ignore_errors=True)
# shutil.rmtree(str(WORKSPACE_DIR / "scores"), ignore_errors=True)


## GRAPHTOPIC START
# Inlined from v1/GraphTopic without algorithm changes.
# ngram_range is fixed at (2, 3) inside KeyBERT and v1 coherence.
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

reducer = PCA(n_components=3, random_state=45)
model_path_or_name = "sentence-transformers/all-mpnet-base-v2"
# Loaded by the comparison shell (same checkpoint as v1 config.py) and passed in.
embeddingModel = None
th = 0.5

from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer as _KBCountVectorizer

# KeyBERT embeds EVERY distinct 2-3-word phrase in its input before ranking.
# On a merged chunk of long documents (PubMed, NIPS, RTL-Wiki, IMDB) that is
# millions of phrases and the process is killed for running out of RAM.
# Keep only the most frequent phrases as candidates. When a text has fewer
# distinct phrases than this, nothing is dropped and results are unchanged.
# ~100k candidates x 384-dim float32 embeddings is roughly 150 MB.
KEYBERT_MAX_CANDIDATES = 100_000


_KEYBERT_MODEL = None


def _get_keybert_model():
    """Build KeyBERT once and reuse it (was rebuilt on every call)."""
    global _KEYBERT_MODEL
    if _KEYBERT_MODEL is None:
        _KEYBERT_MODEL = KeyBERT()
    return _KEYBERT_MODEL


def getKeyPhrasesUsingKeyBERT(text="", n=15):
    kw_model = _get_keybert_model()
    if not text:
        return []
    # Same settings KeyBERT uses internally (ngram (2, 3), English stop words,
    # min_df=1), plus a cap on the number of candidate phrases.
    vectorizer = _KBCountVectorizer(
        ngram_range=(2, 3),
        stop_words="english",
        min_df=1,
        max_features=KEYBERT_MAX_CANDIDATES,
    )
    try:
        return kw_model.extract_keywords(
            text,
            vectorizer=vectorizer,
            top_n=n,
            # diversity=0.5,
        )
    except ValueError:
        # Empty vocabulary (e.g. only stop words); KeyBERT returns [] here
        # without a custom vectorizer, so keep that behaviour.
        return []


import re
from typing import List
import nltk
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("averaged_perceptron_tagger_eng")
# nltk.download("averaged_perceptron_tagger")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

REPLACE_PUNCTUATION_BY_SPACE_RE = re.compile(r"[/(){}\[\]\|@,:;]")  # exclude full stop
REMOVE_NUM = re.compile(r"[\d+]")
EMAIL_RE = re.compile(r"\b[\w\-.]+?@\w+?\.\w{2,4}\b")
PHONE_RE = re.compile(r"\b(\+\d{1,2}\s?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})\b")
NUMBER_RE = re.compile(r"\d+(\.\d+)?")
URLS_RE = re.compile(r"(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)")
EXTRA_SPACE_RE = re.compile(r"\s+")
STOPWORDS = set(stopwords.words("english"))
# BAD_SYMBOLS: except numbers,character,space,newline,.,_
BAD_SYMBOLS_RE = re.compile(r"[^0-9a-zA-Z _.\n]")
HYPHENATED_RE = re.compile(r"-\n")
SPACE_BEFORE_FULLSTOPS = re.compile(r"\s\.")
MORE_THAN_ONE_FULLSTOPS_RE = re.compile(r"\.\.+")


def cleanText(text, stem=True):
    text = HYPHENATED_RE.sub("", text)  # joining Hyphenated words
    text = REPLACE_PUNCTUATION_BY_SPACE_RE.sub(" ", text)
    text = NUMBER_RE.sub("", text)
    text = EMAIL_RE.sub("", text)
    text = URLS_RE.sub("", text)
    text = PHONE_RE.sub("", text)
    text = BAD_SYMBOLS_RE.sub("", text)
    text = EXTRA_SPACE_RE.sub(" ", text)

    # Lower case
    text = text.lower()

    # Remove Stop Words
    # words = [w for w in words if not w in STOPWORDS]

    if stem:
        # Tokenize
        words = word_tokenize(text)
        # # Stemming
        stemmed_words = [ps.stem(w) for w in words]
        # # Join the words back into one string separated by space,
        stemmed_sen = " ".join(stemmed_words)
        # removing space before full stops
        stemmed_sen = SPACE_BEFORE_FULLSTOPS.sub(".", stemmed_sen)
        # removing more than one full stop
        stemmed_sen = MORE_THAN_ONE_FULLSTOPS_RE.sub("", stemmed_sen)
        return stemmed_sen
    else:
        # removing space before full stops
        text = SPACE_BEFORE_FULLSTOPS.sub(".", text)
        # removing more than one full stop
        text = MORE_THAN_ONE_FULLSTOPS_RE.sub("", text)
        return text


def extract_noun_phrases(text):
    text = cleanText(text, stem=False)
    # print(text)
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)
    # Initialize a list to store the noun phrases for each sentence
    noun_phrases_list = []
    # Iterate over each sentence
    for sentence in sentences:
        # Tokenize the sentence into words and tag them with their part of speech
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        # Define a grammar for noun phrases
        # grammar = r'NP: {<DT>?<JJ>*<NN>}'
        # grammar = r'NP: {<DT>?<JJ>*<NNS|NN|NNP|NNPS|PRP$>+}'
        # grammar = r'NP: {<NNS|NN|NNP|NNPS|PRP$>+}'
        # grammar = r'NP: {(<NN.*>+<JJ.*>?)|(<JJ.*>?<NN.*>+)}'
        grammar = r"NP: {(<NN.*>?)|(<NN.*>+)}"

        chunk_parser = nltk.RegexpParser(grammar)
        chunked_pos_tags = chunk_parser.parse(pos_tags)
        # Extract the noun phrases from the chunked POS tags
        noun_phrases = []
        for subtree in chunked_pos_tags.subtrees():
            if subtree.label() == "NP":
                noun_phrase = " ".join([word for (word, pos) in subtree])
                noun_phrases.append(noun_phrase)
        # Append the list of noun phrases for the current sentence to the list
        noun_phrases_list.append(noun_phrases)
        noun_phrases_list.append(".")
    # Flatten the list of lists of noun phrases into a single list
    flattened_noun_phrases = [
        noun_phrase for sublist in noun_phrases_list for noun_phrase in sublist
    ]
    # Join the noun phrases into a single string with full stops
    return " ".join(flattened_noun_phrases).replace(" .", ".")


def split_into_chunks(raw_docs, chunk_size, overlap_ratio=0):
    chunked_documents = []
    overlap = int(chunk_size * overlap_ratio)

    for i in range(0, len(raw_docs), chunk_size - overlap):
        chunk = raw_docs[i : i + chunk_size]
        # print(i, i + chunk_size)
        chunked_documents.append(chunk)

    # print()
    # print(len(chunked_documents))

    return chunked_documents


def _preprocess_text(documents, deep_clean=False) -> List[str]:
    """Basic preprocessing of text

    Steps:
        * Replace \n and \t with whitespace
        * Only keep alpha-numerical characters
        * Apply PorterStemmer for word stemming (optional)
        * Remove stopwords
        * Remove documents with less than 5 words
    """
    cleaned_documents = [doc.lower() for doc in documents]
    cleaned_documents = [doc.replace("\n", " ") for doc in cleaned_documents]
    cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]
    if deep_clean:
        cleaned_documents = [
            re.sub(r"[^A-Za-z0-9 . -]+", "", doc) for doc in cleaned_documents
        ]
        cleaned_documents = [
            doc if doc != "" else "emptydoc" for doc in cleaned_documents
        ]

    # Remove documents with less than 5 words
    cleaned_documents = [doc for doc in cleaned_documents if len(doc.split()) >= 5]

    return cleaned_documents


import os

from pprint import pprint
import pandas as pd
import numpy as np
from netgraph import Graph
import warnings

warnings.filterwarnings("ignore")


from pprint import pprint
from copy import copy
from typing import TypeVar, List, Dict, Union, Literal
from dataclasses import dataclass
from collections import deque
from scipy.spatial.distance import cdist

import networkx as nx
import math

# Create a graph object
import matplotlib.pyplot as plt


@dataclass
class KeyPhrase:
    paper_id: str
    keyphrase: str
    year: int = 0
    score: float = 0

    def __hash__(self):
        return hash(self.keyphrase)

    def __eq__(self, other):
        if isinstance(other, KeyPhrase):
            return self.keyphrase == other.keyphrase
        return False


NodeType = TypeVar("NodeType", bound="Node")
InfoType = Dict[Literal["keyphrase", "topic_index"], Union[KeyPhrase, int]]
AdjacencyList = Dict[NodeType, List[NodeType]]


class Node:
    def __init__(self, data: KeyPhrase):
        self._paper_id = data.paper_id
        self._keyphrase = data.keyphrase
        self._year = data.year

    def __str__(self):
        return f"Node({self._keyphrase})"

    def __repr__(self):
        return str(self)

    @property
    def id(self):
        return self._paper_id

    @property
    def keyphrase(self):
        return copy(self._keyphrase)

    @property
    def year(self):
        return self._year

    def __eq__(self, other: NodeType):
        return True if self._keyphrase == other.keyphrase else False

    def __hash__(self):
        return hash(self._keyphrase)


class TopicManager:
    def __init__(
        self,
    ):
        self._topics: List[AdjacencyList] = []
        self._csv_reports = pd.DataFrame(
            columns=["topic", "node", "neighbor", "weight"]
        )
        self._all_kps_found_history = []

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self._topics)

    def add_new_topic_from_keyphrases(self, kps: list[KeyPhrase]) -> None:
        adj_list = {}
        for item in kps:
            node = Node(item)
            adj_list[node] = []
            for neighbor in kps:
                if neighbor.keyphrase != node.keyphrase:
                    neighbor_node = Node(neighbor)
                    adj_list[node].append((neighbor_node, 1))
        self._topics.append(adj_list)

    def print_details(self):
        for i, topic in enumerate(self._topics):
            print(f"Domain[{i}]")
            for key in topic.keys():
                print(f"   {key}:{'-' * (25 - len(str(key)))}->{topic[key]}")
        print()

    def __get_topic_indices_by_keyphrase(self, kp: KeyPhrase) -> list[tuple[int]]:
        index_list = []
        for i, adj_list in enumerate(self._topics):
            for node in adj_list.keys():
                if node.keyphrase == kp.keyphrase:
                    index_list.append(i)
        return index_list

    def __make_connectivity(self, nodes: List[NodeType], graph: AdjacencyList):
        # Create a set of all the nodes in the adjacency list
        all_nodes = list(graph.keys())
        get_common_nodes = list(set(all_nodes) & set(nodes))
        # print(get_common_nodes)
        for node in get_common_nodes:
            for node_to_add in nodes:
                if node_to_add != node:
                    adj_nodes = graph[node]
                    if node_to_add not in self.__get_only_nodes(adj_nodes):
                        adj_nodes.append((node_to_add, 1))

    def __merge_graphs(self, graph1, graph2, merged_graph):
        # Iterate over the adjacency lists of both graphs
        for graph in [graph1, graph2]:
            for vertex, edges in graph.items():
                # Check if the vertex already exists in the new adjacency list
                if vertex in merged_graph:
                    # Add the unique edges from the entry to the list of edges for the vertex
                    merged_graph[vertex] += list(set(edges) - set(merged_graph[vertex]))
                else:
                    # Add a new entry to the list with the vertex and its edges
                    merged_graph[vertex] = edges
        return merged_graph

    # def get_node_from_topic(self, kp: KeyPhrase):
    #     for adj_list in self._topics:
    #         for node in adj_list.keys():
    #             if node.keyphrase == kp.keyphrase:
    #                 return node

    # def get_topic_by_node(self, node: NodeType):
    #     for i, adj_list in enumerate(self._topics):
    #         for n in adj_list.keys():
    #             if n == node:
    #                 return i

    # def get_keyphrase_obj_from_list(self, kps_list):
    #     kps_obj_list = []
    #     for kp_string in kps_list:
    #         kp_job = KeyPhrase("", kp_string)
    #         kps_obj_list.append(kp_job)
    #     return kps_obj_list

    def __get_only_nodes(self, nodes_weight_tuple_list) -> List[NodeType]:
        return [t[0] for t in nodes_weight_tuple_list]

    def merge_subtopics(self, kps: list[KeyPhrase]):
        # find topics that carry the passed keyphrases
        found_list: List[InfoType] = []
        for kp in kps:
            indices = self.__get_topic_indices_by_keyphrase(kp)
            if indices:
                for index in indices:
                    info: InfoType = {
                        "keyphrase": kp,
                        "topic_index": index,
                    }
                    found_list.append(info)
        # pprint(found_list)
        if not found_list:
            return

        # get unique topics
        unique_topic = set()
        unique_kps = set()
        for item in found_list:
            unique_topic.add(item["topic_index"])
            unique_kps.add(item["keyphrase"])
        unique_topic_list = list(unique_topic)
        unique_kps = list(unique_kps)
        # print(unique_kp)

        if len(unique_topic_list) < 2:
            return
        # unique_topic_list = sorted(unique_topic_list)
        base_topic_index = unique_topic_list[0]
        to_merge_indices = unique_topic_list[1:]

        if len(to_merge_indices) >= 1:
            q = deque(to_merge_indices)
            while q:
                merged_graph = {}
                topic_to_merge_index = q.popleft()
                graph1 = self._topics[base_topic_index]
                graph2 = self._topics[topic_to_merge_index]
                self._topics[base_topic_index] = self.__merge_graphs(
                    graph1, graph2, merged_graph
                )

        self.__make_connectivity(
            [Node(kp) for kp in unique_kps], self._topics[base_topic_index]
        )
        # print(self._topics[base_topic_index])

        # remove from self._topics
        if to_merge_indices:
            for index in sorted(
                to_merge_indices, reverse=True
            ):  # reversed because list shrinks when popped
                try:
                    self._topics.pop(index)
                except Exception as e:
                    print(unique_topic)
                    print(index)
                    print(e)

    def clear_topic_with_node_count_n(self, n):
        self._topics = [d for d in self._topics if len(d) > n]
        self._topics = sorted(self._topics, key=lambda d: len(d), reverse=True)

    def get_topics(self, kps_per_topic=None):
        data = {}
        for i, topic in enumerate(self._topics):
            all_kps = set()
            for node in topic.keys():
                all_kps.add(node.keyphrase)
                for neighbor, weight in topic[node]:
                    all_kps.add(neighbor.keyphrase)
            data[i] = list(all_kps)[:kps_per_topic]
        return data

    def generate_csv(self, path):
        for i, topic in enumerate(self._topics):
            data = []
            for node in topic.keys():
                for neighbor, weight in topic[node]:
                    node_info = {}
                    node_info["topic"] = i
                    node_info["node"] = node.keyphrase
                    node_info["neighbor"] = neighbor.keyphrase
                    node_info["weight"] = weight
                    node_info["node_year"] = node.year
                    node_info["neighbor_year"] = neighbor.year
                    node_info["node_paper_id"] = node.id
                    node_info["neighbor_paper_id"] = neighbor.id
                    data.append(node_info)
            topic_df = pd.DataFrame(data)
            # concat
            self._csv_reports = pd.concat(
                [self._csv_reports, topic_df], ignore_index=True
            )
            #
            self.latent_df = pd.concat([self.latent_df, topic_df], ignore_index=True)

        self._csv_reports.to_csv(path, index=False)

    def generate_df(self):
        latent_df = pd.DataFrame(columns=["topic", "node", "neighbor"])
        for i, topic in enumerate(self._topics):
            data = []
            for node in topic.keys():
                for neighbor, weight in topic[node]:
                    node_info = {}
                    node_info["topic"] = i
                    node_info["node"] = node.keyphrase
                    node_info["neighbor"] = neighbor.keyphrase
                    data.append(node_info)
            topic_df = pd.DataFrame(data)
            latent_df = pd.concat([latent_df, topic_df], ignore_index=True)
        return latent_df

    @property
    def topics(self):
        return copy(self._topics)

    def draw_topics(self, show=True, save=False, save_path="", drawCluster=False):
        n = len(self._topics)
        n = n + 1 if n == 1 else n
        if drawCluster:
            fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(5 * n, 5 * n))
        else:
            fig, axes = plt.subplots(
                nrows=n, ncols=1, figsize=(20, (20 + (15 * n)))
            )  # (20,  (20 + (10*n))))
        ax = axes.flatten()
        subgraphs = []
        for topic in self._topics:
            G = nx.Graph()
            nodes = set()
            for node in topic.keys():
                G.add_node(node.keyphrase)
                nodes.add(node.keyphrase)
                for adj_node, weight in topic[node]:
                    G.add_edge(node.keyphrase, adj_node.keyphrase)
                    nodes.add(adj_node.keyphrase)

            subgraph = nx.subgraph(G, list(nodes))
            subgraphs.append(subgraph)

        for i in range(len(subgraphs)):
            G = subgraphs[i]
            pos = nx.kamada_kawai_layout(
                G
            )  # kamada_kawai_layout | spring_layout(G,k=0.03)
            # Specify node positions using pos argument
            nx.draw(
                G,
                pos=pos,
                ax=ax[i],
                node_size=30,
                edge_color="gray",
                node_color="skyblue",
                width=0.5,
            )
            # Specify offset parameter to push labels up 2 pixels
            # label_pos = {node: (x, y + .01) for node, (x, y) in pos.items()}
            nx.draw_networkx_labels(
                G, pos=pos, ax=ax[i], font_size=6, font_color="black"
            )
            # Set the title for each subplot with a larger font size and red color
            if drawCluster:
                ax[i].set_title(f"Cluster {i}", fontsize=16, color="red")
            else:
                ax[i].set_title(f"Domain {i}", fontsize=16, color="red")

        if save_path != "":
            plt.savefig(save_path, format="pdf", dpi=300, transparent=True)

        if show:
            plt.show()
        else:
            plt.close()


# print("⌛ Importing libraries...")
from typing import List, Dict
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
from rich.console import Console
from rich.table import Table
from enum import Enum

# from helpers.config import reducer, th

from scipy.integrate import quad
import itertools
import random

# Set the seed to ensure reproducibility
seed_value = 42
random.seed(seed_value)


# Projecting the data points onto the line
def projection_onto_line(data_points, a, b):
    # Finding the unit vector along the line
    line_vector = b - a
    unit_vector = line_vector / np.linalg.norm(line_vector)
    unit_vector = unit_vector[np.newaxis, :]

    # Calculating the projection of each data point onto the line
    projections = np.dot(data_points - a, unit_vector.T) * unit_vector + a
    return projections


def pdf(data, mean: float, variance: float):
    # A normal continuous random variable.
    s1 = 1 / (np.sqrt(2 * np.pi * variance))
    s2 = np.exp(-(np.square(data - mean) / (2 * variance)))
    return s1 * s2


def merge_indices(indices_list):
    """
    INPUT: (list of to be overlapped pairs:)
     [(0, 1), (0, 2),(3, 4), (3, 5),(7,8)]
    OUTPUT:
     [[0, 1, 2], [3, 4, 5], [7, 8]]

     Explanation:
     > (0,1) -> 0 to be merged with 1
     > (0,2) -> 0 to be merged with 2

     that means 0,1 and 2 should be merged together
     so the output is (0,1,2)

    """
    # Create an empty list to hold the merged indices.
    merged_list = []

    # Loop through the indices in the given list.
    for indices in indices_list:
        merged = False
        # Loop through the existing clusters in the merged list.
        for i, cluster in enumerate(merged_list):
            # If any of the indices in the current indices are already
            # in the cluster, add the remaining indices to the cluster.
            if any(idx in cluster for idx in indices):
                cluster.extend(set(indices) - set(cluster))
                merged = True
                break
        # If the indices were not merged with any existing clusters,
        # create a new cluster with the given indices.
        if not merged:
            merged_list.append(list(indices))

    # Return the merged list of indices.
    return merged_list


def get_overlap(indices, X_df):
    idx1, idx2 = indices
    X_val = X_df[X_df["topic"].isin(indices)].drop(columns=["topic"])
    # reduction= PCA(n_components=min(X_val.shape[0], X_val.shape[1],10)-1, random_state=45)
    # min(X_val.shape[0], X_val.shape[1],30)-1
    X_reduced = reducer.fit_transform(X_val)
    X_reduced = pd.DataFrame(X_reduced)
    X_reduced["topic"] = X_df[X_df["topic"].isin(indices)]["topic"].tolist()
    C1 = X_reduced[X_reduced["topic"] == idx1].drop(columns=["topic"]).to_numpy()
    C2 = X_reduced[X_reduced["topic"] == idx2].drop(columns=["topic"]).to_numpy()
    c1 = np.mean(C1, axis=0)
    c2 = np.mean(C2, axis=0)
    # Defining the line passing through the center points
    a = c1
    b = c2
    # Projecting the data points onto the line
    C1_projections = projection_onto_line(C1, a, b)
    C2_projections = projection_onto_line(C2, a, b)
    # left most point
    pl = np.minimum(C1_projections.min(axis=0), C2_projections.min(axis=0))
    PC1 = np.zeros((C1.shape[0],))
    for i, p in enumerate(C1_projections):
        PC1[i] = np.linalg.norm(p - pl)
    PC2 = np.zeros((C2.shape[0],))
    for i, p in enumerate(C2_projections):
        PC2[i] = np.linalg.norm(p - pl)

    overlap, _ = quad(
        lambda x: np.minimum(
            pdf(x, PC1.mean(), PC1.var()), pdf(x, PC2.mean(), PC2.var())
        ),
        -np.inf,
        np.inf,
    )
    return overlap


def merge_clusters(topic_pair_combinations, th, embedding_df, flatten_df):
    """
    INPUT: (list of pairs:)
     [(0, 1), (0, 2),(3, 4), (3, 5),(7,8)]
    OUTPUT:
     New embedding_df, flatten_df if overlapping threshold met or None,None
    """

    # [(0, 1), (0, 2), (0, 3),....
    similar_pair_list = []
    for each_pair in topic_pair_combinations:
        overlap = get_overlap(each_pair, embedding_df)

        # print(each_pair,round(overlap,2),f"th={round(th,2)}")
        if overlap >= th:
            similar_pair_list.append(each_pair)
    merged_list = merge_indices(similar_pair_list)  # [[0, 1, 2], [3, 4, 5]]
    if len(merged_list) < 1:
        return None, None
    else:
        for i, mergedIndices in enumerate(merged_list):
            embedding_df.loc[embedding_df["topic"].isin(mergedIndices), "topic"] = (
                mergedIndices[0]
            )
            flatten_df.loc[flatten_df["topic"].isin(mergedIndices), "topic"] = (
                mergedIndices[0]
            )
            return embedding_df, flatten_df


def recursive_merge_clusters(embedding_df, flatten_df, th):
    """
    Recursively merge clusters until no more merging is possible, and return the final merged clusters.
    """
    # global th
    # global counter
    # th = th + counter * alpha
    # counter += 1
    # 1. get all topics
    current_topics = embedding_df["topic"].unique().tolist()  # [0,1,2,3,4,5,6]
    # print(current_topics)
    topic_pair_combinations = list(
        itertools.combinations(current_topics, 2)
    )  # [(0, 1), (0, 2), (3, 4), (3, 5)]
    embedding_df_new, flatten_df_new = merge_clusters(
        topic_pair_combinations, th, embedding_df, flatten_df
    )  # [[0, 1, 2], [3, 4, 5]] # type: ignore
    if embedding_df_new is not None:
        return recursive_merge_clusters(embedding_df_new, flatten_df_new, th)
    else:
        # print(print(embedding_df["topic"].unique().tolist()))
        return embedding_df, flatten_df


# embedding_df, flatten_df = recursive_merge_clusters(embedding_df, flatten_df)


def update_topic(df, flatten_df):
    # make a copy of df to avoid modifying the original dataframe
    new_df = df.copy()
    # loop through rows of flatten_df and update new_df
    for idx, row in flatten_df.iterrows():
        text = row["text"]
        topic = row["topic"]

        mask1 = new_df["neighbor"] == text
        mask2 = new_df["node"] == text
        new_df.loc[mask1, "topic"] = topic
        new_df.loc[mask2, "topic"] = topic

    return new_df


# def remove_outliers(embedding_df, threshold):
#     embeddings = embedding_df.to_numpy()
#     distance_matrix = cdist(embeddings, embeddings, "cosine")

#     similarity_matrix = 1 - distance_matrix
#     similarity_matrix[np.isnan(similarity_matrix)] = 0

#     outlier_indices = []
#     for i in range(len(embeddings)):
#         similarity_scores = similarity_matrix[i, :]
#         num_similar_embeddings = np.sum(similarity_scores >= threshold)

#         if num_similar_embeddings <= 1:  # Exclude the embedding itself
#             outlier_indices.append(i)

#     filtered_embeddings = np.delete(embeddings, outlier_indices, axis=0)
#     filtered_df = pd.DataFrame(filtered_embeddings)

#     return filtered_df


def get_most_outdegree_node(graph):
    # Calculate out-degree for each node in the graph
    outdegrees = graph.out_degree()

    # Find the node with the maximum out-degree
    most_outdegree_node = max(outdegrees, key=lambda x: x[1])

    return most_outdegree_node[0]


def get_weakly_connected_subgraphs(df):
    # Create an empty directed graph
    graph = nx.DiGraph()

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        topic = row["topic"]
        node = row["node"]
        neighbor = row["neighbor"]

        # Add nodes to the graph
        graph.add_node(node, topic=topic)
        graph.add_node(neighbor, topic=topic)

        # Add edges to the graph
        graph.add_edge(node, neighbor, topic=topic)

    # Find weakly connected components
    weakly_connected_components = list(nx.weakly_connected_components(graph))

    # Create a list to store subgraphs
    subgraphs = []

    # Iterate over each weakly connected component
    for component in weakly_connected_components:
        # Create a subgraph for the component
        subgraph = graph.subgraph(component)
        subgraphs.append(subgraph)

    return subgraphs


def process_topics(df):
    for topic in df["topic"].unique().tolist():
        topic_df = df[df["topic"] == topic]
        subgraphs = get_weakly_connected_subgraphs(topic_df)

        # find nodes with most outdegree from each sub graphs
        most_outdegree_node_str_list = []
        for i, subgraph in enumerate(subgraphs):
            most_outdegree_node_str = get_most_outdegree_node(subgraph)
            most_outdegree_node_str_list.append(most_outdegree_node_str)

        # Create a bidirectional link between them
        for node in most_outdegree_node_str_list:
            for neighbor in most_outdegree_node_str_list:
                if node != neighbor:
                    # node_info = df[df["node"] == node].iloc[0]
                    # neighbor_info = df[df["node"] == neighbor].iloc[0]
                    link = {
                        "topic": topic,
                        "node": node,
                        "neighbor": neighbor,
                        # "node_year": node_info["node_year"],
                        # "neighbor_year": neighbor_info["neighbor_year"],
                        # "node_paper_id": node_info["node_paper_id"],
                        # "neighbor_paper_id": neighbor_info["neighbor_paper_id"],
                    }
                    df = pd.concat([df, pd.DataFrame([link])], ignore_index=True)
    return df


# def _print_topics(flatten_df):
#     # text groupby
#     Domains_text_list = (
#         flatten_df.groupby("topic")["text"].apply(list).reset_index()["text"].tolist()
#     )
#     # list of list to list of string
#     Domains_text_list = [" ".join(each) for each in Domains_text_list]
#     for text in Domains_text_list:
#         print(text)
#         print()


def _print_topics(df):
    topic_keyphrases = (
        df.groupby("topic")["text"].apply(list).reset_index().values.tolist()
    )

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")

    # Add columns for each topic
    for topic, keyphrases in topic_keyphrases:
        table.add_column(str(topic))

    # Get the maximum number of keyphrases for any topic
    max_keyphrases = max(len(keyphrases) for _, keyphrases in topic_keyphrases)

    # Add rows for keyphrases for each topic
    for i in range(max_keyphrases):
        row = []
        for _, keyphrases in topic_keyphrases:
            if i < len(keyphrases):
                row.append(keyphrases[i])
            else:
                row.append("")  # Fill empty cells with empty string
        table.add_row(*row)

    console.print(table)
    console.print()


# Rest of the code...


def flatten_graph_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df: DataFrame(cols=[topic,node,neighbor])
    Output df :-> DataFrame(cols=[topic,text])

    """
    new_doc: List[Dict[str, str]] = []
    for topic in df["topic"].unique().tolist():
        topic_df = df[df["topic"] == topic]
        nodes = topic_df["node"].unique().tolist()
        neighbors = topic_df["neighbor"].unique().tolist()
        doc = list(set(neighbors + nodes))
        data = [{"topic": topic, "text": v} for v in doc]
        new_doc.extend(data)
    flatten_df = pd.DataFrame(new_doc)
    return flatten_df


def flatten_df_to_dict(flatten_df: pd.DataFrame) -> Dict[int, List[str]]:
    result_dict: Dict[int, List[str]] = {}
    for topic in flatten_df["topic"].unique().tolist():
        topic_texts = flatten_df[flatten_df["topic"] == topic]["text"].tolist()
        result_dict[topic] = topic_texts
    return result_dict


def topic_dict_to_list(topic_dict, max_topics=None):
    if not max_topics:
        return list(topic_dict.values())
    else:
        return list(topic_dict.values())[:max_topics]


def get_merged_topics(
    latent_graph_df: pd.DataFrame,
    embeddingModel: SentenceTransformer,
    threshold=0.5,
    pretty_print=True,
):
    """
    Corresponding Columns:
    latent_graph_df,updated_graph_df :->  DataFrame(cols=[topic,node,neighbor])
    flatten_df,merged_df :-> DataFrame(cols=[topic,text])
    """
    flatten_df = flatten_graph_df(latent_graph_df)
    phrase_embs = embeddingModel.encode(flatten_df["text"].tolist())
    embedding_df = pd.DataFrame(np.array(phrase_embs.tolist()))  # type: ignore
    embedding_df["topic"] = flatten_df["topic"]
    _, merged_df = recursive_merge_clusters(embedding_df, flatten_df, threshold)
    # if pretty_print:
    #     print(merged_df.shape)
    #     _print_topics(merged_df)
    #     # print(updated_graph_df.shape)  # type: ignore

    # Also update original `df`
    updated_graph_df = update_topic(latent_graph_df, merged_df)
    # print(updated_graph_df.shape)
    # make connection between subgraphs within same topics
    final_graph_df = process_topics(updated_graph_df)
    final_flatten_df = flatten_graph_df(final_graph_df)

    _topics: Dict[int, List[str]] = {}

    _topics = flatten_df_to_dict(final_flatten_df)

    return final_graph_df, final_flatten_df, topic_dict_to_list(_topics)


# Get top keyphrases using cosine distance
def get_top_cdist_kps(
    kps: List[str], embeddingModel: SentenceTransformer, max_keyphrases_per_topic=10
):
    embeddings = embeddingModel.encode(kps)
    distance_matrix = cdist(embeddings, embeddings, "cosine")

    similar_indices = np.argsort(np.sum(distance_matrix, axis=0))[
        :max_keyphrases_per_topic
    ]
    similar_kps = [kps[i] for i in similar_indices]

    # filtered_embeddings = np.delete(embeddings, similar_indices, axis=0)
    # filtered_df = pd.DataFrame(filtered_embeddings)
    # return similar_kps, filtered_df

    return similar_kps


class KeyphrasesSelectionStrategy(Enum):
    IMP_NODE = "outdegree"
    CDIST = "cdist"
    RNDOM = "random"


def get_topics_with_pruned_kps(
    flatten_df,
    embeddingModel: SentenceTransformer,
    max_keyphrases_per_topic=10,
    keyphrases_selection_strategy=KeyphrasesSelectionStrategy.CDIST,
    max_topics=None,
):
    # Initialize the topics dictionary
    _topics = {}
    # Group the text column by the topic column
    grouped = flatten_df.groupby("topic")["text"]
    # Iterate over each group
    for topic, group in grouped:
        # Extract the list of texts for the current topic
        kps = group.tolist()
        # Add the topic and its corresponding texts to the topics dictionary
        if keyphrases_selection_strategy == KeyphrasesSelectionStrategy.IMP_NODE:
            # Handle "first" strategy
            _topics[topic] = kps[:max_keyphrases_per_topic]
        elif keyphrases_selection_strategy == KeyphrasesSelectionStrategy.CDIST:
            # Handle "cdist" strategy
            _topics[topic] = get_top_cdist_kps(
                kps, embeddingModel, max_keyphrases_per_topic
            )
        elif keyphrases_selection_strategy == KeyphrasesSelectionStrategy.RNDOM:
            # Handle "random" strategy
            _topics[topic] = random.sample(kps, min(max_keyphrases_per_topic, len(kps)))
        else:
            # Handle other cases or raise an exception for invalid strategy
            raise ValueError("Invalid keyphrases_selection_strategy")

    # pprint(_topics)

    return topic_dict_to_list(_topics, max_topics=max_topics)


import pandas as pd

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# type
from typing import List, Tuple, Optional, Callable


class GraphTopic:
    """
    GraphTopic: A class for extracting and merging topics from a list of documents using a graph-based approach.

    This class implements a graph-based topic extraction method. It first extracts keyphrases from each document and the merged document.
    Then, it builds an initial graph of trivial topics based on the keyphrases from the merged document.
    Next, it iteratively merges these trivial topics by processing each document and its keyphrases.
    Finally, it merges similar topics based on embedding similarity to generate the final set of topics.

    The main steps involved are:
    1. Keyphrase Extraction: Extract keyphrases from the merged document (for initial topic creation) and each individual document (for topic merging).
    2. Trivial Topic Initialization: Create initial topics based on the top keyphrases from the merged document.
    3. Latent Topic Extraction: Merge trivial topics by processing each document and its keyphrases, building a graph of related topics.
    4. Final Topic Merging: Merge semantically similar latent topics based on keyphrase embeddings to produce a concise set of final topics.

    The algorithm leverages keyphrase embeddings and graph structures to identify and merge related topics,
    allowing for a more nuanced and comprehensive topic representation compared to traditional methods.

    Args:
        docs (List[str]): List of document texts to analyze.
        TopM (int, optional):  Determines the granularity of initial trivial topics.
                                Higher TopM (↑) considers more keyphrases from the merged document, potentially leading to more granular and specific initial topics.
                                Defaults to 500.
        TopN (int, optional):  Controls the keyphrases extracted per document during latent topic extraction.
                                Lower TopN (↓) extracts fewer keyphrases per document, potentially leading to broader and more general latent topics.
                                Defaults to 10.
        keyphrase_extractor (Callable[[str, int], List[Tuple[str, float]]], optional):
            Function to extract keyphrases from text.
            It should take text and the number of keyphrases to extract as input and return a list of tuples,
            where each tuple contains a keyphrase (str) and its weight (float).
            Defaults to `getKeyPhrasesUsingKeyBERT`.
            Example:
            ```python
            def getKeyPhrases(text, n):
                # ... keyphrase extraction logic ...
                return [("keyphrase1", 0.9), ("keyphrase2", 0.8), ...]
            ```
        embedding_model (SentenceTransformer, optional): SentenceTransformer model for embedding keyphrases during topic merging.
                                                        Defaults to `embeddingModel` from `GraphTopic.helpers.config`.
        pretty_print (bool, optional): Enables pretty printing of intermediate steps for debugging and visualization.
                                       Defaults to False.
        prune_node_count_less_than (int, optional):  Minimum number of nodes (keyphrases) a topic must have to be retained in the latent topic graph.
                                                       Topics with fewer nodes are pruned, helping to remove very specific or less relevant topics.
                                                       Defaults to 2.
        overlapping_threshold (float, optional):  Threshold for merging topics based on embedding similarity during final topic merging.
                                                  A higher threshold means topics need to be more similar to be merged, resulting in more, potentially more specific, final topics.
                                                  Defaults to 0.5.
        max_topics (float, optional): Maximum number of final topics to retrieve. If 'inf', all topics are returned.
                                       Defaults to float("inf").
        max_keyphrases_per_topic (int, optional): Maximum number of keyphrases to include in each final topic.
                                                     Defaults to 10**2 (100).
        keyphrases_selection_strategy (KeyphrasesSelectionStrategy, optional):
            Strategy for selecting top keyphrases to represent each final topic when `max_topics` is specified.
            - KeyphrasesSelectionStrategy.IMP_NODE (default): Selects keyphrases based on outdegree node centrality in the topic graph (most important nodes).
            - KeyphrasesSelectionStrategy.CDIST: Selects keyphrases based on cosine similarity to the topic centroid embedding (closest to the topic center).
            - KeyphrasesSelectionStrategy.RNDOM: Selects keyphrases randomly.
            Defaults to KeyphrasesSelectionStrategy.CDIST.

    Usage:
        ```python
        from GraphTopic import GraphTopic

        documents = [
            "Document 1 text...",
            "Document 2 text...",
            # ... more documents
        ]

        graph_topic_model = GraphTopic(
            docs=documents,
            TopM=300,
            TopN=8,
            overlapping_threshold=0.6,
        )

        topics = graph_topic_model.get_topics()
    """

    def __init__(
        self,
        docs: List[str],
        TopM: int = 500,
        TopN: int = 10,
        keyphrase_extractor: Callable[
            [str, int], List[Tuple[str, float]]
        ] = getKeyPhrasesUsingKeyBERT,
        embedding_model: SentenceTransformer = embeddingModel,
        pretty_print=False,
        prune_node_count_less_than=2,
        overlapping_threshold=0.5,
        max_topics: float = float("inf"),
        max_keyphrases_per_topic: int = 10**2,
        keyphrases_selection_strategy=KeyphrasesSelectionStrategy.CDIST,
    ):
        """
        Initializes the GraphTopic model with documents and configurations.

        Args:
            docs (List[str]): List of document texts.
            TopM (int): Top M keyphrases to use for merging topics.
            TopN (int): Top N keyphrases to extract from each document for latent topic extraction.
            keyphrase_extractor (callable): Keyphrase extraction function.
                def getKeyPhrases(text,n):
                    return [(kp1,weight),(kp2,weight)]
            prune_node_count_less_than (int): Number of nodes to prune from the graph.
            pretty_print (bool): Flag to enable/disable pretty printing of merging steps.
            overlapping_threshold (float): Threshold for merging topics based on embedding similarity.
            max_keyphrases_per_topic (int): Maximum keyphrases per topic in the final output.
            max_topics (int): Maximum number of topics to retrieve in the final output.
            keyphrases_selection_strategy (KeyphrasesSelectionStrategy):
                Strategy for selecting top keyphrases from each topic:
                'IMP_NODE'(default;based on outdegree node centrality),
                'CDIST' (cosine similarity to topic centroid),
                'RNDOM'(random).
        """
        self._TopM = TopM
        self._TopN = TopN
        self._docs = docs
        self._pretty_print = pretty_print
        self._keyphrase_extractor = keyphrase_extractor
        self._overlapping_threshold = overlapping_threshold
        self._max_topics = max_topics
        self._max_keyphrases_per_topic = max_keyphrases_per_topic
        self._keyphrases_selection_strategy = keyphrases_selection_strategy
        self._prune_node_count_less_than = prune_node_count_less_than
        self._embedding_model = embeddingModel
        self._final_graph_df = None
        self._final_flatten_df = None
        self.topics: List[List] = []

        if self._keyphrase_extractor is None:
            raise Exception("Please provide a keyphrase extractor function")

        # !get merged kps from merged docs
        self._merged_kps: List[str] = self.__extract_kps_merged_from_merged_doc()
        self.__extract_latent_topics()

    def __get_kps(
        self,
        text: str,
        n: int,
        keyphrase_extractor: Callable[[str, int], List[Tuple[str, float]]],
    ) -> List[str]:
        """
        Extracts keyphrases from the given text using the provided keyphrase extractor function.

        Args:
            text (str): Input text from which to extract keyphrases.
            n (int): Number of keyphrases to extract.
            keyphrase_extractor(text, n): Function that takes text and n and returns a list of keyphrase tuples (keyphrase, weight).

        Returns:
            List[str]: List of extracted keyphrases (keyphrases only, weights are discarded).
        """
        kps = keyphrase_extractor(text, n)
        return [kp for kp, _ in kps]

    def __extract_kps_merged_from_merged_doc(self) -> List[str]:
        """
        Extracts the top M keyphrases from the merged document.

        The merged document is created by joining all input documents with ". " separator.
        These keyphrases are used to initialize trivial topics in the latent topic extraction phase.

        Returns:
            List[str]: List of keyphrases extracted from the merged document.
        """
        merged_doc = ". ".join(self._docs)
        # if len(merged_doc) >= 1_000_000:
        #     raise Exception(
        #         "Merged document length is 1_000_000. Please increase the length of the documents."
        #     )
        print(f"> Extracting top M:{self._TopM} keyphrases from merged documents...")
        extracted = self.__get_kps(
            merged_doc,
            n=self._TopM,
            keyphrase_extractor=self._keyphrase_extractor,
        )
        print(f"Total {len(extracted)} keyphrases extracted\n")
        return extracted

    def __extract_latent_topics(self) -> None:
        """
        Extracts latent topics from the documents using a graph-based merging approach.

        This method performs the following steps:
        1. Initializes trivial topics: Creates initial topics based on the top M keyphrases extracted from the merged document.
           Each keyphrase becomes a trivial topic. Only multi-word keyphrases are considered to avoid single-word topics.
        2. Merges subtopics: Iterates through each document and its top N keyphrases. For each document's keyphrases, it merges them into existing trivial topics
           if they are semantically related. This builds a graph where topics are connected if they share keyphrases from the same document.
        3. Prunes topics: Removes topics with a node count (number of associated keyphrases) less than `self._prune_node_count_less_than`.
           This step filters out very specific or less significant topics.
        4. Generates latent topic DataFrame: Creates a Pandas DataFrame representing the latent topic graph.
        5. Flattens DataFrame:  Transforms the graph DataFrame into a flattened DataFrame and dictionary for easier access and processing.

        The `self.latent_topics` attribute is updated with a list of dictionaries, where each dictionary represents a latent topic
        and contains its keyphrases and related information.

        Returns:
            None
        """
        try:
            d = TopicManager()

            # building trivial topics
            print(
                f"> Initializing {len(self._merged_kps)} trivial topics from top M keyphrases..."
            )
            for kp in self._merged_kps:
                # Only Considering Phrase, single words are avoid [as we are using phrase-bert]
                if len(str(kp).split(" ")) < 2:
                    continue
                d.add_new_topic_from_keyphrases([KeyPhrase(paper_id="", keyphrase=kp)])

            print("> Extracting latent topics...")
            for doc in tqdm(self._docs):
                kps = self.__get_kps(
                    doc,
                    n=self._TopN,
                    keyphrase_extractor=self._keyphrase_extractor,
                )
                kps_record = []
                for kp in kps:
                    kps_record.append(KeyPhrase(paper_id="", keyphrase=kp))
                d.merge_subtopics(kps_record)

            d.clear_topic_with_node_count_n(self._prune_node_count_less_than)
            latent_graph_df = d.generate_df()

            flatten_latent_df = flatten_graph_df(latent_graph_df)
            latent_topics_dicts = flatten_df_to_dict(flatten_latent_df)

            print(f"Total {len(latent_topics_dicts)} latent subtopics generated\n")

            self._latent_graph_df = latent_graph_df
            self._flatten_latent_df = flatten_latent_df
            self.latent_topics = latent_topics_dicts

        except Exception as e:
            print(e)

    def __extract_final_topics(
        self, th
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, List[List]]]:
        """
        Merges semantically similar latent topics to generate the final set of topics.

        This method uses keyphrase embeddings and a cosine similarity threshold (`th`) to merge latent topics.
        Topics with an embedding similarity greater than or equal to `th` are merged into a single final topic.

        Args:
            th (float): Cosine similarity threshold for merging latent topics.

        Returns:
            Optional[Tuple[pd.DataFrame, pd.DataFrame, List[List]]]:
                A tuple containing:
                - final_graph_df (pd.DataFrame): DataFrame representing the final topic graph after merging.
                - final_flatten_df (pd.DataFrame): Flattened DataFrame of the final topic graph.
                - _topics (List[List]): List of final topics, where each topic is a list of keyphrases.
                Returns None if latent topics have not been extracted yet or if an error occurs.
        """

        print("> Merging subtopics using embeddings...")
        if (
            self._latent_graph_df is None
            or self._latent_graph_df.shape[0] == 0  # type: ignore
            or self._flatten_latent_df is None
        ):
            raise Exception("Please extract latent topics first")
        try:
            final_graph_df, final_flatten_df, _topics = get_merged_topics(
                self._latent_graph_df,  # type: ignore
                self._embedding_model,
                th,
                self._pretty_print,
            )
            print(f"Final {len(_topics)} topics generated\n")
            return final_graph_df, final_flatten_df, _topics

        except Exception as e:
            print(e)

    def get_topics(self) -> List[List]:
        """
        Retrieves the final topics extracted by the GraphTopic model.

        This method orchestrates the final topic extraction process by calling `__extract_final_topics`
        with the `overlapping_threshold` as the similarity threshold.

        It also handles the case where `max_topics` is specified, pruning the final topics to the desired number
        using the specified `keyphrases_selection_strategy`. If `max_topics` is 'inf', all extracted topics are returned.

        Returns:
            List[List]: List of final topics.
                          If `max_topics` is 'inf', each topic is a list of all keyphrases belonging to that topic.
                          If `max_topics` is a finite number, each topic is a list of top keyphrases (up to `max_keyphrases_per_topic`)
                          selected according to the `keyphrases_selection_strategy`.
        """
        (
            self._final_graph_df,
            self._final_flatten_df,
            self.topics,  # final topics are saved in this var.
        ) = self.__extract_final_topics(th=self._overlapping_threshold)  # type: ignore

        if self._max_topics == float("inf"):
            return [kps[: self._max_keyphrases_per_topic] for kps in self.topics]

        return get_topics_with_pruned_kps(
            flatten_df=self._final_flatten_df,
            embeddingModel=self._embedding_model,
            max_keyphrases_per_topic=self._max_keyphrases_per_topic,
            keyphrases_selection_strategy=self._keyphrases_selection_strategy,
            max_topics=self._max_topics,
        )  # type: ignore

    def get_latent_graph_df(self):
        """
        Returns the Pandas DataFrame representing the latent topic graph.

        This DataFrame contains information about the latent topics extracted before the final merging step.
        It can be used for further analysis and visualization of the intermediate topic structure.

        Returns:
            pd.DataFrame: DataFrame representing the latent topic graph.
                          The DataFrame structure typically includes columns for topic IDs, keyphrases, node counts, and potentially other graph-related metrics.
        """
        return self._latent_graph_df


# https://github.com/MaartenGr/BERTopic/issues/90
# https://radimrehurek.com/gensim/models/coherencemodel.html
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from typing import List, Dict
import numpy as np


def calculate_coherence_scores(
    topics: List[List[str]], docs: List[str], topn=20
) -> Dict[str, float]:
    # https://stackoverflow.com/questions/38115367/scikit-learn-dont-separate-hyphenated-words-while-tokenization
    token_pattern = "(?u)\\b[\\w-]+\\b"
    vectorizer_model = CountVectorizer(
        stop_words="english", ngram_range=(2, 3), token_pattern=token_pattern
    )
    # MEMORY FIX (OOM on long-document datasets such as TopicNet/NIPS):
    # 1. No vectorizer_model.fit(docs): build_analyzer() does not need a fitted
    #    vectorizer, and the fit built a full 2-3-gram vocabulary that was
    #    never used.
    # 2. No `tokens` list: it held every n-gram string of every document and
    #    stayed alive through all coherence measures, although `texts` is
    #    built from `corpus`, not from `tokens`. The dictionary is built from
    #    a generator and the corpus re-runs the analyzer per document.
    # 3. processes=1 in CoherenceModel: no worker processes copying the data.
    # 4. Each CoherenceModel is freed before the next measure.
    # Scores should be identical to the previous version; verify once on a
    # small dataset (see notes in the chat).
    import gc

    analyzer = vectorizer_model.build_analyzer()
    dictionary = Dictionary(analyzer(doc) for doc in docs)
    corpus = [dictionary.doc2bow(analyzer(doc)) for doc in docs]
    texts = [[dictionary[word_id] for word_id, freq in doc] for doc in corpus]
    gc.collect()

    # (Removed the unused `topic_words` block: it was computed but never
    # passed to CoherenceModel, which always received `topics`.)

    coherence_scores = {}
    for measure in ["c_v", "u_mass", "c_uci", "c_npmi"]:
        coherence_model = CoherenceModel(
            topics=topics,
            texts=texts,
            corpus=corpus,
            dictionary=dictionary,
            coherence=measure,
            topn=topn,
            processes=1,
        )
        # coherence_score = coherence_model.get_coherence()
        s = coherence_model.get_coherence_per_topic()
        coherence_scores[measure] = np.mean(s)
        del coherence_model
        gc.collect()
        # nan_values = np.isnan(s)
        # nan_values = np.sum(nan_values)
        # https://github.com/piskvorky/gensim/issues/3040
        # s = np.nan_to_num(s, nan=np.nanmean(s))
        # print(s)
        # Count NaN values
        # coherence_scores[f"{measure}_nan"] = nan_values

    # print(coherence_scores)
    return coherence_scores


def get_avg_intra_score(model, topics, debug=False):
    def calculate_cosine_similarity(texts):
        # NumPy output (not a CUDA tensor) so sklearn can consume it;
        # float32 because the model may be loaded in half precision.
        embeddings = np.asarray(model.encode(texts), dtype=np.float32)
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix

    avg_intra_similarities = []

    for topic in topics:
        similarity_matrix = calculate_cosine_similarity(topic)
        intra_similarity = np.mean(similarity_matrix)
        avg_intra_similarities.append(intra_similarity)

    overall_avg_intra_similarity = np.mean(avg_intra_similarities)
    if debug:
        print("Average Intra-Similarities for Each Topic:")
        for i, topic in enumerate(topics):
            print(f"Topic {i + 1}: {avg_intra_similarities[i]:.4f}")

        print(f"Overall Average Intra-Similarity: {overall_avg_intra_similarity:.4f}\n")

    return overall_avg_intra_similarity


def get_avg_inter_score(model, topics, debug=False):
    if len(topics) < 2:
        return 0.0

    # Function to calculate cosine similarity
    def calculate_inter_distances(texts1, texts2):
        embeddings1 = np.asarray(
            model.encode(texts1), dtype=np.float32
        )  # Shape:[Topic_Size1,Embedding_Dim]
        embeddings2 = np.asarray(
            model.encode(texts2), dtype=np.float32
        )  # Shape:[Topic_Size2,Embedding_Dim]
        similarity_matrix = cosine_similarity(
            embeddings1, embeddings2
        )  # Shape: [Topic_Size1,Topic_Size2]
        inter_distances = 1 - similarity_matrix  # Calculate inter-distances
        return inter_distances

    inter_distances = []

    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            inter_distance_matrix = calculate_inter_distances(topics[i], topics[j])
            inter_distance = np.mean(inter_distance_matrix)
            inter_distances.append(inter_distance)

    overall_avg_inter_distance = np.mean(inter_distances)

    # print(inter_distances)
    if debug:
        print("Inter-Distances between Topics:")
        pair_count = 0
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                print(f"Topics {i + 1} and {j + 1}: {inter_distances[pair_count]:.4f}")
                pair_count += 1

        print(f"Overall Average Inter-Distance: {overall_avg_inter_distance:.4f}\n")

    return overall_avg_inter_distance


def get_semantic_score(embeddingModel, topics_list, debug=False):
    intra_avg = get_avg_intra_score(embeddingModel, topics_list, debug)
    inter_avg = get_avg_inter_score(embeddingModel, topics_list, debug)
    coherence = np.array([intra_avg, inter_avg]).mean()

    return {
        "intra_similarity_avg": intra_avg,  # Higher Better
        "inter_distances_avg": inter_avg,  # Higher Better
        "coherence": coherence,  # Coherence
    }


def calculate_topic_diversity(topics_list) -> float:
    total_unique_words = set()
    total_words = []
    for topic_words in topics_list:
        total_words.extend(topic_words)
        total_unique_words.update(topic_words)

    total_unique_words_should_be = len(total_words)
    total_unique_words_found = len(total_unique_words)
    diversity_score = total_unique_words_found / total_unique_words_should_be
    return {"topic_diversity": diversity_score}


def prepare_v1_docs(docs):
    """v1 test-runner preprocess: noun phrases, then deep clean."""
    docs = [extract_noun_phrases(doc) for doc in docs]
    return _preprocess_text(docs, deep_clean=True)


def clean_text(text: str) -> str:
    """Shell-only cleaner for vocab chunking. Model input uses prepare_v1_docs."""
    return str(text).strip()


def clean_docs(docs: List[str]) -> List[str]:
    return [clean_text(d) for d in docs if clean_text(d)]


def calculate_v1_metrics(topics, docs, embedding_model, topn=10):
    """v1 scores/main.py metrics. topn=10 matches test-runner.py."""
    coherence = calculate_coherence_scores(topics, docs, topn=topn)
    semantic = get_semantic_score(embedding_model, topics)
    diversity = calculate_topic_diversity(topics)
    return {
        "c_v": coherence.get("c_v"),
        "u_mass": coherence.get("u_mass"),
        "c_uci": coherence.get("c_uci"),
        "c_npmi": coherence.get("c_npmi"),
        "intra_similarity_avg": semantic.get("intra_similarity_avg"),
        "inter_distances_avg": semantic.get("inter_distances_avg"),
        "semantic_coherence": semantic.get("coherence"),
        "topic_diversity": diversity.get("topic_diversity"),
    }


## GRAPHTOPIC END

# ============================================================================
# ── PAPER STYLE HELPER  ──────────────────────────────────────────────────────
# ============================================================================

PAPER_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#111111",
    "axes.titlecolor": "#000000",
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "text.color": "#111111",
    "legend.facecolor": "white",
    "legend.edgecolor": "#AAAAAA",
    "legend.labelcolor": "#111111",
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}

METRIC_LABELS = {
    "c_npmi": "C-NPMI",
    "c_v": "C-V",
    "u_mass": "U-Mass",
    "c_uci": "C-UCI",
    "topic_diversity": "Topic Diversity",
    "semantic_coherence": "Sem. Coherence",
    "intra_similarity_avg": "Intra-Sim.",
    "inter_distances_avg": "Inter-Dist.",
}


def _mlabel(m: str) -> str:
    return METRIC_LABELS.get(m, m.replace("_", " ").title())


def _paper_style() -> None:
    plt.rcParams.update(PAPER_RC)


# ============================================================================
# ── CONFIGURATION  ───────────────────────────────────────────────────────────
# ============================================================================


class ComparisonConfig:
    """Mirrors the original Config so imports / helpers work unchanged."""

    OUTPUT_DIR = LOCAL_OUTPUT_DIR

    DATASETS = [
        {"name": "SetFit/20_newsgroups", "text_key": "text"},
        {"name": "SetFit/bbc-news", "text_key": "text"},
        {
            "name": "TrumpTweet",
            "path": str(TWEETS_CSV_PATH),
            "text_key": "text",
        },
        {"name": "ronitHF/pubmed-10k", "text_key": "article"},
        {"name": "TopicNet/RTL-Wiki", "text_key": "raw_text"},
        {"name": "TopicNet/NIPS", "text_key": "raw_text"},
        {"name": "SetFit/imdb", "text_key": "text"},
    ]

    GENSIM_MODELS = DEFAULT_GENSIM_MODELS
    TOPMOST_MODELS = DEFAULT_TOPMOST_MODELS
    BASELINE_MODELS = DEFAULT_BASELINE_MODELS
    BASELINE_NUM_TOPICS = 10  # v1 test-runner.py max_topics
    BASELINE_NUM_WORDS = 10
    BASELINE_EPOCHS = 100
    BASELINE_MAX_FEATURES = 3_000

    PRIMARY_METRIC = "c_npmi"
    REPORT_METRICS = [
        "c_npmi",
        "c_v",
        "u_mass",
        "c_uci",
        "intra_similarity_avg",
        "inter_distances_avg",
        "semantic_coherence",
    ]

    ENABLE_HYPERPARAM_OPTIMIZATION = ENABLE_HYPERPARAM_OPTIMIZATION
    N_TRIALS = HYPERPARAM_N_TRIALS
    N_STARTUP_TRIALS = HYPERPARAM_N_STARTUP_TRIALS
    ENABLE_EARLY_STOPPING = HYPERPARAM_EARLY_STOPPING
    EARLY_STOPPING_PATIENCE = HYPERPARAM_EARLY_STOPPING_PATIENCE
    USE_MULTI_OBJECTIVE = HYPERPARAM_USE_MULTI_OBJECTIVE
    MULTI_OBJECTIVE_METRICS = HYPERPARAM_MULTI_OBJECTIVE_METRICS
    MULTI_OBJECTIVE_WEIGHTS = HYPERPARAM_MULTI_OBJECTIVE_WEIGHTS
    OPTIMIZE_HYPERPARAMS = OPTIMIZE_HYPERPARAMS

    # ── Searched hyperparameters ──────────────────────────────────────────
    PARAM_SPACE = PARAM_SPACE

    FIXED_PARAMS: Dict = {
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "ngram_range": (2, 3),  # fixed; not searched
        "max_topics": 10,
        "score_topn": 10,  # v1 test-runner coherence topn
    }

    SEED = 42


config = ComparisonConfig()


# ============================================================================
# ── UTILITIES  ───────────────────────────────────────────────────────────────
# ============================================================================


def _val(v):
    return v.value if hasattr(v, "value") else v


def _load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pkl(path: Path, obj) -> None:
    path.write_bytes(pickle.dumps(obj))


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _source_hash(*objects) -> str:
    payload = "\n\n".join(inspect.getsource(obj) for obj in objects)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_comparison_state(
    *,
    dataset_name: str,
    doc_chunk_size: int,
    n_equal_chunks: int,
    cfg: ComparisonConfig,
    smoke: bool = False,
    smoke_max_chunks: Optional[int] = None,
) -> dict:
    """
    JSON-serializable cache state. Source hashes make cleaning, chunking, and
    aggregation changes invalidate cache automatically.
    """
    return {
        "params": {
            "dataset_name": dataset_name,
            "doc_chunk_size": int(doc_chunk_size),
            "n_equal_chunks": int(n_equal_chunks),
            "fixed_doc_chunk_size": dict(FIXED_DOC_CHUNK_SIZE),
            "smoke": bool(smoke),
            "smoke_max_chunks": (
                None if smoke_max_chunks is None else int(smoke_max_chunks)
            ),
            "report_metrics": list(cfg.REPORT_METRICS),
            "baseline_models": list(cfg.BASELINE_MODELS),
            "baseline_num_topics": int(cfg.BASELINE_NUM_TOPICS),
            "baseline_num_words": int(cfg.BASELINE_NUM_WORDS),
            "baseline_max_features": int(cfg.BASELINE_MAX_FEATURES),
            "baseline_epochs": int(cfg.BASELINE_EPOCHS),
            "ngram_range": list(cfg.FIXED_PARAMS["ngram_range"]),
            "hyperparam_optimization_enabled": bool(cfg.ENABLE_HYPERPARAM_OPTIMIZATION),
            "hyperparam_n_trials": int(cfg.N_TRIALS),
            "hyperparam_n_startup_trials": int(cfg.N_STARTUP_TRIALS),
            "hyperparam_use_multi_objective": bool(cfg.USE_MULTI_OBJECTIVE),
            "hyperparam_multi_objective_metrics": list(cfg.MULTI_OBJECTIVE_METRICS),
            "hyperparam_multi_objective_weights": list(cfg.MULTI_OBJECTIVE_WEIGHTS),
            "optimize_hyperparams": dict(cfg.OPTIMIZE_HYPERPARAMS),
            "param_space": cfg.PARAM_SPACE,
        },
        "source_hashes": {
            "cleaning": _source_hash(
                extract_noun_phrases, _preprocess_text, prepare_v1_docs
            ),
            "chunking": _source_hash(split_into_chunks, make_fixed_doc_chunks),
            "aggregation": _source_hash(aggregate_chunk_results),
            "model_entrypoints": _source_hash(
                run_graphtopic_on_chunk,
                run_hyperparam_optimization,
                run_baseline_on_chunk,
            ),
        },
    }


def apply_smoke_overrides(cfg: ComparisonConfig, smoke: bool) -> None:
    """Mutate cfg + ACTIVE_COMPARISON_DIR for smoke runs. No-op when smoke=False."""
    global ACTIVE_COMPARISON_DIR, SMOKE_TEST
    SMOKE_TEST = bool(smoke)
    if not smoke:
        ACTIVE_COMPARISON_DIR = FULL_COMPARISON_DIR
        return
    ACTIVE_COMPARISON_DIR = FULL_COMPARISON_SMOKE_DIR
    ACTIVE_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    cfg.N_TRIALS = SMOKE_N_TRIALS
    cfg.N_STARTUP_TRIALS = SMOKE_N_STARTUP_TRIALS
    cfg.EARLY_STOPPING_PATIENCE = SMOKE_EARLY_STOPPING_PATIENCE
    cfg.BASELINE_EPOCHS = SMOKE_BASELINE_EPOCHS


def comparison_state_hash(state: dict) -> str:
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _force_gc(label: str = "") -> None:
    import gc as _gc

    _gc.collect()
    _gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    time.sleep(0.2)


def _free_ram_mib() -> float:
    return psutil.virtual_memory().available / 1024 / 1024


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def optimal_batch_size(device: str) -> int:
    try:
        import torch

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return (
                4096
                if vram_gb >= 35
                else 2048
                if vram_gb >= 14
                else 1024
                if vram_gb >= 6
                else 512
            )
    except ImportError:
        pass
    return 512


def _load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    def _try_onnx(provider):
        try:
            import onnxruntime as ort

            if provider not in ort.get_available_providers():
                return None
            m = SentenceTransformer(
                model_name,
                device=device,
                backend="onnx",
                model_kwargs={"file_name": "onnx/model_O3.onnx", "provider": provider},
            )
            print(f"  ✓ ONNX O3 [{provider}] loaded")
            return m
        except Exception as exc:
            print(f"  ⚠  ONNX [{provider}] failed: {exc}")
            return None

    if device == "cuda":
        m = _try_onnx("CUDAExecutionProvider")
        if m:
            return m
        try:
            m = SentenceTransformer(model_name, device=device)
            m.half()
            print("  ✓ PyTorch fp16 CUDA loaded")
            return m
        except Exception as exc:
            print(f"  ⚠  CUDA failed: {exc}")

    m = _try_onnx("CPUExecutionProvider")
    if m:
        return m
    m = SentenceTransformer(model_name, device="cpu")
    print("  ✓ PyTorch CPU fallback loaded")
    return m


# ── Dataset loading ───────────────────────────────────────────────────────────


def load_dataset_docs(dataset_config: dict) -> List[str]:
    name = dataset_config.get("name")
    key = dataset_config.get("text_key")
    path = dataset_config.get("path")

    if path is not None and Path(path).exists():
        import csv

        with open(path, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            docs = [row[key].strip() for row in reader if row.get(key, "").strip()]
        print(f"  ✓ {len(docs):,} documents (from CSV: {path})")
        return docs

    print(f"  Loading {name}...")
    ds = load_dataset(name)
    docs = [
        str(item[key]).strip()
        for split in ds
        for item in ds[split]
        if key in item and item[key]
    ]
    docs = [d for d in docs if d]
    print(f"  ✓ {len(docs):,} documents")
    return docs


def compute_vocab_size(docs: List[str]) -> int:
    from sklearn.feature_extraction.text import CountVectorizer

    docs = clean_docs(docs)
    if not docs:
        return 0
    vec = CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        ngram_range=(1, 1),
        max_features=None,
        min_df=1,
    )
    try:
        vec.fit(docs)
        return len(vec.vocabulary_)
    except ValueError:
        return 0


# ============================================================================
# ── VOCAB-CALIBRATED CHUNKING  ───────────────────────────────────────────────
# ============================================================================


def make_contiguous_chunks(docs: List[str], n_chunks: int) -> List[List[str]]:
    """Split docs into n contiguous, near-equal chunks without shuffling."""
    if not docs:
        return []

    n_chunks = max(1, min(n_chunks, len(docs)))
    base, extra = divmod(len(docs), n_chunks)
    chunks: List[List[str]] = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        stop = start + size
        chunks.append(docs[start:stop])
        start = stop
    return chunks


def make_fixed_doc_chunks(
    docs: List[str],
    dataset_name: str,
    n_chunks: int = N_EQUAL_CHUNKS,
    doc_chunk_size: Optional[int] = None,
    max_chunks: Optional[int] = None,
) -> Tuple[List[List[str]], int]:
    """
    v1-style contiguous document chunks.

    Classic datasets use exact prepare.py sizes (3768 / 2225 / 11314).
    Others use equal 5-way splits. Remainder docs that do not fill a chunk
    are dropped (same as v1).

    Smoke overrides: pass ``doc_chunk_size`` (all datasets) and optional
    ``max_chunks`` to keep only the first N chunks. Production callers leave
    both as None so FIXED_DOC_CHUNK_SIZE / equal-n logic is unchanged.
    """
    if not docs:
        return [], 0

    if doc_chunk_size is not None:
        chunk_size = int(doc_chunk_size)
    elif dataset_name in FIXED_DOC_CHUNK_SIZE:
        chunk_size = int(FIXED_DOC_CHUNK_SIZE[dataset_name])
    else:
        chunk_size = len(docs) // max(1, n_chunks)

    if chunk_size < 1:
        return [list(docs)], len(docs)

    usable = docs[: (len(docs) // chunk_size) * chunk_size]
    if not usable:
        # Smoke / tiny corpora: allow a single short chunk instead of empty.
        if doc_chunk_size is not None:
            chunk = list(docs[:chunk_size])
            return [chunk], len(chunk)
        return [], chunk_size
    chunks = split_into_chunks(usable, chunk_size=chunk_size, overlap_ratio=0)
    if max_chunks is not None:
        chunks = chunks[: max(0, int(max_chunks))]
    return chunks, chunk_size


def make_vocab_calibrated_chunks(
    docs: List[str],
    vocab_chunk_size: int = 35_000,
    initial_chunks: int = 1,
) -> Tuple[List[List[str]], List[int]]:
    """Kept for reference; v1 comparison uses make_fixed_doc_chunks instead."""
    if not docs:
        return [], []

    n_chunks = max(1, initial_chunks)
    while True:
        chunks = make_contiguous_chunks(docs, n_chunks)
        chunk_vocabs = [compute_vocab_size(chunk) for chunk in chunks]
        if max(chunk_vocabs, default=0) <= vocab_chunk_size or n_chunks >= len(docs):
            return chunks, chunk_vocabs
        n_chunks += 1


# ============================================================================
# ── OPTIMAL PARAM LOADER  ────────────────────────────────────────────────────
# ============================================================================


def load_optimal_params(
    output_dir: Path,
    dataset_name: str,
    cfg: ComparisonConfig,
) -> Optional[dict]:
    """
    Load the best GraphTopic params saved by the optimisation script.

    Priority:
      1. Any `gt_eval.pkl` checkpoint (most reliable — written after GT eval).
      2. `final_result.pkl` (complete result dict).

    Searches all opt_size folders for the given dataset and picks the one
    whose best objective value is highest.
    """
    safe_name = dataset_name.replace("/", "_")
    best_params = None
    best_value = -np.inf

    for exp_dir in sorted(output_dir.glob(f"{safe_name}_opt*")):
        gt_eval_path = exp_dir / "gt_eval.pkl"
        final_path = exp_dir / "final_result.pkl"

        # Priority 1: gt_eval.pkl
        if gt_eval_path.exists():
            try:
                data = _load_pkl(gt_eval_path)
                val = data.get("value", -np.inf)
                if val > best_value:
                    best_value = val
                    best_params = data.get("best_params", {})
                    print(
                        f"  ↳ Loaded gt_eval from {exp_dir.name}  (objective={val:.4f})"
                    )
            except Exception as e:
                print(f"  ⚠  Could not load gt_eval.pkl from {exp_dir}: {e}")

        # Priority 2: final_result.pkl (only if gt_eval not found or worse)
        elif final_path.exists():
            try:
                data = _load_pkl(final_path)
                val = data.get("evaluation_value", -np.inf)
                if val > best_value:
                    best_value = val
                    best_params = data.get("best_params", {})
                    print(
                        f"  ↳ Loaded final_result from {exp_dir.name}  "
                        f"(eval_value={val:.4f})"
                    )
            except Exception as e:
                print(f"  ⚠  Could not load final_result.pkl from {exp_dir}: {e}")

    if best_params is None:
        print(
            f"  ⚠  No saved optimal params found for '{dataset_name}'. "
            "Using default/fixed params."
        )
        # Sensible fallback: use midpoints of the search space
        best_params = {
            "TopM": 500,
            "TopN": 10,
            "overlapping_threshold": 0.5,
        }

    # Merge fixed params
    merged = {k: _val(v) for k, v in cfg.FIXED_PARAMS.items()}
    merged.update({k: _val(v) for k, v in best_params.items()})
    return merged


# ============================================================================
# ── GRAPHTOPIC HYPERPARAMETER OPTIMIZATION  ─────────────────────────────────
# ============================================================================


def _active_hyperparams(cfg: ComparisonConfig) -> List[str]:
    unknown = set(cfg.OPTIMIZE_HYPERPARAMS) - set(cfg.PARAM_SPACE)
    if unknown:
        raise ValueError(f"Unknown configured hyperparameters: {unknown}")
    return [
        name
        for name in cfg.PARAM_SPACE
        if bool(cfg.OPTIMIZE_HYPERPARAMS.get(name, False))
    ]


def _sample_hyperparams(
    trial: optuna.Trial,
    base_params: dict,
    cfg: ComparisonConfig,
) -> dict:
    params = dict(base_params)
    for name in _active_hyperparams(cfg):
        spec = cfg.PARAM_SPACE[name]
        if spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported parameter type for {name}: {spec['type']}")
    return params


class HyperparamEarlyStopping:
    def __init__(
        self,
        patience: int,
        min_delta: float = 0.001,
        study: Optional[optuna.Study] = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.counter = 0
        if study:
            for trial in study.trials:
                value = trial.value
                if value is None or np.isnan(value):
                    continue
                if value > self.best + self.min_delta:
                    self.best = value
                    self.counter = 0
                else:
                    self.counter += 1

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        if trial.value is None:
            return
        if trial.value > self.best + self.min_delta:
            self.best = trial.value
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            print(f"    Early stop: no improvement for {self.patience} trials")
            study.stop()


def run_hyperparam_optimization(
    chunk_docs: List[str],
    base_params: dict,
    emb_model: SentenceTransformer,
    cfg: ComparisonConfig,
    dataset_name: str,
    chunk_idx: int,
) -> Tuple[dict, dict]:
    """Optimize enabled parameters for one chunk; keep all others at defaults."""
    params = dict(base_params)
    active_params = _active_hyperparams(cfg)
    metadata = {
        "hyperparam_optimization_enabled": bool(
            cfg.ENABLE_HYPERPARAM_OPTIMIZATION and active_params
        ),
        "optimized_hyperparameters": ",".join(active_params),
        "hyperparam_n_trials": 0,
        "hyperparam_best_value": float("nan"),
    }
    if not cfg.ENABLE_HYPERPARAM_OPTIMIZATION or not active_params:
        return params, metadata

    safe_name = dataset_name.replace("/", "_")
    optimization_signature = hashlib.sha256(
        json.dumps(
            {
                "active_params": active_params,
                "param_space": {name: cfg.PARAM_SPACE[name] for name in active_params},
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    output_dir = (
        ACTIVE_COMPARISON_DIR
        / "hyperparameter_optimization"
        / safe_name
        / f"chunk_{chunk_idx + 1}_{optimization_signature}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{output_dir / 'optuna.db'}"
    study_name = f"{safe_name}_chunk_{chunk_idx + 1}_{optimization_signature}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(
            n_startup_trials=cfg.N_STARTUP_TRIALS,
            seed=cfg.SEED,
            multivariate=True,
        ),
        storage=storage,
        load_if_exists=True,
    )
    # Trials left in RUNNING state by a killed process never finished; do not
    # count them, otherwise every crash silently eats one of the N_TRIALS.
    finished_trials = [
        t for t in study.trials if t.state != optuna.trial.TrialState.RUNNING
    ]
    n_remaining = max(0, cfg.N_TRIALS - len(finished_trials))
    trials_csv = output_dir / "trials.csv"
    print(
        f"    Optimizing {active_params}: "
        f"{len(finished_trials)}/{cfg.N_TRIALS} trials complete"
    )

    def objective(trial: optuna.Trial) -> float:
        trial_params = _sample_hyperparams(trial, params, cfg)
        result = run_graphtopic_on_chunk(chunk_docs, trial_params, emb_model, cfg)
        value = result.get(cfg.PRIMARY_METRIC, float("-inf"))
        optimization_value = (
            sum(
                weight * result.get(metric, 0.0)
                for weight, metric in zip(
                    cfg.MULTI_OBJECTIVE_WEIGHTS,
                    cfg.MULTI_OBJECTIVE_METRICS,
                )
            )
            if cfg.USE_MULTI_OBJECTIVE
            else value
        )
        if optimization_value is None or np.isnan(float(optimization_value)):
            optimization_value = float("-inf")
        result["multi_objective"] = optimization_value
        pd.DataFrame(
            [
                {
                    "trial": trial.number,
                    "dataset": dataset_name,
                    "chunk_idx": chunk_idx,
                    **{k: _val(trial_params[k]) for k in cfg.PARAM_SPACE},
                    **{k: _val(v) for k, v in result.items()},
                    "timestamp": datetime.now().isoformat(),
                }
            ]
        ).to_csv(
            trials_csv,
            mode="a",
            header=not trials_csv.exists(),
            index=False,
        )
        return float(optimization_value)

    callbacks = (
        [HyperparamEarlyStopping(cfg.EARLY_STOPPING_PATIENCE, study=study)]
        if cfg.ENABLE_EARLY_STOPPING
        else []
    )
    if n_remaining > 0:
        try:
            study.optimize(
                objective,
                n_trials=n_remaining,
                callbacks=callbacks,
                show_progress_bar=True,
            )
        except KeyboardInterrupt:
            print("\n    Optimization interrupted; completed trials were saved")

    completed = [trial for trial in study.trials if trial.value is not None]
    if completed:
        params.update({k: _val(v) for k, v in study.best_params.items()})
        metadata["hyperparam_n_trials"] = len(study.trials)
        metadata["hyperparam_best_value"] = float(study.best_value)
    return params, metadata


def _hyperparam_result_fields(
    params: dict,
    metadata: dict,
    cfg: ComparisonConfig,
) -> dict:
    return {
        **metadata,
        **{f"graphtopic_{name}": _val(params.get(name)) for name in cfg.PARAM_SPACE},
    }


# ============================================================================
# ── GRAPHTOPIC RUNNER  ───────────────────────────────────────────────────────
# ============================================================================


def run_graphtopic_on_chunk(
    chunk_docs: List[str],
    params: dict,
    emb_model: SentenceTransformer,
    cfg: ComparisonConfig,
) -> dict:
    """Run v1 GraphTopic on noun-phrase docs; score with v1 metrics."""
    global embeddingModel
    embeddingModel = emb_model

    chunk_docs = prepare_v1_docs(chunk_docs)
    prof_start = time.time()
    peak_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    try:
        gt = GraphTopic(
            docs=chunk_docs,
            embedding_model=emb_model,
            max_topics=int(params.get("max_topics", cfg.FIXED_PARAMS["max_topics"])),
            TopM=int(params.get("TopM", 500)),
            TopN=int(params.get("TopN", 10)),
            overlapping_threshold=float(params.get("overlapping_threshold", 0.5)),
        )
        topics = gt.get_topics()
        del gt
        _force_gc("GraphTopic-chunk")

        if not topics:
            return {
                "status": "no_topics",
                cfg.PRIMARY_METRIC: float("-inf"),
                "num_topics": 0,
            }

        scores = calculate_v1_metrics(
            topics,
            chunk_docs,
            emb_model,
            topn=int(cfg.FIXED_PARAMS["score_topn"]),
        )
        scores["num_topics"] = len(topics)
        scores["status"] = "success"
        scores["runtime_sec"] = time.time() - prof_start
        peak_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        scores["peak_memory_mb"] = max(0.0, peak_after - peak_before)
        return scores

    except MemoryError as exc:
        _force_gc("GT-OOM")
        return {
            "status": "error_oom",
            "error": str(exc),
            cfg.PRIMARY_METRIC: float("-inf"),
        }
    except Exception as exc:
        traceback.print_exc()
        return {"status": "error", "error": str(exc), cfg.PRIMARY_METRIC: float("-inf")}


def run_baseline_on_chunk(
    name: str,
    chunk_docs: List[str],
    emb_model: SentenceTransformer,
    cfg: ComparisonConfig,
) -> dict:
    """Run a single baseline model on v1-preprocessed docs; score with v1 metrics."""
    from baseline import (
        get_bertopic_topics,
        get_hdp_topics,
        get_lda_topics,
        get_lsi_topics,
        get_nmf_topics,
        get_topmost_topics,
        preprocess_gensim,
    )

    chunk_docs = prepare_v1_docs(chunk_docs)
    t0 = time.time()
    peak_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    topics = None
    try:
        if name in cfg.GENSIM_MODELS:
            corpus, dictionary = preprocess_gensim(
                chunk_docs,
                ngram_range=cfg.FIXED_PARAMS["ngram_range"],
                max_features=cfg.BASELINE_MAX_FEATURES,
            )
            gensim_runners = {
                "LDA": get_lda_topics,
                "NMF": get_nmf_topics,
                "HDP": get_hdp_topics,
                "LSI": get_lsi_topics,
            }
            topics = gensim_runners[name](
                corpus,
                dictionary,
                num_topics=cfg.BASELINE_NUM_TOPICS,
                num_words=cfg.BASELINE_NUM_WORDS,
            )
        elif name == "BERTopic":
            topics = get_bertopic_topics(
                chunk_docs,
                num_topics=cfg.BASELINE_NUM_TOPICS,
                num_words=cfg.BASELINE_NUM_WORDS,
                ngram_range=cfg.FIXED_PARAMS["ngram_range"],
                max_features=cfg.BASELINE_MAX_FEATURES,
            )
        elif name in cfg.TOPMOST_MODELS:
            topics = get_topmost_topics(
                chunk_docs,
                model_type=name,
                num_topics=cfg.BASELINE_NUM_TOPICS,
                num_words=cfg.BASELINE_NUM_WORDS,
                ngram_range=cfg.FIXED_PARAMS["ngram_range"],
                dataset_batch_size=64,
                trainer_batch_size=64,
                epochs=cfg.BASELINE_EPOCHS,
                verbose=False,
                max_features=cfg.BASELINE_MAX_FEATURES,
            )
        else:
            raise ValueError(f"Unknown baseline model: {name}")

        if not topics:
            return {
                "status": "no_topics",
                cfg.PRIMARY_METRIC: float("-inf"),
                "num_topics": 0,
                "runtime_sec": time.time() - t0,
            }

        scores = calculate_v1_metrics(
            topics,
            chunk_docs,
            emb_model,
            topn=int(cfg.FIXED_PARAMS["score_topn"]),
        )
        scores["num_topics"] = len(topics)
        scores["status"] = "success"
        scores["runtime_sec"] = time.time() - t0
        peak_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        scores["peak_memory_mb"] = max(0.0, peak_after - peak_before)
        return scores

    except MemoryError as exc:
        return {
            "status": "error_oom",
            "error": str(exc),
            cfg.PRIMARY_METRIC: float("-inf"),
        }
    except Exception as exc:
        traceback.print_exc()
        return {"status": "error", "error": str(exc), cfg.PRIMARY_METRIC: float("-inf")}
    finally:
        del topics
        _force_gc(f"{name}-chunk")


# ============================================================================
# ── AVERAGING CHUNKS  ────────────────────────────────────────────────────────
# ============================================================================


def aggregate_chunk_results(
    chunk_results: List[dict],
    metrics: List[str],
) -> dict:
    """
    Average metric scores across chunks.
    Skips failed chunks (status != 'success') for metric averaging but
    reports how many succeeded. The primary metric columns keep the simple
    unweighted chunk mean; weighted variants are also reported with the
    ``*_weighted`` suffix using chunk_docs as weights.
    """
    good = [r for r in chunk_results if r.get("status") == "success"]
    n_good = len(good)
    n_total = len(chunk_results)
    total_weighted_docs = int(sum(int(r.get("chunk_docs", 0) or 0) for r in good))
    agg = {
        "n_chunks": n_total,
        "n_good_chunks": n_good,
        "weighted_docs": total_weighted_docs,
    }

    if not good:
        for m in metrics:
            agg[m] = float("nan")
            agg[f"{m}_std"] = float("nan")
            agg[f"{m}_weighted"] = float("nan")
            agg[f"{m}_weighted_std"] = float("nan")
        agg["status"] = "all_failed"
        return agg

    for m in metrics:
        vals_and_weights = [
            (r.get(m, float("nan")), int(r.get("chunk_docs", 0) or 0)) for r in good
        ]
        finite = [v for v, _ in vals_and_weights if np.isfinite(v)]
        agg[m] = float(np.mean(finite)) if finite else float("nan")
        agg[f"{m}_std"] = float(np.std(finite)) if finite else float("nan")

        weighted = [(v, w) for v, w in vals_and_weights if np.isfinite(v) and w > 0]
        if weighted:
            vals = np.array([v for v, _ in weighted], dtype=float)
            weights = np.array([w for _, w in weighted], dtype=float)
            weighted_mean = float(np.average(vals, weights=weights))
            weighted_var = float(
                np.average((vals - weighted_mean) ** 2, weights=weights)
            )
            agg[f"{m}_weighted"] = weighted_mean
            agg[f"{m}_weighted_std"] = float(np.sqrt(weighted_var))
        else:
            agg[f"{m}_weighted"] = float("nan")
            agg[f"{m}_weighted_std"] = float("nan")

    for extra in ("runtime_sec", "peak_memory_mb", "num_topics"):
        vals = [r.get(extra, float("nan")) for r in good]
        finite = [v for v in vals if np.isfinite(v)]
        agg[extra] = float(np.mean(finite)) if finite else float("nan")

    agg["status"] = "success"
    return agg


# ============================================================================
# ── DATASET-LEVEL RUNNER  ────────────────────────────────────────────────────
# ============================================================================


def run_full_dataset_comparison(
    dataset_cfg: dict,
    emb_model: SentenceTransformer,
    cfg: ComparisonConfig,
    force_rerun: bool = False,
    n_equal_chunks: int = N_EQUAL_CHUNKS,
    smoke: bool = False,
) -> pd.DataFrame:
    """
    End-to-end comparison on the full dataset for one dataset config.
    Returns a DataFrame with one row per (model, chunk_idx) plus averaged rows.
    """
    dataset_name = dataset_cfg["name"]
    safe_name = dataset_name.replace("/", "_")
    out_file = ACTIVE_COMPARISON_DIR / f"{safe_name}_full_comparison.csv"
    cache_pkl = ACTIVE_COMPARISON_DIR / f"{safe_name}_raw_results.pkl"
    state_file = ACTIVE_COMPARISON_DIR / f"{safe_name}_comparison_state.json"

    print(f"\n{'#' * 70}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'#' * 70}")

    # ── Load full corpus first (chunk size may depend on corpus length) ───
    all_docs = load_dataset_docs(dataset_cfg)
    all_clean = clean_docs(all_docs)
    doc_token_lens = [len(d.split()) for d in all_clean]
    dataset_stats = {
        "n_docs": len(all_clean),
        "avg_doc_length": float(np.mean(doc_token_lens)) if doc_token_lens else 0.0,
        "median_doc_length": float(np.median(doc_token_lens))
        if doc_token_lens
        else 0.0,
        "vocab_size": int(compute_vocab_size(all_clean)),
    }
    print(
        f"\n  Corpus stats: n={dataset_stats['n_docs']:,}  "
        f"avg_len={dataset_stats['avg_doc_length']:.1f}  "
        f"median_len={dataset_stats['median_doc_length']:.1f}  "
        f"vocab={dataset_stats['vocab_size']:,}"
    )

    # ── Fixed document-size chunks (v1 prepare.py) / smoke overrides ──────
    print(f"\n  Corpus: {len(all_clean):,} docs")
    smoke_chunk_kw = {}
    if smoke:
        smoke_chunk_kw = {
            "doc_chunk_size": SMOKE_DOC_CHUNK_SIZE,
            "max_chunks": SMOKE_MAX_CHUNKS,
        }
    chunks, doc_chunk_size = make_fixed_doc_chunks(
        all_clean, dataset_name, n_chunks=n_equal_chunks, **smoke_chunk_kw
    )
    chunk_vocabs = [compute_vocab_size(c) for c in chunks]
    comparison_state = build_comparison_state(
        dataset_name=dataset_name,
        doc_chunk_size=doc_chunk_size,
        n_equal_chunks=n_equal_chunks,
        cfg=cfg,
        smoke=smoke,
        smoke_max_chunks=SMOKE_MAX_CHUNKS if smoke else None,
    )
    state_hash = comparison_state_hash(comparison_state)

    # ── Load cache if available ───────────────────────────────────────────
    if not force_rerun and cache_pkl.exists():
        print(f"  ✓ Loaded cached results from {cache_pkl}")
        raw_cache = _load_pkl(cache_pkl)
        if raw_cache.get("_comparison_state") != comparison_state:
            print("  ↻ Cached comparison state differs; recomputing chunk results")
            raw_cache = {"_comparison_state": comparison_state}
    else:
        raw_cache: Dict[str, List[dict]] = {"_comparison_state": comparison_state}

    if smoke:
        print("  Chunk mode     : SMOKE (fixed small chunks)")
    else:
        fixed = dataset_name in FIXED_DOC_CHUNK_SIZE
        print(
            f"  Chunk mode     : "
            f"{'fixed prepare.py size' if fixed else f'equal {n_equal_chunks}-way'}"
        )
    print(f"  Doc chunk size : {doc_chunk_size:,}")
    print(f"  Num chunks     : {len(chunks)}")
    for i, (c, v) in enumerate(zip(chunks, chunk_vocabs)):
        print(f"    chunk {i + 1}: {len(c):,} docs  |  vocab: {v:,} tokens")

    # ── Load optimal GraphTopic params ────────────────────────────────────
    print(f"\n  Loading optimal GraphTopic parameters …")
    gt_params = load_optimal_params(cfg.OUTPUT_DIR, dataset_name, cfg)
    gt_params["batch_size"] = optimal_batch_size(detect_device())
    print("  Base params:")
    for k, v in sorted(gt_params.items()):
        print(f"    {k:<32s} = {v}")
    if cfg.ENABLE_HYPERPARAM_OPTIMIZATION:
        print(f"  Optimized per chunk: {_active_hyperparams(cfg)}")

    # ── Run GraphTopic + all default baselines ────────────────────────────
    model_names = ["GraphTopic"] + list(cfg.BASELINE_MODELS)
    rows: List[dict] = []

    for model_name in model_names:
        print(f"\n  {'─' * 60}")
        print(f"  MODEL: {model_name}")
        print(f"  {'─' * 60}")

        # Re-run a model if any cached chunk failed (status "error",
        # "error_oom" or "no_topics"), so a fixed bug is not locked in by the
        # cache. get_topmost_topics() catches its own exceptions and returns no
        # topics, which shows up here as "no_topics", not "error".
        cached_failed = model_name in raw_cache and any(
            str(r.get("status", "")).startswith("error")
            or r.get("status") == "no_topics"
            for r in raw_cache[model_name]
        )
        if cached_failed:
            print(f"  ↻ Cached results for {model_name} contain errors; re-running")
        if model_name in raw_cache and not cached_failed:
            print(f"  ✓ Using cached chunk results for {model_name}")
            chunk_results = raw_cache[model_name]
        else:
            chunk_results = []
            for ci, chunk in enumerate(chunks):
                print(
                    f"\n    Chunk {ci + 1}/{len(chunks)}  "
                    f"({len(chunk):,} docs | vocab≈{chunk_vocabs[ci]:,})"
                )
                _force_gc(f"pre-{model_name}-chunk{ci}")

                if model_name == "GraphTopic":
                    chunk_params, optimization_info = run_hyperparam_optimization(
                        chunk,
                        gt_params,
                        emb_model,
                        cfg,
                        dataset_name,
                        ci,
                    )
                    res = run_graphtopic_on_chunk(chunk, chunk_params, emb_model, cfg)
                    res.update(
                        _hyperparam_result_fields(chunk_params, optimization_info, cfg)
                    )
                else:
                    res = run_baseline_on_chunk(model_name, chunk, emb_model, cfg)

                res["chunk_idx"] = ci
                res["chunk_docs"] = len(chunk)
                res["chunk_vocab"] = chunk_vocabs[ci]
                chunk_results.append(res)

                status_str = res.get("status", "?")
                primary = res.get(cfg.PRIMARY_METRIC, float("nan"))
                print(
                    f"    status={status_str}  {cfg.PRIMARY_METRIC}={primary:.4f}"
                    if np.isfinite(primary)
                    else f"    status={status_str}"
                )

            raw_cache[model_name] = chunk_results
            _save_pkl(cache_pkl, raw_cache)  # save after each model

        # ── Per-chunk rows ────────────────────────────────────────────────
        for r in chunk_results:
            row = {
                "model": model_name,
                "dataset": dataset_name,
                "total_docs": len(all_clean),
                "avg_doc_length": dataset_stats["avg_doc_length"],
                "median_doc_length": dataset_stats["median_doc_length"],
                "vocab_size": dataset_stats["vocab_size"],
                "chunk_size": doc_chunk_size,
                "doc_chunk_size": doc_chunk_size,
                "n_equal_chunks": n_equal_chunks,
                "comparison_state_hash": state_hash,
                **r,
            }
            rows.append(row)

        # ── Aggregated row ────────────────────────────────────────────────
        agg = aggregate_chunk_results(chunk_results, cfg.REPORT_METRICS)
        agg_row = {
            "model": model_name,
            "dataset": dataset_name,
            "total_docs": len(all_clean),
            "avg_doc_length": dataset_stats["avg_doc_length"],
            "median_doc_length": dataset_stats["median_doc_length"],
            "vocab_size": dataset_stats["vocab_size"],
            "chunk_size": doc_chunk_size,
            "doc_chunk_size": doc_chunk_size,
            "n_equal_chunks": n_equal_chunks,
            "comparison_state_hash": state_hash,
            "chunk_idx": "AVERAGE",
            **agg,
        }
        if model_name == "GraphTopic":
            finite_best_values = [
                r["hyperparam_best_value"]
                for r in chunk_results
                if np.isfinite(r.get("hyperparam_best_value", float("nan")))
            ]
            agg_row.update(
                {
                    "hyperparam_optimization_enabled": bool(
                        cfg.ENABLE_HYPERPARAM_OPTIMIZATION and _active_hyperparams(cfg)
                    ),
                    "optimized_hyperparameters": ",".join(_active_hyperparams(cfg)),
                    "hyperparam_n_trials": sum(
                        int(r.get("hyperparam_n_trials", 0)) for r in chunk_results
                    ),
                    "hyperparam_best_value": (
                        float(np.mean(finite_best_values))
                        if finite_best_values
                        else float("nan")
                    ),
                    **{
                        f"graphtopic_{name}": json.dumps(
                            [r.get(f"graphtopic_{name}") for r in chunk_results]
                        )
                        for name in cfg.PARAM_SPACE
                    },
                }
            )
        rows.append(agg_row)
        print(f"\n  ── {model_name} AVERAGED RESULTS ──")
        for m in cfg.REPORT_METRICS:
            v = agg.get(m, float("nan"))
            s = agg.get(f"{m}_std", float("nan"))
            if np.isfinite(v):
                print(f"    {m:<30s} {v:8.4f} ± {s:.4f}  (chunk mean)")
            vw = agg.get(f"{m}_weighted", float("nan"))
            sw = agg.get(f"{m}_weighted_std", float("nan"))
            if np.isfinite(vw):
                print(
                    f"    {m + '_weighted':<30s} {vw:8.4f} ± {sw:.4f}  (doc-weighted)"
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False, float_format="%.5f")
    _save_json(state_file, comparison_state)
    print(f"\n  ✓ Saved per-chunk+averaged results → {out_file}")
    print(f"  ✓ Saved comparison state → {state_file}")
    return df


# ============================================================================
# ── GLOBAL SUMMARY BUILDER  ──────────────────────────────────────────────────
# ============================================================================


def build_global_summary(
    all_dfs: List[pd.DataFrame], cfg: ComparisonConfig
) -> pd.DataFrame:
    """Extract only the AVERAGE rows from all dataset DataFrames."""
    avg_rows = []
    for df in all_dfs:
        avg = df[df["chunk_idx"] == "AVERAGE"].copy()
        avg_rows.append(avg)
    summary = pd.concat(avg_rows, ignore_index=True)
    out = ACTIVE_COMPARISON_DIR / "global_comparison_summary.csv"
    summary.to_csv(out, index=False, float_format="%.5f")
    print(f"\n  ✓ Global summary → {out}")
    return summary


# ============================================================================
# ── PUBLICATION-QUALITY VISUALISATIONS  ──────────────────────────────────────
# ============================================================================
#
# Both plot functions are *reusable*:
#   • model_subset   – list of model names to include; None = all
#   • model_styles   – override the global DEFAULT_MODEL_STYLES registry
#   • save_path      – explicit output path, or auto-generated if None
#
# ─────────────────────────────────────────────────────────────────────────────


def _make_avg_df(
    df: pd.DataFrame, models: List[str], metrics: List[str]
) -> pd.DataFrame:
    """Extract average-row sub-DataFrame for the requested models & metrics."""
    avg = df[df["chunk_idx"] == "AVERAGE"].copy()
    avg = avg[avg["model"].isin(models)].reset_index(drop=True)
    return avg


# ── Spider / Radar plot ───────────────────────────────────────────────────────


def plot_radar(
    df: pd.DataFrame,
    metrics: List[str],
    *,
    model_subset: Optional[List[str]] = None,
    model_styles: Optional[Dict[str, Dict]] = None,
    title: str = "Model Comparison — Radar",
    save_path: Optional[Path] = None,
    std_shading: bool = True,
) -> Path:
    """
    Publication-quality spider / radar plot.

    Parameters
    ----------
    df           : DataFrame with at least columns [model, chunk_idx, *metrics].
                   Can be the per-dataset or global summary DataFrame.
    metrics      : Which metrics to show on the radar axes.
    model_subset : Names of models to include.  ``None`` → all models in df.
    model_styles : Override style registry (see DEFAULT_MODEL_STYLES).
    title        : Plot title.
    save_path    : Output PNG path.  Auto-generated if None.
    std_shading  : Fill ± std band for each model.
    """
    _paper_style()
    styles = model_styles or DEFAULT_MODEL_STYLES
    models = ordered_models(
        df[df["chunk_idx"] == "AVERAGE"]["model"].unique(),
        model_subset,
    )
    avg_df = _make_avg_df(df, models, metrics)

    valid_metrics = [m for m in metrics if m in avg_df.columns]
    if not valid_metrics:
        print("  ⚠  plot_radar: no valid metrics found — skipping")
        return None

    n_met = len(valid_metrics)
    angles = np.linspace(0, 2 * np.pi, n_met, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Normalise each metric to [0, 1] across the models being compared
    norm_df = avg_df[valid_metrics].copy()
    for m in valid_metrics:
        col = norm_df[m].replace([np.inf, -np.inf], np.nan)
        lo, hi = col.min(), col.max()
        norm_df[m] = (norm_df[m] - lo) / (hi - lo + 1e-12)

    # Also compute std-normalised band
    std_df = avg_df[
        [f"{m}_std" for m in valid_metrics if f"{m}_std" in avg_df.columns]
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")
    ax.spines["polar"].set_edgecolor("#AAAAAA")
    ax.grid(color="#DDDDDD", linewidth=0.7, linestyle="--")

    legend_handles = []
    for model in models:
        st = get_style(model, styles)
        row = norm_df[avg_df["model"] == model]
        if row.empty:
            continue
        vals = row[valid_metrics].values.flatten().tolist() + [
            row[valid_metrics[0]].values[0]
        ]
        ax.plot(
            angles,
            vals,
            color=st["color"],
            linewidth=st["linewidth"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            markersize=st["markersize"],
            zorder=4,
            label=st["label"],
        )
        ax.fill(angles, vals, color=st["color"], alpha=0.08)

        # ± std shading
        if std_shading:
            std_row = df[(df["chunk_idx"] == "AVERAGE") & (df["model"] == model)]
            raw_row = avg_df[avg_df["model"] == model][valid_metrics]
            if not std_row.empty:
                std_vals = []
                for m in valid_metrics:
                    s_col = f"{m}_std"
                    raw_v = float(raw_row[m].values[0])
                    # Convert std to normalised space (same scale used above)
                    c_lo = avg_df[m].min()
                    c_hi = avg_df[m].max()
                    s_v = float(std_row[s_col].values[0]) if s_col in std_row else 0.0
                    std_norm = s_v / (c_hi - c_lo + 1e-12)
                    std_vals.append(std_norm)
                std_vals += std_vals[:1]
                hi_band = [min(1, v + s) for v, s in zip(vals, std_vals)]
                lo_band = [max(0, v - s) for v, s in zip(vals, std_vals)]
                ax.fill_between(
                    angles,
                    lo_band,
                    hi_band,
                    color=st["color"],
                    alpha=0.06,
                    zorder=3,
                )

        h = plt.Line2D(
            [0],
            [0],
            color=st["color"],
            linewidth=st["linewidth"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            markersize=st["markersize"],
            label=st["label"],
        )
        legend_handles.append(h)

    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        [_mlabel(m) for m in valid_metrics],
        fontsize=12,
        fontweight="bold",
        color="#111111",
    )
    ax.set_rlabel_position(20)
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelsize=9, colors="#444444")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=22, color="#000000")

    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.40, 1.18),
        fontsize=11,
        frameon=True,
        framealpha=0.95,
        edgecolor="#AAAAAA",
        ncol=1,
        handlelength=2.5,
    )
    legend.get_frame().set_linewidth(0.8)

    # Annotation: normalised scale notice
    ax.text(
        0.5,
        -0.10,
        "Scores normalised to [0, 1] per metric across compared models.\n"
        "Shaded area = ±1 std across chunks.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
        style="italic",
    )

    if save_path is None:
        safe_title = title.replace(" ", "_").replace("/", "_")[:50]
        save_path = ACTIVE_COMPARISON_DIR / f"spider_{safe_title}.png"

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Radar saved → {save_path}")
    return save_path


# ── Grouped bar plot ──────────────────────────────────────────────────────────


def plot_bars(
    df: pd.DataFrame,
    metrics: List[str],
    *,
    model_subset: Optional[List[str]] = None,
    model_styles: Optional[Dict[str, Dict]] = None,
    title: str = "Model Comparison — Metrics",
    save_path: Optional[Path] = None,
    show_std: bool = True,
    annotate_best: bool = True,
) -> Path:
    """
    Publication-quality grouped bar chart.

    Parameters
    ----------
    df            : Same format as plot_radar.
    metrics       : Metrics to include as bar groups.
    model_subset  : Models to include; None → all.
    model_styles  : Style override registry.
    title         : Figure title.
    save_path     : Output PNG; auto-generated if None.
    show_std      : Draw error bars (±1 std across chunks).
    annotate_best : Highlight the winning bar per metric with a ★ annotation.
    """
    _paper_style()
    styles = model_styles or DEFAULT_MODEL_STYLES
    models = ordered_models(
        df[df["chunk_idx"] == "AVERAGE"]["model"].unique(),
        model_subset,
    )
    avg_df = _make_avg_df(df, models, metrics)

    valid_metrics = [m for m in metrics if m in avg_df.columns]
    if not valid_metrics:
        print("  ⚠  plot_bars: no valid metrics found — skipping")
        return None

    n_models = len(models)
    n_metrics = len(valid_metrics)
    bar_w = 0.70 / n_models
    x = np.arange(n_metrics)

    fig_w = max(12, n_metrics * 1.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Identify best model per metric for annotation
    best_model_per_metric: Dict[str, str] = {}
    if annotate_best:
        for m in valid_metrics:
            vals = {
                mod: float(avg_df[avg_df["model"] == mod][m].values[0])
                for mod in models
                if not avg_df[avg_df["model"] == mod].empty
            }
            finite_vals = {k: v for k, v in vals.items() if np.isfinite(v)}
            if finite_vals:
                best_model_per_metric[m] = max(finite_vals, key=finite_vals.get)

    for i, model in enumerate(models):
        st = get_style(model, styles)
        row = avg_df[avg_df["model"] == model]
        if row.empty:
            continue

        vals = [float(row[m].values[0]) for m in valid_metrics]
        stds = []
        for m in valid_metrics:
            s_col = f"{m}_std"
            stds.append(float(row[s_col].values[0]) if s_col in row.columns else 0.0)

        offset = (i - n_models / 2 + 0.5) * bar_w
        err_kw = dict(elinewidth=1.3, ecolor="#444444", capsize=3.5, capthick=1.2)

        bars = ax.bar(
            x + offset,
            vals,
            width=bar_w * 0.90,
            color=st["color"],
            alpha=0.82,
            label=st["label"],
            yerr=stds if show_std else None,
            error_kw=err_kw,
            hatch=st.get("hatch", ""),
            edgecolor="#333333",
            linewidth=0.6,
            zorder=3,
        )

        # Value annotations above each bar
        for j, (bar, v, s) in enumerate(zip(bars, vals, stds)):
            if not np.isfinite(v):
                continue
            metric_name = valid_metrics[j]
            is_best = best_model_per_metric.get(metric_name) == model

            y_top = bar.get_height() + (s if show_std else 0)
            ann_text = f"{v:.3f}"
            if is_best:
                ann_text = f"★ {v:.3f}"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_top + abs(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
                ann_text,
                ha="center",
                va="bottom",
                fontsize=8.5 if not is_best else 9.0,
                fontweight="bold" if is_best else "normal",
                color="#000000" if is_best else "#333333",
                rotation=40,
                zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_mlabel(m) for m in valid_metrics],
        fontsize=12,
        fontweight="bold",
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.axhline(0, color="#999999", lw=0.9, linestyle="--")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, linestyle="--", zorder=0)

    # Legend
    legend = ax.legend(
        loc="upper left",
        fontsize=11,
        frameon=True,
        framealpha=0.95,
        edgecolor="#AAAAAA",
        ncol=max(1, n_models // 3),
        handlelength=2.0,
    )
    legend.get_frame().set_linewidth(0.8)

    # Caption
    if show_std:
        ax.text(
            0.5,
            -0.18,
            "Error bars = ±1 std across corpus chunks.  "
            "★ marks the best model per metric.",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            color="#555555",
            style="italic",
        )

    fig.tight_layout()
    if save_path is None:
        safe_title = title.replace(" ", "_").replace("/", "_")[:50]
        save_path = ACTIVE_COMPARISON_DIR / f"bars_{safe_title}.png"

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Bar chart saved → {save_path}")
    return save_path


# ── Combined cross-dataset plots ──────────────────────────────────────────────


def plot_global_radar(
    summary_df: pd.DataFrame,
    metrics: List[str],
    *,
    model_subset: Optional[List[str]] = None,
    model_styles: Optional[Dict[str, Dict]] = None,
) -> None:
    """One radar per dataset + an overall-average radar, arranged in a grid."""
    _paper_style()
    styles = model_styles or DEFAULT_MODEL_STYLES
    datasets = sorted(summary_df["dataset"].unique())
    models = ordered_models(summary_df["model"].unique(), model_subset)

    n_ds = len(datasets)
    ncols = min(3, n_ds + 1)
    nrows = int(np.ceil((n_ds + 1) / ncols))
    fig = plt.figure(figsize=(ncols * 7, nrows * 7))
    fig.patch.set_facecolor("white")

    valid_metrics = [m for m in metrics if m in summary_df.columns]
    n_met = len(valid_metrics)
    angles = np.linspace(0, 2 * np.pi, n_met, endpoint=False).tolist()
    angles += angles[:1]

    def _radar_on_ax(ax, sub_df, title_str):
        ax.set_facecolor("#FAFAFA")
        ax.spines["polar"].set_edgecolor("#AAAAAA")
        ax.grid(color="#DDDDDD", linewidth=0.6, linestyle="--")

        # Normalise within this sub_df
        norm = sub_df[valid_metrics].copy()
        for m in valid_metrics:
            col = norm[m].replace([np.inf, -np.inf], np.nan)
            lo, hi = col.min(), col.max()
            norm[m] = (norm[m] - lo) / (hi - lo + 1e-12)

        for model in models:
            row = norm[sub_df["model"] == model]
            if row.empty:
                continue
            st = get_style(model, styles)
            vals = row[valid_metrics].values.flatten().tolist() + [
                row[valid_metrics[0]].values[0]
            ]
            ax.plot(
                angles,
                vals,
                color=st["color"],
                linewidth=st["linewidth"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                markersize=st["markersize"] - 1,
                zorder=4,
                label=st["label"],
            )
            ax.fill(angles, vals, color=st["color"], alpha=0.08)

        ax.set_thetagrids(
            np.degrees(angles[:-1]),
            [_mlabel(m) for m in valid_metrics],
            fontsize=10,
            color="#111111",
        )
        ax.set_ylim(0, 1)
        ax.set_title(title_str, fontsize=12, fontweight="bold", pad=18)

    # Per-dataset subplots
    for idx, ds in enumerate(datasets):
        sub = summary_df[
            (summary_df["dataset"] == ds)
            & (summary_df["chunk_idx"] == "AVERAGE")
            & (summary_df["model"].isin(models))
        ].reset_index(drop=True)
        ax = fig.add_subplot(nrows, ncols, idx + 1, polar=True)
        _radar_on_ax(ax, sub, ds.split("/")[-1])

    # Overall-average subplot
    overall = (
        summary_df[
            (summary_df["chunk_idx"] == "AVERAGE") & (summary_df["model"].isin(models))
        ]
        .groupby("model")[valid_metrics]
        .mean()
        .reset_index()
    )
    ax_all = fig.add_subplot(nrows, ncols, n_ds + 1, polar=True)
    _radar_on_ax(ax_all, overall, "All Datasets (mean)")

    # Shared legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=get_style(m, styles)["color"],
            linewidth=2,
            linestyle=get_style(m, styles)["linestyle"],
            marker=get_style(m, styles)["marker"],
            markersize=7,
            label=get_style(m, styles)["label"],
        )
        for m in models
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(models), 5),
        fontsize=11,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        edgecolor="#AAAAAA",
    )

    fig.suptitle(
        "GraphTopic vs. Baselines — Full Dataset Comparison (Radar)",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )

    out = ACTIVE_COMPARISON_DIR / "spider_global.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Global radar grid → {out}")


def plot_global_bars(
    summary_df: pd.DataFrame,
    metrics: List[str],
    *,
    model_subset: Optional[List[str]] = None,
    model_styles: Optional[Dict[str, Dict]] = None,
) -> None:
    """One grouped-bar subplot per dataset + an overall average, in a grid."""
    _paper_style()
    styles = model_styles or DEFAULT_MODEL_STYLES
    datasets = sorted(summary_df["dataset"].unique())
    models = ordered_models(summary_df["model"].unique(), model_subset)
    valid_m = [m for m in metrics if m in summary_df.columns]

    n_ds = len(datasets) + 1  # +1 for overall
    ncols = min(2, n_ds)
    nrows = int(np.ceil(n_ds / ncols))
    fig_w = max(14, len(valid_m) * 1.6 * ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, 5.5 * nrows), constrained_layout=True
    )
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten()

    def _bars_on_ax(ax, sub_df, title_str):
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, linestyle="--", zorder=0)
        x = np.arange(len(valid_m))
        bar_w = 0.70 / len(models)

        for i, model in enumerate(models):
            st = get_style(model, styles)
            row = sub_df[sub_df["model"] == model]
            if row.empty:
                continue
            vals = [
                float(row[m].values[0]) if m in row.columns else float("nan")
                for m in valid_m
            ]
            stds = [
                float(row[f"{m}_std"].values[0]) if f"{m}_std" in row.columns else 0.0
                for m in valid_m
            ]

            offset = (i - len(models) / 2 + 0.5) * bar_w
            ax.bar(
                x + offset,
                vals,
                width=bar_w * 0.88,
                color=st["color"],
                alpha=0.82,
                label=st["label"],
                yerr=stds,
                error_kw=dict(
                    elinewidth=1.1, ecolor="#444444", capsize=2.5, capthick=1.0
                ),
                hatch=st.get("hatch", ""),
                edgecolor="#333333",
                linewidth=0.5,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [_mlabel(m) for m in valid_m], fontsize=9, rotation=20, ha="right"
        )
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(title_str, fontsize=12, fontweight="bold")
        ax.axhline(0, color="#999999", lw=0.8, linestyle="--")

    for idx, ds in enumerate(datasets):
        sub = summary_df[
            (summary_df["dataset"] == ds)
            & (summary_df["chunk_idx"] == "AVERAGE")
            & (summary_df["model"].isin(models))
        ].reset_index(drop=True)
        _bars_on_ax(axes_flat[idx], sub, ds.split("/")[-1])

    # Overall average
    overall = (
        summary_df[
            (summary_df["chunk_idx"] == "AVERAGE") & (summary_df["model"].isin(models))
        ]
        .groupby("model")[
            valid_m + [f"{m}_std" for m in valid_m if f"{m}_std" in summary_df.columns]
        ]
        .mean()
        .reset_index()
    )
    _bars_on_ax(axes_flat[len(datasets)], overall, "All Datasets (mean)")

    for ax in axes_flat[n_ds:]:
        ax.set_visible(False)

    handles = [
        mpatches.Patch(
            facecolor=get_style(m, styles)["color"],
            label=get_style(m, styles)["label"],
            hatch=get_style(m, styles).get("hatch", ""),
            edgecolor="#333333",
        )
        for m in models
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(models), 5),
        fontsize=11,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        edgecolor="#AAAAAA",
    )

    fig.suptitle(
        "GraphTopic vs. Baselines — Full Dataset Comparison (Bars)",
        fontsize=16,
        fontweight="bold",
    )

    out = ACTIVE_COMPARISON_DIR / "bars_global.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Global bar grid → {out}")


# ============================================================================
# ── MAIN  ────────────────────────────────────────────────────────────────────
# ============================================================================


def main(
    force_rerun: bool = False,
    n_equal_chunks: int = N_EQUAL_CHUNKS,
    model_subset: Optional[List[str]] = None,
    model_styles: Optional[Dict[str, Dict]] = None,
    smoke: bool = False,
) -> None:
    """
    Main entry point.

    Parameters
    ----------
    force_rerun       : Re-run even if cached results exist.
    n_equal_chunks    : Equal-part count for datasets without a fixed prepare.py size.
    model_subset      : List of model names to include in visualisations.
                        ``None`` → include all.
    model_styles      : Custom style registry; overrides DEFAULT_MODEL_STYLES.
    smoke             : Tiny chunks + short Optuna; writes to full_comparison_v1_smoke.
                        Production FIXED_DOC_CHUNK_SIZE is unchanged when False.
    """
    apply_smoke_overrides(config, smoke)

    mode = "SMOKE MODE" if smoke else "FULL DATASET COMPARISON (FIXED DOC CHUNKS)"
    print("\n" + "=" * 70)
    print(f"  GRAPHTOPIC v1 — {mode}")
    print(f"  Output dir     : {ACTIVE_COMPARISON_DIR}")
    if smoke:
        print(f"  Smoke chunk    : {SMOKE_DOC_CHUNK_SIZE} docs × {SMOKE_MAX_CHUNKS} chunk(s)")
        print(f"  Smoke Optuna   : {config.N_TRIALS} trials (startup {config.N_STARTUP_TRIALS})")
        print(f"  Smoke epochs   : {config.BASELINE_EPOCHS}")
    else:
        print(f"  Equal chunks   : {n_equal_chunks}")
        print(f"  Fixed sizes    : {FIXED_DOC_CHUNK_SIZE}")
    print("=" * 70)

    device = detect_device()
    batch_size = optimal_batch_size(device)
    config.FIXED_PARAMS["batch_size"] = batch_size
    model_name = config.FIXED_PARAMS["embedding_model"]
    print(f"\n  Device     : {device}")
    print(f"  Batch size : {batch_size}  (auto-detected)")
    print(f"  Emb. model : {model_name}")

    emb_model = _load_embedding_model(model_name, device)
    global embeddingModel
    embeddingModel = emb_model
    print("  ✓ Embedding model ready\n")

    all_dfs: List[pd.DataFrame] = []
    for ds_cfg in config.DATASETS:
        safe_name = ds_cfg["name"].replace("/", "_")
        cache_csv = ACTIVE_COMPARISON_DIR / f"{safe_name}_full_comparison.csv"
        state_json = ACTIVE_COMPARISON_DIR / f"{safe_name}_comparison_state.json"
        expected_models = set(["GraphTopic"] + list(config.BASELINE_MODELS))

        if smoke:
            expected_doc_chunk_size = SMOKE_DOC_CHUNK_SIZE
        elif ds_cfg["name"] in FIXED_DOC_CHUNK_SIZE:
            expected_doc_chunk_size = int(FIXED_DOC_CHUNK_SIZE[ds_cfg["name"]])
        else:
            peek_docs = clean_docs(load_dataset_docs(ds_cfg))
            expected_doc_chunk_size = len(peek_docs) // max(1, n_equal_chunks)

        expected_state = build_comparison_state(
            dataset_name=ds_cfg["name"],
            doc_chunk_size=expected_doc_chunk_size,
            n_equal_chunks=n_equal_chunks,
            cfg=config,
            smoke=smoke,
            smoke_max_chunks=SMOKE_MAX_CHUNKS if smoke else None,
        )
        expected_state_hash = comparison_state_hash(expected_state)

        if not force_rerun and cache_csv.exists():
            df = pd.read_csv(cache_csv)
            cached_models = set(
                df[df["chunk_idx"] == "AVERAGE"]["model"].astype(str).unique()
            )
            missing_models = sorted(expected_models - cached_models)
            # FIX: a finished dataset was never revisited when a model had
            # failed on it (e.g. FASTopic on every dataset). Treat models with
            # failed chunks as missing, so run_full_dataset_comparison reloads
            # the raw-results cache and re-runs ONLY those models; successful
            # models load from cache unchanged.
            if "status" in df.columns:
                failed_models = set(
                    df[
                        df["status"]
                        .astype(str)
                        .isin(["error", "error_oom", "no_topics", "all_failed"])
                    ]["model"].astype(str)
                ) & expected_models
                missing_models = sorted(set(missing_models) | failed_models)
            cached_state = _load_json(state_json)
            cache_matches_state = (
                cached_state == expected_state
                and "doc_chunk_size" in df.columns
                and "n_equal_chunks" in df.columns
                and "comparison_state_hash" in df.columns
                and "weighted_docs" in df.columns
                and df["doc_chunk_size"]
                .dropna()
                .astype(int)
                .eq(expected_doc_chunk_size)
                .all()
                and df["n_equal_chunks"].dropna().astype(int).eq(n_equal_chunks).all()
                and df["comparison_state_hash"]
                .dropna()
                .astype(str)
                .eq(expected_state_hash)
                .all()
            )
            if missing_models or not cache_matches_state:
                reason = (
                    f"missing {missing_models}"
                    if missing_models
                    else "different comparison state"
                )
                print(
                    f"\n  ↻ Cached CSV for {ds_cfg['name']} has {reason}; "
                    "completing comparison"
                )
                try:
                    df = run_full_dataset_comparison(
                        ds_cfg,
                        emb_model,
                        config,
                        force_rerun=force_rerun,
                        n_equal_chunks=n_equal_chunks,
                        smoke=smoke,
                    )
                except KeyboardInterrupt:
                    print("\n  ⚠  Interrupted — progress saved")
                    break
                except Exception as exc:
                    print(f"  ✗ Error on {ds_cfg['name']}: {exc}")
                    traceback.print_exc()
                    continue
            else:
                print(f"\n  ✓ Found cached CSV for {ds_cfg['name']} → loading")
        else:
            try:
                df = run_full_dataset_comparison(
                    ds_cfg,
                    emb_model,
                    config,
                    force_rerun=force_rerun,
                    n_equal_chunks=n_equal_chunks,
                    smoke=smoke,
                )
            except KeyboardInterrupt:
                print("\n  ⚠  Interrupted — progress saved")
                break
            except Exception as exc:
                print(f"  ✗ Error on {ds_cfg['name']}: {exc}")
                traceback.print_exc()
                continue

        all_dfs.append(df)

        ds_label = ds_cfg["name"].split("/")[-1]
        safe_ds = ds_cfg["name"].replace("/", "_")
        models_in_df = ordered_models(
            df[df["chunk_idx"] == "AVERAGE"]["model"].unique()
        )
        vis_models = model_subset or models_in_df

        plot_radar(
            df,
            config.REPORT_METRICS,
            model_subset=vis_models,
            model_styles=model_styles,
            title=f"Model Comparison — {ds_label}",
            save_path=ACTIVE_COMPARISON_DIR / f"spider_{safe_ds}.png",
        )
        plot_bars(
            df,
            config.REPORT_METRICS,
            model_subset=vis_models,
            model_styles=model_styles,
            title=f"Model Comparison — {ds_label}",
            save_path=ACTIVE_COMPARISON_DIR / f"bars_{safe_ds}.png",
        )

    if all_dfs:
        summary = build_global_summary(all_dfs, config)
        vis_models = ordered_models(
            summary[summary["chunk_idx"] == "AVERAGE"]["model"].unique(),
            model_subset,
        )
        plot_global_radar(
            summary,
            config.REPORT_METRICS,
            model_subset=vis_models,
            model_styles=model_styles,
        )
        plot_global_bars(
            summary,
            config.REPORT_METRICS,
            model_subset=vis_models,
            model_styles=model_styles,
        )

    print(f"\n{'=' * 70}")
    print(f"  DONE — all outputs in {ACTIVE_COMPARISON_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GraphTopic v1 full-dataset comparison (or --smoke wiring check)."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run smoke test: 80-doc chunks, 1 chunk/dataset, 2 Optuna trials; "
            "outputs to full_comparison_v1_smoke/. Production chunk sizes unchanged."
        ),
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore caches and recompute.",
    )
    args = parser.parse_args()
    main(
        force_rerun=args.force_rerun,
        n_equal_chunks=N_EQUAL_CHUNKS,
        smoke=args.smoke or SMOKE_TEST,
    )
