# ============================================================================
# BASELINE MODELS
# ============================================================================
import warnings
import torch
import numpy as np
from gensim.matutils import Sparse2Corpus
from gensim.models import LdaModel
from gensim.models import LsiModel
from gensim.models import HdpModel
from gensim.models import Nmf
from bertopic import BERTopic
from gensim.corpora import Dictionary

# TopMost
from topmost import RawDataset, Preprocess, DynamicDataset
from topmost import (
    BasicTrainer,
    FASTopicTrainer,
    HierarchicalTrainer,
    DynamicTrainer,
    ECRTM,
    ProdLDA,
    ETM,
    SawETM,
    DecTM,
    NSTM,
    HyperMiner,
    DETM,
)

from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")


def preprocess_gensim(texts, ngram_range=(2, 3), max_features=20_000):
    # Create and fit CountVectorizer with desired n-gram range
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        min_df=2,
        max_df=0.85,
        max_features=max_features,
    )
    corpus_vect = vectorizer.fit_transform(texts)
    corpus_vect_gensim = Sparse2Corpus(corpus_vect, documents_columns=False)
    dictionary = Dictionary.from_corpus(
        corpus_vect_gensim,
        id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()),
    )
    return corpus_vect_gensim, dictionary


def get_lda_topics(corpus, dictionary, num_topics=10, num_words=10):
    """LDA - exact user implementation"""

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=20,
        random_state=40,
    )
    topics = []
    for topic_id, topic in lda_model.show_topics(
        num_topics=num_topics, num_words=num_words, formatted=False
    ):
        topics.append([word for word, _ in topic])
    return topics


def get_lsi_topics(corpus, dictionary, num_topics=10, num_words=10):
    """LSI - exact user implementation"""

    lsi_model = LsiModel(
        corpus=corpus, id2word=dictionary, num_topics=num_topics, random_seed=40
    )
    topics = []
    for topic_id, topic in lsi_model.show_topics(
        num_topics=num_topics, num_words=num_words, formatted=False
    ):
        topics.append([word for word, _ in topic])
    return topics


def get_hdp_topics(corpus, dictionary, num_topics=10, num_words=10):
    """HDP - exact user implementation"""

    hdp_model = HdpModel(corpus=corpus, id2word=dictionary, random_state=40)
    topics = []
    for topic_id, topic in hdp_model.show_topics(
        num_topics=num_topics, num_words=num_words, formatted=False
    ):
        topics.append([word for word, _ in topic])
    return topics


def get_nmf_topics(corpus, dictionary, num_topics=10, num_words=10):
    """NMF - exact user implementation"""

    nmf_model = Nmf(
        corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=40
    )
    topics = []
    for topic_id, topic in nmf_model.show_topics(
        num_topics=num_topics, num_words=num_words, formatted=False
    ):
        topics.append([word for word, _ in topic])
    return topics


def get_bertopic_topics(
    docs, num_topics=10, num_words=10, ngram_range=(2, 3), max_features=20_000
):
    """BERTopic with n-gram support"""
    if len(docs) <= 5:
        docs = docs * 5
    try:
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words="english",
            min_df=2,
            max_df=0.85,
            max_features=max_features,
        )
        model = BERTopic(
            nr_topics=num_topics,
            top_n_words=num_words,
            vectorizer_model=vectorizer,
            min_topic_size=2,
            verbose=False,
        )
        model.fit_transform(docs)
        topics_res = model.get_topics()
        bert_topics = []
        for topic_idx, topic_keywords in topics_res.items():
            if topic_idx != -1:
                bert_topics.append([keyword for keyword, _ in topic_keywords])
        return bert_topics
    except Exception as e:
        print(f"BERTopic error: {e}")
        return None


# def get_berttopic_topics(docs, words_per_topic=10):
#     """BERTopic - exact user implementation"""

#     if len(docs) <= 5:
#         docs = docs * 5

#     cv = CountVectorizer(ngram_range=(2, 3), stop_words="english")
#     try:
#         topic_model = BERTopic(
#             language="english",
#             top_n_words=words_per_topic,
#             vectorizer_model=cv,
#             min_topic_size=2,
#         )
#         topic_model.fit_transform(docs)
#         topics_res = topic_model.get_topics()
#         bert_topics = []
#         for topic_idx, topic_keywords in topics_res.items():
#             if topic_idx != -1:
#                 bert_topics.append([keyword for keyword, _ in topic_keywords])
#         return bert_topics
#     except Exception as e:
#         print(f"BERTopic error: {e}")
#         return None


# ==================== NEW BASELINE MODELS ====================
# Small dataset (<1k docs): max_features = 2k–5k
# Medium (10k docs): 5k–15k
# Large (100k+): 20k+


class NgramPreprocessing:
    def __init__(
        self,
        ngram_range=(2, 3),
        stopwords="english",
        max_features=20_000,
        return_sparse=False,
    ):
        self.ngram_range = ngram_range
        self.stopwords = stopwords
        self.max_features = max_features
        # FIX (FASTopic): FASTopic's internal data loader calls .toarray() on
        # train_bow, so it needs the sparse matrix. Default False keeps the
        # dense output used by ProdLDA / ECRTM / HyperMiner unchanged.
        self.return_sparse = return_sparse
        self.preprocessing = Preprocess(stopwords=stopwords)

        self.vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            stop_words=stopwords,
            min_df=1,
            max_df=0.85,
            max_features=self.max_features,
            analyzer=self._custom_analyzer,
        )

    def _custom_analyzer(self, doc):
        """
        Custom analyzer to join n-grams with underscores.
        This makes bigrams like "deep learning" into "deep_learning"
        """
        # Create a temporary vectorizer to get tokens
        temp_vec = CountVectorizer(
            ngram_range=self.ngram_range,
            stop_words=self.stopwords,
            min_df=2,
            max_df=0.85,
            max_features=self.max_features,
        )
        tokens = temp_vec.build_analyzer()(doc)

        # Replace spaces in n-grams with underscores
        return [token.replace(" ", "_") for token in tokens]

    def preprocess(self, docs, pretrained_WE=False):
        """
        Preprocess documents with n-gram generation.

        Args:
            docs: List of document strings
            pretrained_WE: Whether to use pretrained word embeddings

        Returns:
            Dictionary with train_bow, train_texts, and vocab
        """
        # First use base preprocessing for cleaning
        parsed_docs = self.preprocessing.preprocess(docs, pretrained_WE=pretrained_WE)[
            "train_texts"
        ]

        # Then create n-gram BOW representation
        train_bow = self.vectorizer.fit_transform(parsed_docs)
        if not self.return_sparse:
            train_bow = train_bow.toarray()

        rst = {
            "train_bow": train_bow,
            "train_texts": parsed_docs,
            "vocab": self.vectorizer.get_feature_names_out(),
        }

        return rst


def get_topmost_topics(
    docs,
    model_type="ECRTM",
    num_topics=10,
    num_words=10,
    ngram_range=(2, 3),
    max_features=20_000,
    # ---------------------------
    # Dataset Parameters
    # ---------------------------
    dataset_batch_size=200,
    # ---------------------------
    # Trainer Common Parameters
    # ---------------------------
    epochs=200,
    trainer_batch_size=200,
    verbose=False,
    **kwargs,
):
    """
    Unified interface for all TopMost models with n-gram support.

    Args:
        docs: List of document strings
        model_type: One of ["ECRTM", "FASTopic", "ProdLDA", "ETM", "DecTM", "NSTM",
                            "HyperMiner", "SawETM", "DETM"]
        num_topics: Number of topics to extract
        num_words: Number of top words per topic
        ngram_range: Tuple (min_n, max_n) for n-gram generation
                     (1,1) = unigrams, (2,2) = bigrams only, (1,2) = uni+bi, (2,3) = bi+tri
        vocab_size: Maximum vocabulary size
        **kwargs: Additional model-specific parameters

    Returns:
        List[List[str]] - Top words for each topic (with underscores removed from n-grams)
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Preprocessing & Dataset with N-gram support
        preprocess = NgramPreprocessing(
            ngram_range=ngram_range, max_features=max_features
        )

        if model_type == "DETM":
            n = len(docs)
            # timestamps = [
            #     0 if i < n // 3 else 1 if i < 2 * n // 3 else 2 for i in range(n)
            # ]
            # dataset = DynamicDataset(docs, timestamps, preprocess, device=device)
            dataset = DynamicDataset(
                dataset_dir=docs,
                batch_size=dataset_batch_size,
                read_labels=False,
                device=device,
            )
        else:
            dataset = RawDataset(
                docs=docs,
                preprocess=preprocess,
                batch_size=dataset_batch_size,
                device=device,
            )

        # 2. Model & Trainer Initialization
        vocab_size_actual = dataset.vocab_size
        is_hierarchical = False  # Flag for special handling

        if model_type == "FASTopic":
            # trainer = FASTopicTrainer(
            #     dataset, num_topics=num_topics, epochs=50, verbose=False
            # )
            # FIX (FASTopic): FASTopicTrainer otherwise builds its own default
            # single-word vocabulary (Preprocess(stopwords=[])), ignoring the
            # n-gram preprocessing used by every other model. Its topics were
            # then single words that the 2-3-gram coherence dictionary could
            # not find ("unable to interpret topic ..."), and the ~44k-word
            # vocabulary made full-batch training run out of GPU memory.
            # Give it the same NgramPreprocessing (same ngram_range and
            # max_features) as ProdLDA / ECRTM / HyperMiner.
            trainer = FASTopicTrainer(
                dataset=dataset,
                num_topics=num_topics,
                num_top_words=num_words,
                epochs=epochs,
                verbose=verbose,
                preprocess=NgramPreprocessing(
                    ngram_range=ngram_range,
                    max_features=max_features,
                    return_sparse=True,
                ),
            )
        elif model_type in ["HyperMiner", "SawETM"]:
            model_class = HyperMiner if model_type == "HyperMiner" else SawETM
            model = model_class(
                vocab_size_actual, num_topics_list=[num_topics, max(2, num_topics // 2)]
            )
            # trainer = HierarchicalTrainer(model, dataset, epochs=50, verbose=False)
            trainer = HierarchicalTrainer(
                model=model,
                dataset=dataset,
                num_top_words=num_words,
                epochs=epochs,
                batch_size=trainer_batch_size,
                verbose=verbose,
            )
            # For hierarchical models, we need special handling
            is_hierarchical = True
        elif model_type == "DETM":
            model = DETM(vocab_size_actual, num_topics=num_topics, num_times=3)
            # trainer = DynamicTrainer(model, dataset, epochs=50, verbose=False)
            trainer = DynamicTrainer(
                model=model,
                dataset=dataset,
                num_top_words=num_words,
                epochs=epochs,
                batch_size=trainer_batch_size,
                verbose=verbose,
            )
        else:
            model_map = {
                "ECRTM": ECRTM,
                "ProdLDA": ProdLDA,
                "ETM": ETM,
                "DecTM": DecTM,
                "NSTM": NSTM,
            }
            model = model_map[model_type](vocab_size_actual, num_topics=num_topics)
            trainer = BasicTrainer(
                model=model,
                dataset=dataset,
                num_top_words=num_words,
                epochs=epochs,
                batch_size=trainer_batch_size,
                verbose=verbose,
            )

        # 3. Training
        trainer.train()

        # 4. Extract topics using the trainer's method
        try:
            # For hierarchical models, use only first level topics
            if is_hierarchical:
                beta_np = trainer.get_beta()
                # Get only the first level (top-level topics)
                if isinstance(beta_np, list):
                    beta_np = beta_np[0]  # First hierarchy level

                vocab = np.array(dataset.vocab)
                top_words = []

                num_actual_topics = min(num_topics, beta_np.shape[0])

                for topic_idx in range(num_actual_topics):
                    topic_dist = beta_np[topic_idx]

                    # Ensure we don't exceed vocabulary size
                    num_words_actual = min(num_words, len(vocab), len(topic_dist))

                    # Sort indices by probability (descending)
                    top_indices = np.argsort(topic_dist)[-num_words_actual:][::-1]

                    # Ensure indices are within bounds
                    top_indices = top_indices[top_indices < len(vocab)]
                    top_indices = top_indices[
                        :num_words_actual
                    ]  # Limit to requested number

                    # Map indices to words
                    topic_words_raw = vocab[top_indices].tolist()
                    # Remove underscores from n-grams
                    topic_words_clean = [
                        word.replace("_", " ") for word in topic_words_raw
                    ]
                    top_words.append(topic_words_clean)

                return top_words

            # Try to get topics directly from trainer if available
            if hasattr(trainer, "get_topics"):
                topics_data = trainer.get_topics(num_words)
                top_words = []
                for topic in topics_data:
                    # Clean underscores from n-grams
                    topic_words_clean = [word.replace("_", " ") for word in topic]
                    top_words.append(topic_words_clean)
                return top_words[:num_topics]
        except Exception as e:
            if is_hierarchical:
                print(f"Hierarchical model extraction failed: {e}, using fallback")
            pass

        # Fallback: Extract from beta matrix
        beta_np = trainer.get_beta()
        if model_type == "FASTopic":
            # FIX (FASTopic): its beta columns follow FASTopic's own vocabulary
            # (built inside fit_transform), not dataset.vocab.
            vocab = np.array(trainer.model.vocab)
        else:
            vocab = np.array(dataset.vocab)
        top_words = []

        # Extract top words for each topic
        num_actual_topics = min(num_topics, beta_np.shape[0])

        for topic_idx in range(num_actual_topics):
            topic_dist = beta_np[topic_idx]

            # Ensure we don't exceed vocabulary size
            num_words_actual = min(num_words, len(vocab))

            # Get top indices (handle edge case where topic_dist might be shorter)
            if len(topic_dist) < num_words_actual:
                num_words_actual = len(topic_dist)

            # Sort indices by probability (descending)
            top_indices = np.argsort(topic_dist)[-num_words_actual:][::-1]

            # Ensure indices are within bounds
            top_indices = top_indices[top_indices < len(vocab)]

            # Map indices to words
            topic_words_raw = vocab[top_indices].tolist()
            # Remove underscores from n-grams to make them readable
            topic_words_clean = [word.replace("_", " ") for word in topic_words_raw]
            top_words.append(topic_words_clean)

        return top_words

    except Exception as e:
        print(f"TopMost [{model_type}] engine error: {e}")
        import traceback

        traceback.print_exc()
        return None
