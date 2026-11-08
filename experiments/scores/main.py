# ============================================================================
# METRICS - EXACT USER IMPLEMENTATIONS
# ============================================================================

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary


def calculate_topic_diversity(topics_list) -> Dict[str, float]:
    """Calculate topic diversity score"""
    total_unique_words = set()
    total_words = []
    for topic_words in topics_list:
        total_words.extend(topic_words)
        total_unique_words.update(topic_words)

    total_unique_words_should_be = len(total_words)
    total_unique_words_found = len(total_unique_words)
    diversity_score = (
        total_unique_words_found / total_unique_words_should_be
        if total_unique_words_should_be > 0
        else 0.0
    )

    return {
        "topic_diversity": diversity_score,
        # "unique_keyphrases": total_unique_words_found,
        # "total_keyphrases": total_unique_words_should_be,
    }


def calculate_coherence_scores(
    topics: List[List[str]],
    docs: List[str],
    ngram_range: Tuple[int, int] = (2, 3),
    num_of_words_per_topic: int = 10,
    coherence_window_size: int = 110,  # ← wider window helps n-gram phrases
) -> Dict[str, float]:
    try:
        import re

        token_pattern = r"(?u)\b[\w-]+\b"

        # ── Build UNIGRAM tokenisation for coherence ─────────────────────
        # c_uci / c_npmi / u_mass need individual word co-occurrence stats.
        # Multi-word keyphrases are split so that pairs can actually be found
        # in the sliding window over the raw text.
        uni_vectorizer = CountVectorizer(
            stop_words="english", token_pattern=token_pattern
        )
        uni_vectorizer.fit(docs)
        uni_analyzer = uni_vectorizer.build_analyzer()
        tokens = [uni_analyzer(doc) for doc in docs]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(tok) for tok in tokens]
        texts = [[dictionary[wid] for wid, _ in doc] for doc in corpus]

        # ── Decompose n-gram topic words → unigrams ──────────────────────
        stop_words = uni_vectorizer.get_stop_words() or set()

        def _to_unigrams(phrase: str) -> List[str]:
            words = re.findall(r"\b\w+\b", phrase.lower())
            return [
                w for w in words if w not in stop_words and w in dictionary.token2id
            ]

        topic_words = []
        for topic in topics:
            unigrams = []
            seen = set()
            for phrase in topic:
                for w in _to_unigrams(phrase):
                    if w not in seen:
                        seen.add(w)
                        unigrams.append(w)
            if unigrams:
                topic_words.append(unigrams[:num_of_words_per_topic])

        topic_words = [t for t in topic_words if len(t) > 0]

        coherence_scores = {}
        for measure in ["c_v", "u_mass", "c_uci", "c_npmi"]:
            try:
                cm = CoherenceModel(
                    topics=topic_words,
                    texts=texts,
                    corpus=corpus,
                    dictionary=dictionary,
                    coherence=measure,
                    topn=num_of_words_per_topic,
                    window_size=coherence_window_size
                    if measure in ("c_uci", "c_npmi", "c_v")
                    else None,
                )
                per_topic = cm.get_coherence_per_topic()
                score = float(np.nanmean(per_topic))
                coherence_scores[measure] = (
                    0.0 if (np.isnan(score) or np.isinf(score)) else score
                )
            except Exception as e:
                coherence_scores[measure] = 0.0
        return coherence_scores

    except Exception:
        return {"c_v": np.nan, "u_mass": np.nan, "c_uci": np.nan, "c_npmi": np.nan}


def get_semantic_score(model, topics_list, debug=False):
    """Semantic scores - exact user implementation"""

    def get_avg_intra_score(model, topics):
        avg_intra_similarities = []
        for topic in topics:
            if len(topic) > 0:
                embeddings = model.encode(topic, convert_to_numpy=True)
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.reshape(1, -1)
                if embeddings.size > 0:
                    similarity_matrix = cosine_similarity(embeddings)
                    avg_intra_similarities.append(np.mean(similarity_matrix))
        return np.mean(avg_intra_similarities) if avg_intra_similarities else 0.0

    def get_avg_inter_score(model, topics):
        valid_topics = [t for t in topics if len(t) > 0]
        if len(valid_topics) < 2:
            return 0.0

        inter_distances = []
        for i in range(len(valid_topics)):
            for j in range(i + 1, len(valid_topics)):
                embeddings1 = model.encode(valid_topics[i], convert_to_numpy=True)
                embeddings2 = model.encode(valid_topics[j], convert_to_numpy=True)

                if len(embeddings1.shape) == 1:
                    embeddings1 = embeddings1.reshape(1, -1)
                if len(embeddings2.shape) == 1:
                    embeddings2 = embeddings2.reshape(1, -1)

                if embeddings1.size == 0 or embeddings2.size == 0:
                    continue

                similarity_matrix = cosine_similarity(embeddings1, embeddings2)
                inter_distances.append(1 - np.mean(similarity_matrix))

        return np.mean(inter_distances) if inter_distances else 0.0

    intra_avg = get_avg_intra_score(model, topics_list)
    inter_avg = get_avg_inter_score(model, topics_list)
    coherence = np.array([intra_avg, inter_avg]).mean()

    return {
        "intra_similarity_avg": intra_avg,
        "inter_distances_avg": inter_avg,
        "coherence": coherence,
    }


def calculate_comprehensive_metrics(
    topics, docs, embedding_model, ngram_range=(2, 3), num_of_words_per_topic=10
) -> Dict[str, float]:
    """Calculate all metrics"""
    metrics = {}

    # Basic stats
    # metrics["num_topics"] = len(topics)
    # metrics["avg_topic_size"] = np.mean([len(t) for t in topics]) if topics else 0
    # metrics["min_topic_size"] = min([len(t) for t in topics]) if topics else 0
    # metrics["max_topic_size"] = max([len(t) for t in topics]) if topics else 0
    # metrics["std_topic_size"] = np.std([len(t) for t in topics]) if topics else 0

    # Topic diversity (REQUIRED - used in paper)
    diversity_metrics = calculate_topic_diversity(topics)
    metrics.update(diversity_metrics)

    # Coherence scores (REQUIRED - c_v, u_mass, c_uci, c_npmi)
    try:
        coherence_metrics = calculate_coherence_scores(
            topics, docs, ngram_range, num_of_words_per_topic
        )
        metrics["c_v"] = coherence_metrics.get("c_v", 0.0)
        metrics["u_mass"] = coherence_metrics.get("u_mass", 0.0)
        metrics["c_uci"] = coherence_metrics.get("c_uci", 0.0)
        metrics["c_npmi"] = coherence_metrics.get("c_npmi", 0.0)
    except Exception:
        metrics.update(
            {
                "c_v": np.nan,
                "u_mass": np.nan,
                "c_uci": np.nan,
                "c_npmi": np.nan,
            }
        )

    # Semantic scores (REQUIRED - intra_similarity_avg, inter_distances_avg, coherence)
    semantic_metrics = get_semantic_score(embedding_model, topics)
    metrics["intra_similarity_avg"] = semantic_metrics["intra_similarity_avg"]
    metrics["inter_distances_avg"] = semantic_metrics["inter_distances_avg"]
    metrics["semantic_coherence"] = semantic_metrics["coherence"]

    # # Topic overlap (additional stat)
    # try:
    #     overlaps = []
    #     for i, t1 in enumerate(topics):
    #         for j, t2 in enumerate(topics[i + 1 :], i + 1):
    #             overlap = (
    #                 len(set(t1) & set(t2)) / len(set(t1) | set(t2))
    #                 if (set(t1) | set(t2))
    #                 else 0
    #             )
    #             overlaps.append(overlap)
    #     metrics["avg_topic_overlap"] = np.mean(overlaps) if overlaps else 0.0
    #     metrics["max_topic_overlap"] = max(overlaps) if overlaps else 0.0
    # except Exception:
    #     metrics["avg_topic_overlap"] = 0.0
    #     metrics["max_topic_overlap"] = 0.0

    return metrics
