import numpy as np


def compute_dcg(doc_names, scores, gold_standard):
    rel = np.array([int(name == gold_standard) for name in doc_names])
    return (rel * scores / np.log2(np.arange(2, len(doc_names) + 2))).sum()


def cosine_similarity(a, b):
    """
    Computes cosine similarity between two vectors.
    :param a: First vector.
    :param b: Second vector.
    :return: Cosine similarity score.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def similarity_scores(actual, expected):
    """
    Computes the similarity scores between two lists of vectors.
    :param actual: List of actual vectors.
    :param expected: List of expected vectors.
    :return: List of similarity scores.
    """
    scores = []
    for a, b in zip(actual, expected):
        score = cosine_similarity(a, b)
        scores.append(score)

    return np.array(scores)
