import numpy as np


def dcg_at_k(relevance_scores, k):
    """
    Computes DCG@k for given relevance scores.
    :param relevance_scores: List of relevance scores in rank order (first element is top result).
    :param k: Number of top documents to consider.
    """
    relevance_scores = np.array(relevance_scores[:k])

    indices = np.arange(0, len(relevance_scores), 1)
    return (relevance_scores / np.log2(indices + 2)).sum()


def ndcg_at_k(relevance_scores, k):
    """
    Computes nDCG@k.
    :param relevance_scores: List of relevance scores in rank order.
    :param k: Rank cutoff.
    """
    dcg = dcg_at_k(relevance_scores, k)
    ideal_dcg = dcg_at_k(sorted(relevance_scores, reverse=True), k)
    return dcg / ideal_dcg if ideal_dcg != 0 else 0.0
