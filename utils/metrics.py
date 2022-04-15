# TODO: Recall@K, NDCG@K

import numpy as np


def map_at_k(actual: np.ndarray, pred: np.ndarray, top_k: int) -> float:
    r"""Mean average precision at k.

    Parameters
    ----------
    actual : np.ndarray
        A matrix with actual values.
    pred : np.ndarray
        A matrix with predictions.
    top_k : int

    Returns
    -------
    float
        Mean average precision at k
    """
    if not _assert_same_dimension(actual, pred):
        raise AssertionError("Two input matrices should have same dimension.")

    map_ = 0

    num_users = len(pred)
    top_k_items = _topk(input=pred, k=top_k)

    for i in range(num_users):
        actual_item = set(actual[i].nonzero()[0])
        pred_item = top_k_items[i]

        map_ += _ap_at_k(actual=actual_item, pred=pred_item, top_k=top_k)

    return map_ / num_users


def _ap_at_k(actual: np.array, pred: np.array, top_k: int) -> float:
    r"""Avearge precision at k

    Parameters
    ----------
    actual : np.array
        A list of item are to be predicted
    pred : np.array
        A list of predicted items
    top_k : int

    Returns
    -------
    float
        Average precision at k
    """

    if len(pred) > top_k:
        pred = pred[:top_k]

    p, cnt = 0, 0

    if not actual:
        return 0.0

    for idx, item in enumerate(pred):
        if item in actual:
            cnt += 1
            p += cnt / (idx + 1)

    return 0.0 if cnt == 0 else p / cnt


def recall_at_k(actual: np.ndarray, pred: np.ndarray, top_k: int) -> float:
    r"""_summary_

    Parameters
    ----------
    actual : np.ndarray
        _description_
    pred : np.ndarray
        _description_
    top_k : int
        _description_

    Returns
    -------
    float
        _description_
    """
    raise NotImplementedError


def ndcg_at_k(actual: np.ndarray, pred: np.ndarray, top_k: int) -> float:
    r"""_summary_

    Parameters
    ----------
    actual : np.ndarray
        _description_
    pred : np.ndarray
        _description_
    top_k : int
        _description_

    Returns
    -------
    float
        _description_
    """
    raise NotImplementedError


def _assert_same_dimension(actual: np.ndarray, pred: np.ndarray) -> bool:
    r"""Check the actual matrix and the prediction have same dimension.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    pred : np.ndarray
        Predicted values

    Returns
    -------
    bool
    """
    return actual.shape == pred.shape


def _topk(input: np.ndarray, k: int) -> np.ndarray:
    r"""Returns indices of k largest element of the given input matrix along
    the horizontal axis.

    Parameters
    ----------
    input : np.ndarray
        _description_
    k : int
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    return np.argsort(input)[:, -k:][:, ::-1]
