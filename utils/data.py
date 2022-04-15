from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import train_test_split


class CDAEData(data.Dataset):
    def __init__(self, data: np.ndarray) -> None:
        super(CDAEData, self).__init__()

        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> None:
        return index, self.data[index]


def preprocess(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    r"""Helper function to preprocess data.

    Parameters
    ----------
    data : pd.DataFrame
        ML-1M data which consists of user_id, item_id, rating.

    test_size : float, optional
        the proportion of the dataset to include in the test split,
        by default 0.2

    random_state : int, optional
        Seed for shuffling data, by default 0

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int, int]
    """
    num_users = np.max(data["user"])
    num_items = np.max(data["item"])

    # Every values of user and item would be used as inde
    data["user"] -= 1
    data["item"] -= 1

    train, test = train_test_split(
        data.values, test_size=test_size, random_state=random_state
    )

    train_mat = _to_matrix(data=train, num_users=num_users, num_items=num_items)
    test_mat = _to_matrix(data=test, num_users=num_users, num_items=num_items)

    return train_mat, test_mat, num_users, num_items


def _to_matrix(
    data: Iterable,
    num_users: int = None,
    num_items: int = None,
) -> np.ndarray:
    r"""Helper function to convert an iterable object into the form of matrix.

    Parameters
    ----------
    data : Iterable
        ML-1M data which consists of user_id, item_id, rating.
    num_users : int, optional
        The number of users, by default None
    num_items : int, optional
        The number of items, by default None

    Returns
    -------
    np.ndarray
        Rating matrix
    """
    if isinstance(data, np.ndarray):
        data = np.array(data, dtype=int)

    if num_users is None:
        num_users = np.max(data[:, 0])
    if num_items is None:
        num_items = np.max(data[:, 1])

    # Convert explicit ratings into the implicit form.
    data[:, 2] = (data[:, 2] > 0).astype(float)

    # initialize a matrix
    mat = np.zeros((num_users, num_items))

    for user, item, rating in {tuple(x) for x in data}:
        mat[user, item] = rating

    return mat
