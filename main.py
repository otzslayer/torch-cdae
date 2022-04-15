import os

import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from tqdm.auto import tqdm

from .model import CDAE
from .utils.data import CDAEData, preprocess
from .utils.helper import load_config
from .utils.metrics import map_at_k

CONFIG_PATH = "./config/config.yaml"

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(
        data_path,
        sep="::",
        header=None,
        names=["user", "item", "rating"],
        usecols=[0, 1, 2],
        dtype={0: np.int32, 1: np.int32, 2: np.int32},
    )


if __name__ == "__main__":

    config = load_config(CONFIG_PATH)
    params = config["params"]

    # Load data
    ratings = load_data(data_path=config["data_path"])

    # Get matrices and basic information of data
    train_mat, test_mat, num_users, num_items = preprocess(ratings)

    # Create datasets
    train_set = CDAEData(data=train_mat)
    test_set = CDAEData(data=test_mat)

    # Create data loaders from datasets
    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
    )
    test_loader = data.DataLoader(
        dataset=test_set,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
    )

    model = CDAE(
        num_users=num_users,
        num_items=num_items,
        num_hidden_units=params["num_hidden_units"],
        corruption_ratio=params["corruption_ratio"],
    )
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    for epoch in tqdm(range(params["epochs"])):
        loss = model.train_one_epoch(train_loader, optimizer)
        print(f"[Epoch {epoch}]:: Loss: {loss}")

    preds = model.predict(train_loader=train_loader)

    eval_result = map_at_k(actual=test_mat, pred=preds, top_k=10)

    print(f"MAP@10: {eval_result:.6f}")
