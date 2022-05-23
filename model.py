import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data


class CDAE(nn.Module):
    r"""Collaborative Denoising Auto-Encoder

    Parameters
    ----------
    num_users : int
        _description_
    num_items : int
        _description_
    num_hidden_units : int
        _description_
    corruption_ratio : float
        _description_

    References
    ----------
    [1] Wu, Yao, et al. "Collaborative denoising auto-encoders for top-n
        recommender systems." Proceedings of the ninth ACM international
        conference on web search and data mining. 2016.

    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_hidden_units: int,
        corruption_ratio: float,
    ) -> None:
        super(CDAE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_hidden_units = num_hidden_units
        self.corruption_ratio = corruption_ratio

        # CDAE consists of user embedding, encoder, decoder
        self.user_embedding = nn.Embedding(num_users, num_hidden_units)
        self.encoder = nn.Linear(num_items, num_hidden_units)
        self.decoder = nn.Linear(num_hidden_units, num_items)

        # Set to use GPU
        self.cuda()

    def forward(
        self, user_idx: torch.Tensor, matrix: torch.Tensor
    ) -> torch.Tensor:
        # Apply corruption
        matrix = F.dropout(
            matrix, p=self.corruption_ratio, training=self.training
        )
        encoder = torch.sigmoid(
            self.encoder(matrix) + self.user_embedding(user_idx)
        )
        return self.decoder(encoder)

    def train_one_epoch(
        self, train_loader: data.DataLoader, optimizer: optim.Optimizer
    ) -> torch.Tensor:
        r"""Train a single epoch.

        Parameters
        ----------
        data_loader : data.DataLoader
            Training data loader
        optimizer : optim.Optimizer
            An optimizer

        Returns
        -------
        torch.Tensor
            Loss value after training one epoch.
        """
        loss = 0
        # Turn training mode on
        self.train()

        # By using BCEWithLogitsLoss, the return value of 'forward()' method
        # is not using sigmoid function.
        loss_f = nn.BCEWithLogitsLoss()

        for (indices, input_mat) in train_loader:
            indices = indices.cuda()
            input_mat = input_mat.float().cuda()
            self.zero_grad()

            predict_mat = self.forward(user_idx=indices, matrix=input_mat)
            batch_loss = loss_f(input=predict_mat, target=input_mat)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss

        return loss / len(train_loader)

    def predict(self, train_loader: data.DataLoader) -> np.ndarray:
        r"""Predict items per users.
        Observations that already seen by each user are masked with `-inf`.

        Parameters
        ----------
        train_loader : data.DataLoader
            Training data loader

        Returns
        -------
        np.ndarray
            Prediction matrix which of dimension is same as training data.
        """
        with torch.no_grad():
            preds = np.zeros_like(train_loader.dataset.data)

            for (indices, input_mat) in train_loader:
                indices = indices.cuda()
                input_mat = input_mat.float().cuda()
                batch_pred = torch.sigmoid(self.forward(indices, input_mat))
                batch_pred = batch_pred.masked_fill(
                    input_mat.bool(), float("-inf")
                )

                indices = indices.detach().cpu().numpy()
                preds[indices] = batch_pred.detach().cpu().numpy()

        return preds
