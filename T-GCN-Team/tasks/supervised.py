import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses
from utils.dict_processing import dict_processing_loss
import torch

class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
        loss="mse",
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        applying_player: bool = True,
        applying_score: bool = False,
        # applying_score: bool = True,
        team_2_player: dict = {},
        t_dim: int = 0,
        p_dim: int = 0,
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        print('\n================torch.cuda.is_available()======================', torch.cuda.is_available(), '\n')
        self.save_hyperparameters()
        self.model = model
        self.applying_player = applying_player
        self.using_other = False
        if applying_score:
            self.regressor = nn.Linear(
                        self.model.hyperparameters.get("hidden_dim")*2,
                        1,
                    )
        else:
            if self.applying_player:
                if self.using_other:
                    self.regressor = nn.Linear(
                            self.model.hyperparameters.get("hidden_dim")*4 + 4,
                            1,
                        )
                else:
                    # GCN
                    # self.regressor1 = nn.Linear(
                    #         # self.model.hyperparameters.get("hidden_dim")*6,
                    #         self.model.hyperparameters.get("hidden_dim")*14,
                    #         16,
                    #     )
                    # self.regressor2 = nn.Linear(
                    #         16,
                    #         1,
                    #     )
                    # T2T
                    self.regressor1 = nn.Linear(
                            self.model.hyperparameters.get("hidden_dim"),
                            8,
                        )
                    self.regressor2 = nn.Linear(
                            8,
                            1,
                        )
            else:
                self.regressor = nn.Linear(
                        self.model.hyperparameters.get("hidden_dim")*2,
                        1,
                    )
        
        self._loss = loss
        self.feat_max_val = feat_max_val
        self.team_2_player = dict_processing_loss(team_2_player, t_dim, p_dim)

    def forward(self, x):
        # (batch_size, seq_len, num_nodes)
        # batch_size, _, num_nodes, feature_size = x.size()
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x)
        # modify
        predictions = hidden
        # (batch_size, num_nodes, hidden_dim)
        return predictions
        # origin
        # # (batch_size * num_nodes, hidden_dim)
        # hidden = hidden.reshape((-1, hidden.size(2)))
        # # (batch_size * num_nodes, pre_len)
        # if self.regressor is not None:
        #     predictions = self.regressor(hidden)
        # else:
        #     predictions = hidden
        # predictions = predictions.reshape((batch_size, num_nodes, -1))
        # return predictions

    def shared_step(self, batch, batch_idx):
        # (batch_size, seq_len/pre_len, num_nodes)
        x, y = batch
        # num_nodes = x.size(2)
        predictions = self(x)
        # predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        # y = y.reshape((-1, y.size(2)))
        return predictions, y

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        if self._loss == 'nba_mse':
            return utils.losses.nba_mse_with_regularizer_loss(inputs, targets, self)
        if self._loss == 'nba_rmse':
            if self.applying_player:
                return utils.losses.nba_rmse_with_player_with_regularizer_loss(inputs, targets, self, self.team_2_player)
            else:
                return utils.losses.nba_rmse_with_regularizer_loss(inputs, targets, self)
        if self._loss == 'nba_mae':
            if self.applying_player:
                # return utils.losses.nba_mae_with_player_with_regularizer_loss(inputs, targets, self, self.team_2_player)
                return utils.losses.nba_mae_with_player_with_regularizer_loss_name(inputs, targets, self, self.team_2_player)
            else:
                return utils.losses.nba_rmse_with_regularizer_loss(inputs, targets, self)
        if self._loss == 'nba_ce':
            return utils.losses.nba_cross_entropy_loss_with_player(inputs, targets, self, self.team_2_player)
        if self._loss == 'nba_score':
            return utils.losses.nba_rmse_with_score_loss(inputs, targets, self, self.team_2_player)
        if self._loss == 'nba_T2T':
            return utils.losses.nba_mae_with_player_with_regularizer_loss_T2T(inputs, targets, self)    

        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        loss = self.loss(predictions, y)
        # GCN
        # mse_mean, rmse_mean = utils.metrics.get_mean(0, y)
        # mse, rmse = utils.metrics.get_rmse(predictions, y, self, self.team_2_player)
        # mae = utils.metrics.get_mae(predictions, y, self, self.team_2_player)
        # accr = utils.metrics.get_accuracy(predictions, y, self, self.team_2_player)
        # score
        # mse, rmse = utils.metrics.get_rmse_score(predictions, y, self, self.team_2_player)
        # mae = utils.metrics.get_mae_score(predictions, y, self, self.team_2_player)
        # Name
        # mse, rmse = utils.metrics.get_rmse_name(predictions, y, self, self.team_2_player)
        # mae = utils.metrics.get_mae_name(predictions, y, self, self.team_2_player)
        # accr = utils.metrics.get_accuracy_name(predictions, y, self, self.team_2_player)
        # T2T
        mse, rmse = utils.metrics.get_rmse_T2T(predictions, y, self)
        mae = utils.metrics.get_mae_T2T(predictions, y, self)
        accr = utils.metrics.get_accuracy_T2T(predictions, y, self)
        metrics = {
            "val_loss_mse": loss,
            # "rmse_mean": rmse_mean,
            "rmse": rmse,
            "mae": mae,
            "accuracy": accr,
        }
        print('rmse: ', rmse, '=====', 'mae: ', mae, '=====', 'accr: ', accr, '=====')
        self.log_dict(metrics)
        # GCN
        # p, real_y = utils.losses.nba_output_with_player(predictions, y, self, self.team_2_player)
        # score
        # p, real_y = utils.losses.nba_output_with_player_score(predictions, y, self, self.team_2_player)
        # Name
        # p, real_y = utils.losses.nba_output_with_player_name(predictions, y, self, self.team_2_player)
        # T2T
        p, real_y = utils.losses.nba_output_with_player_T2T(predictions, y, self)
        return p, real_y

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--loss", type=str, default="mse")
        return parser
