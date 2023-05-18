import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import utils.metrics
import utils.losses
import torch

class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        attentionLayer: nn.Module,
        loss: str="nba_mae",
        applying_attention: bool = False,
        applying_player: bool = False,
        t_dim: int = 0,
        p_dim: int = 0,
        output_attention: str = "None",
        model_name: str = "BGCN",
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        print('\n================torch.cuda.is_available()======================', torch.cuda.is_available(), '\n')
        self.save_hyperparameters()
        self.model = model
        self.model_name = model_name
        self.attentionLayer = attentionLayer
        self.applying_player = applying_player
        self.output_attention = output_attention
        self._loss = loss
        self.applying_attention = applying_attention

        if self.output_attention in ["None", "self", "V1", "V2", "V2_reverse", "co", "encoder", "encoder_all"]:
            self.lt1 = nn.Linear(5, 16)
            self.lt2 = nn.Linear(3, 16)
            self.lt3 = nn.Linear(2, 16)
            self.lt4 = nn.Linear(1, 16)

        if self.applying_attention:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.model.hyperparameters.get("hidden_dim")*7, nhead=8)
            self.seq_attention = nn.TransformerEncoder(encoder_layer, num_layers=2)

        if self.applying_player:
            if self.output_attention in ["self", "V1", "V2"]:
                self.MLP_input_dim = self.model.hyperparameters.get("hidden_dim")*16
            elif self.output_attention in ["encoder"]:
                self.MLP_input_dim = self.model.hyperparameters.get("hidden_dim")*16
                encoder_layer = nn.TransformerEncoderLayer(d_model=self.model.hyperparameters.get("hidden_dim"), nhead=8)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            elif self.output_attention in ["encoder_all"]:
                self.MLP_input_dim = self.model.hyperparameters.get("hidden_dim")*38
                encoder_layer = nn.TransformerEncoderLayer(d_model=self.model.hyperparameters.get("hidden_dim"), nhead=8)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            elif self.output_attention == "co":
                self.MLP_input_dim = self.model.hyperparameters.get("hidden_dim")*38
            elif self.output_attention == "V2_reverse":
                self.MLP_input_dim = self.model.hyperparameters.get("hidden_dim")*24
            else:
                self.MLP_input_dim = self.model.hyperparameters.get("hidden_dim")*16
            self.regressor1 = nn.Linear(
                self.MLP_input_dim,
                256
            )
            self.regressor2 = nn.Linear(
                256,
                64,
            )
            self.regressor3 = nn.Linear(
                64,
                8,
            )
            self.regressor4 = nn.Linear(
                8,
                1,
            )
        elif model_name == "T2TGRU":
            self.regressor1 = nn.Linear(
                self.model.hyperparameters.get("hidden_dim"),
                8,
            )
            self.regressor2 = nn.Linear(
                8,
                1,
            )
        else:
            self.regressor1 = nn.Linear(
                self.model.hyperparameters.get("hidden_dim")*4,
                64,
            )
            self.regressor2 = nn.Linear(
                64,
                8,
            )
            self.regressor3 = nn.Linear(
                8,
                1,
            )

    def mask_aspect(self, feature_dim, weight, feature_index, aspect_dim):
        aspect_weight = weight.transpose(0, 1)
        mask_vector = torch.zeros(feature_dim, aspect_dim)
        weight_index = 0
        for i in feature_index:
            mask_vector[i] = aspect_weight[weight_index]
            weight_index += 1
        return mask_vector

    def forward(self, x):
        # (batch_size, num_nodes, hidden_dim)
        predictions = self.model(x)
        # (batch_size, num_nodes, hidden_dim)
        return predictions

    def shared_step(self, batch, batch_idx):
        # (batch_size, seq_len/pre_len, num_nodes)
        x, y = batch
        # (batch_size, num_nodes, hidden_dim)
        hidden = self(x)
        return hidden, y

    def loss(self, inputs, targets):
        if self._loss == 'nba_T2T':
            return utils.losses.nba_loss_funtion_with_regularizer_loss_T2T(inputs, targets, self)  
        if self.applying_attention:
            return utils.losses.nba_loss_funtion_with_regularizer_loss_with_seq(inputs, targets, self, self._loss)
        if self.applying_player:
            return utils.losses.nba_loss_funtion_with_regularizer_loss(inputs, targets, self, self._loss, self.output_attention)
        if self.applying_player == False:
            return utils.losses.nba_loss_funtion_with_regularizer_loss_only_team(inputs, targets, self, self._loss)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        loss, p_, y_, o_ = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        loss, p, real_y, odds = self.loss(predictions, y)
        rmse, mae, accr, gain = utils.metrics.nba_metrics(p, real_y, odds)
        metrics = {
            "val_loss_mse": loss,
            "rmse": rmse,
            "mae": mae,
            "accuracy": accr,
            "gain": gain,
        }
        print('\n=====rmse: ', rmse, '=====mae: ', mae, '=====accr: ', accr, '=====gain: ', gain, '=====\n')
        self.log_dict(metrics)
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
        parser.add_argument("--output_attention", type=str, default="None")
        return parser
