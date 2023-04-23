import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import utils.data.functions
import torch.nn as nn

class SpatioTemporalCSVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        feat_path: str,
        p_feat_path: str,
        player_team_path: str,
        y_path: str,
        adj_path: str,
        adj_1_path: str,
        adj_2_path: str,
        adj_3_path: str,
        adj_4_path: str,
        adj_5_path: str,
        batch_size: int = 64,
        seq_len: int = 12,
        pre_len: int = 1,
        split_ratio: float = 0.8,
        normalize: bool = True,
        T2T: bool = False,
        output_attention: str = "None",
        **kwargs
    ):
        super(SpatioTemporalCSVDataModule, self).__init__()
        self.T2T = T2T
        self._feat_path = feat_path
        self._p_feat_path = p_feat_path
        self._player_team_path = player_team_path
        self._y_path = y_path
        self._adj_path = adj_path
        self._adj_1_path = adj_1_path
        self._adj_2_path = adj_2_path
        self._adj_3_path = adj_3_path
        self._adj_4_path = adj_4_path
        self._adj_5_path = adj_5_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        if self.T2T:
            self._feat = utils.data.functions.load_T2T_features(self._feat_path)
            self._y = utils.data.functions.load_T2T_targets(self._y_path)
        else:
            self._feat = utils.data.functions.load_features(self._feat_path, self._p_feat_path)
            self._y = utils.data.functions.load_targets(self._y_path)
        self._player_team_dict = utils.data.functions.load_team_player_dict(self._player_team_path)
        self._adj = utils.data.functions.load_adjacency_matrix(self._adj_path)
        self._adj_1 = utils.data.functions.load_adjacency_matrix(self._adj_1_path)
        self._adj_2 = utils.data.functions.load_adjacency_matrix(self._adj_2_path)
        self._adj_3 = utils.data.functions.load_adjacency_matrix(self._adj_3_path)
        self._adj_4 = utils.data.functions.load_adjacency_matrix(self._adj_4_path)
        self._adj_5 = utils.data.functions.load_adjacency_matrix(self._adj_5_path)
        self._output_attention = output_attention

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--seq_len", type=int, default=12)
        parser.add_argument("--split_ratio", type=float, default=0.8)
        parser.add_argument("--normalize", type=bool, default=True)
        return parser

    def setup(self, stage: str = None):
        if self.T2T:
            (
                self.train_dataset,
                self.val_dataset,
            ) = utils.data.functions.generate_torch_datasets_T2T(
                self._feat,
                self._y,
                split_ratio=self.split_ratio,
                normalize=self.normalize,
            )
        else:
            (
                self.train_dataset,
                self.val_dataset,
            ) = utils.data.functions.generate_torch_datasets(
                self._feat,
                self._y,
                output_attention=self._output_attention,
                split_ratio=self.split_ratio,
                normalize=self.normalize
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

    @property
    def adj(self):
        return self._adj

    @property
    def adj_1(self):
        return self._adj_1

    @property
    def adj_2(self):
        return self._adj_2

    @property
    def adj_3(self):
        return self._adj_3

    @property
    def adj_4(self):
        return self._adj_4

    @property
    def adj_5(self):
        return self._adj_5

    @property
    def feat(self):
        return self._feat

    @property
    def player_2_team(self):
        return self._player_team_dict