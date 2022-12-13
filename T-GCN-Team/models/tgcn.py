import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop
import threading
import numpy as np

class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, feature_dim: int, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._feature_dim = feature_dim #21
        self._num_gru_units = num_gru_units #100
        self._output_dim = output_dim #200
        self._bias_init_value = bias #1.0
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + self._feature_dim, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes, feature_size = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, feature_size)
        inputs = inputs.reshape((batch_size, num_nodes, feature_size))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + feature_size)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + feature_size, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + feature_size) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + self._feature_dim) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + feature_size) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + feature_size, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + self._feature_dim, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + feature_size)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + feature_size)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + self._feature_dim)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }

class RelationalGraphConvLayer(nn.Module):
    def __init__(
        self, adj_1, adj_2, feature_dim: int, num_gru_units: int, output_size, bias=0.0
    ):
        super(RelationalGraphConvLayer, self).__init__()
        self.register_buffer(
            "laplacian_1", calculate_laplacian_with_self_loop(torch.FloatTensor(adj_1))
        )
        self.register_buffer(
            "laplacian_2", calculate_laplacian_with_self_loop(torch.FloatTensor(adj_2))
        )
        # self.input_size = input_size
        self._feature_dim = feature_dim #21
        self._num_gru_units = num_gru_units #100
        self.output_size = output_size
        self.num_bases = 30
        self.num_rel = 2
        self._bias_init_value = bias #1.0

        # R-GCN weights
        self.w_bases = nn.Parameter(
            torch.FloatTensor(self.num_bases, self._num_gru_units + self._feature_dim, self.output_size)
        )
        self.w_rel = nn.Parameter(torch.FloatTensor(self.num_rel, self.num_bases))
        # R-GCN bias
        self.bias = nn.Parameter(torch.FloatTensor(self.output_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_bases)
        nn.init.xavier_uniform_(self.w_rel)
        nn.init.constant_(self.bias, self._bias_init_value)

    def forward(self, X, hidden_state):
        batch_size, num_nodes, feature_size = X.shape
        inputs = X.reshape((batch_size, num_nodes, feature_size))
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + feature_size)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + feature_size, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + feature_size) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + self._feature_dim) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + feature_size) * batch_size)
        supports = []
        supports.append(self.laplacian_1 @ concatenation)
        supports.append(self.laplacian_2 @ concatenation)
        tmp = torch.cat(supports, dim=1)
        tmp = tmp.reshape(
            (num_nodes, self.num_rel * (self._num_gru_units + self._feature_dim), batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + feature_size)
        tmp = tmp.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + feature_size)
        tmp = tmp.reshape(
            (batch_size * num_nodes, 2 * (self._num_gru_units + self._feature_dim))
        )
        self.w = (
            torch.einsum("rb, bio -> rio", (self.w_rel, self.w_bases))
        )
        weights = self.w.view(
            self.w.shape[0] * self.w.shape[1], self.w.shape[2]
        )  
        outputs = tmp @ weights + self.bias
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self.output_size))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self.output_size))
        # out = torch.mm(tmp.float(), weights)  # shape(#node, output_size)
        return outputs

class TGCNCell(nn.Module):
    def __init__(self, adj, adjs, input_dim: int, input_dim_p: int, feature_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim # num_nodes for prediction(207)
        self._input_dim_p = input_dim_p
        self._feature_dim = feature_dim # feature size
        self._hidden_dim = hidden_dim # set 100
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.register_buffer("adj_1", torch.FloatTensor(adjs[0]))
        self.register_buffer("adj_2", torch.FloatTensor(adjs[1]))
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._feature_dim, self._hidden_dim, self._hidden_dim*2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._feature_dim, self._hidden_dim, self._hidden_dim
        )
        self.r_graph_conv1 = RelationalGraphConvLayer(
            self.adj_1, self.adj_2, self._feature_dim, self._hidden_dim, self._hidden_dim*2, bias=1.0
        )
        self.r_graph_conv2 = RelationalGraphConvLayer(
            self.adj_1, self.adj_2, self._feature_dim, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):
        team_inputs = inputs[:, :self._input_dim, :]
        player_inputs = inputs[:, self._input_dim:, :]
        team_hidden_state = hidden_state[:, :self._input_dim*self._hidden_dim]
        player_hidden_state = hidden_state[:, self._input_dim*self._hidden_dim:]

        # gcn
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(team_inputs, team_hidden_state))
        concatenation_p = torch.sigmoid(self.r_graph_conv1(player_inputs, player_hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)

        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        r_p, u_p = torch.chunk(concatenation_p, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)

        c = torch.tanh(self.graph_conv2(team_inputs, r * team_hidden_state))
        c_p = torch.tanh(self.r_graph_conv2(player_inputs, r_p * player_hidden_state))

        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * team_hidden_state + (1.0 - u) * c
        new_hidden_state_p = u_p * player_hidden_state + (1.0 - u_p) * c_p
        
        new_hidden_state_all = torch.cat((new_hidden_state, new_hidden_state_p), 1)
        return new_hidden_state_all, new_hidden_state_all

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN(nn.Module):
    def __init__(self, adj, adjs, feat, hidden_dim: int, linear_transfomation: bool, **kwargs):
        super(TGCN, self).__init__()
        self._input_dim = adj.shape[0] # num_nodes for prediction(207)
        self._hidden_dim = hidden_dim # set 100
        self._linear_transfomation = linear_transfomation
        self.register_buffer("adj", torch.FloatTensor(adj))
        self._adjs = adjs
        self._input_dim_p = adjs[0].shape[0]

        if linear_transfomation:
            self._aspect_dim = 10
            self.linear_transformation_offense = nn.Linear(5, self._aspect_dim)
            self.linear_transformation_defend = nn.Linear(3, self._aspect_dim)
            self.linear_transformation_error = nn.Linear(2, self._aspect_dim)
            self.linear_transformation_influence = nn.Linear(2, self._aspect_dim)
            self._feature_dim = 4*10
        else:
            self._feature_dim = feat.shape[2]

        self.tgcn_cell = TGCNCell(self.adj, self._adjs, self._input_dim, self._input_dim_p, self._feature_dim, self._hidden_dim)

    # multi thread
    def linear_transformation(self, data, new_inputs, i):
        x = data[i]
        s, n, f = x.shape
        n_inputs = torch.zeros(s, n, self._feature_dim).type_as(x)
        for j in range(s):
            for k in range(n):
                offense_input = self.linear_transformation_offense(torch.Tensor([x[j][k][2], x[j][k][5], x[j][k][8], x[j][k][9], x[j][k][12]]))
                defend_input = self.linear_transformation_defend(torch.Tensor([x[j][k][10], x[j][k][14], x[j][k][15]]))
                error_input = self.linear_transformation_error(torch.Tensor([x[j][k][13], x[j][k][17]]))
                influence_input = self.linear_transformation_influence(torch.Tensor([x[j][k][19], x[j][k][20]]))
                n_inputs[j][k] = torch.cat((offense_input, defend_input, error_input, influence_input))
        new_inputs[i] = n_inputs

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, feature_size = inputs.shape
        hidden_state = torch.zeros(batch_size, (self._input_dim + self._input_dim_p) * self._hidden_dim).type_as(
            inputs
        )
        
        # linear transforamtion
        if self._linear_transfomation:
            new_inputs = torch.zeros(batch_size, seq_len, num_nodes, self._feature_dim).type_as(inputs)

            # threads = [None] * batch_size
            # for i in range(batch_size):
            #     threads[i] = threading.Thread(target=self.linear_transformation, args = (inputs, new_inputs, i))
            #     threads[i].start()
            # for i in range(batch_size):
            #     threads[i].join()

            for i in range(batch_size):
                for j in range(seq_len):
                    for k in range(num_nodes):
                        offense_input = self.linear_transformation_offense(torch.Tensor([inputs[i][j][k][2], inputs[i][j][k][5], inputs[i][j][k][8], inputs[i][j][k][9], inputs[i][j][k][12]]))
                        defend_input = self.linear_transformation_defend(torch.Tensor([inputs[i][j][k][10], inputs[i][j][k][14], inputs[i][j][k][15]]))
                        error_input = self.linear_transformation_error(torch.Tensor([inputs[i][j][k][13], inputs[i][j][k][17]]))
                        influence_input = self.linear_transformation_influence(torch.Tensor([inputs[i][j][k][19], inputs[i][j][k][20]]))
                        new_inputs[i][j][k] = torch.cat((offense_input, defend_input, error_input, influence_input))

        output = None
        for i in range(seq_len):
            # if linear transformation
            if self._linear_transfomation:
                output, hidden_state = self.tgcn_cell(new_inputs[:, i, :, :], hidden_state)
            else:
                output, hidden_state = self.tgcn_cell(inputs[:, i, :, :], hidden_state)
            output = output.reshape((batch_size, (self._input_dim + self._input_dim_p), self._hidden_dim))
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
