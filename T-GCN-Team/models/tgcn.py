import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop


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


class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, feature_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim # num_nodes for prediction(207)
        self._feature_dim = feature_dim # feature size
        self._hidden_dim = hidden_dim # set 100
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._feature_dim, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._feature_dim, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN(nn.Module):
    def __init__(self, adj, feat, hidden_dim: int, **kwargs):
        super(TGCN, self).__init__()
        self._input_dim = adj.shape[0] # num_nodes for prediction(207)
        self._hidden_dim = hidden_dim # set 100
        self.register_buffer("adj", torch.FloatTensor(adj))

        # if no linear transformation
        # self._feature_dim = feat.shape[2]
        # if linear transformation
        self._aspect_dim = 10
        self.linear_transformation_offense = nn.Linear(5, self._aspect_dim)
        self.linear_transformation_defend = nn.Linear(3, self._aspect_dim)
        self.linear_transformation_error = nn.Linear(2, self._aspect_dim)
        self.linear_transformation_influence = nn.Linear(2, self._aspect_dim)
        self._feature_dim = 4*10

        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._feature_dim, self._hidden_dim)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, feature_size = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        
        # linear transforamtion
        new_inputs = torch.zeros(batch_size, seq_len, num_nodes, self._feature_dim).type_as(inputs)
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
            # if no linear transformation
            # output, hidden_state = self.tgcn_cell(inputs[:, i, :, :], hidden_state)
            # if linear transformation
            output, hidden_state = self.tgcn_cell(new_inputs[:, i, :, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
