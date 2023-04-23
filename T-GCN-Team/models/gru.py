import argparse
import torch
import torch.nn as nn


class GRULinear(nn.Module):
    def __init__(self, feature_dim: int, input_dim: int, output_dim: int, bias: float = 0.0):
        super(GRULinear, self).__init__()
        # feature dimension
        self._feature_dim = feature_dim
        # input dimension
        self._input_dim = input_dim
        # output dimension
        self._output_dim = output_dim
        # bias initial value
        self._bias_init_value = bias
        # weight ((feature dimension + input dimension) * output dimension)
        self.weights = nn.Parameter(
            torch.FloatTensor(self._feature_dim + self._input_dim, self._output_dim)
        )
        # biases (output dimension)
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_dim, num_nodes, feature_dim = inputs.shape
        # batch size * num of nodes * feature dimension
        inputs = inputs.reshape((batch_dim, num_nodes, feature_dim))
        # batch_size * num of nodes * input dimension
        hidden_state = hidden_state.reshape(
            (batch_dim, num_nodes, self._input_dim)
        )
        # [x, h] (batch size * num of nodes * (feature dimension + input dimension))
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        batch_dim_, num_nodes_, concat_dim_ = concatenation.shape
        assert concat_dim_ == feature_dim + self._input_dim
        # [x, h] ((batch_size * num_nodes) * (feature dimension + input dimension))
        concatenation = concatenation.reshape((batch_dim_ * num_nodes_, concat_dim_))
        # [x, h]W + b ((batch_size * num_nodes) * output dimension)
        outputs = concatenation @ self.weights + self.biases
        # [x, h]W + b (batch size * num of team nodes * output dimension)
        outputs = outputs.reshape((batch_dim, num_nodes, self._output_dim))
        # [x, h]W + b (batch size * (num of team nodes * output dimension))
        outputs = outputs.reshape((batch_dim, num_nodes * self._output_dim))
        return outputs

    def hyperparameters(self):
        return {
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class GRUCell(nn.Module):
    def __init__(self, feature_dim: int, input_dim: int, hidden_dim: int):
        super(GRUCell, self).__init__()
        # num of nodes for prediction
        self._input_dim = input_dim
        # hidden dimension
        self._hidden_dim = hidden_dim
        # feature dimension
        self._feature_dim = feature_dim
        # GRU 1
        self.linear1 = GRULinear(self._feature_dim, self._hidden_dim, self._hidden_dim * 2, bias=1.0)
        # GRU 1
        self.linear2 = GRULinear(self._feature_dim, self._hidden_dim, self._hidden_dim)

    def forward(self, inputs, hidden_state):
        # GRU
        # [r, u] = sigmoid([x, h]W + b)
        # batch size * (num of nodes * (2 * hidden state dimension))
        concatenation = torch.sigmoid(self.linear1(inputs, hidden_state))
        # r (batch size * (num of nodes * hidden state dimension))
        # u (batch size * (num of nodes * hidden state dimension))
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh([x, (r * h)W + b])
        # c (batch size * (num of team nodes * hidden state dimension))
        c = torch.tanh(self.linear2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch size, (num of team nodes * hidden state dimension))
        new_hidden_state = u * hidden_state + (1 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {
            "input_dim": self._input_dim, 
            "hidden_dim": self._hidden_dim
        }


class GRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, aspect_dim: int = 16,**kwargs):
        super(GRU, self).__init__()
        # num of nodes for prediction
        self._input_dim = input_dim
        # hidden state dimension
        self._hidden_dim = hidden_dim
        # linear transformation
        self._aspect_dim = aspect_dim
        self.linear_transformation_offense = nn.Linear(5, self._aspect_dim)
        self.linear_transformation_defend = nn.Linear(3, self._aspect_dim)
        self.linear_transformation_error = nn.Linear(2, self._aspect_dim)
        self.linear_transformation_influence = nn.Linear(4, self._aspect_dim)
        self._feature_dim = self._aspect_dim * 4
        # GRU cell
        self.gru_cell = GRUCell(self._feature_dim, self._input_dim, self._hidden_dim)

    def mask_aspect(self, feature_dim, weight, feature_index):
        # aspect feature dim * aspect dim
        aspect_weight = weight.transpose(0, 1)
        # origin feature dim * aspect dim
        mask_vector = torch.zeros(feature_dim, self._aspect_dim)
        weight_index = 0
        for i in feature_index:
            mask_vector[i] = aspect_weight[weight_index]
            weight_index += 1
        return mask_vector

    def forward(self, inputs):
        # batch size * seq length * num of nodes * feature dimension
        batch_dim, seq_dim, input_dim, feature_dim = inputs.shape
        # batch_size, seq_len, num_nodes = inputs.shape
        assert input_dim == self._input_dim
        # batch size * (num of nodes * hidden state dimension)
        hidden_state = torch.zeros(batch_dim, self._input_dim * self._hidden_dim).type_as(
            inputs
        )
        # origin feature dimension * aspect dimension
        offense_weight = self.mask_aspect(feature_dim, self.linear_transformation_offense.weight, [2, 5, 8, 9, 12])
        defend_weight = self.mask_aspect(feature_dim, self.linear_transformation_defend.weight, [10, 14, 15])
        error_weight = self.mask_aspect(feature_dim, self.linear_transformation_error.weight, [13, 17])
        influence_weight = self.mask_aspect(feature_dim, self.linear_transformation_influence.weight, [19, 20, 21, 22])
        # origin feature dimension * feature dimension
        aspect_weight = torch.cat((offense_weight, defend_weight, error_weight, influence_weight), dim=1)
        # feature dimension
        aspect_bias = torch.cat((self.linear_transformation_offense.bias, self.linear_transformation_defend.bias, self.linear_transformation_error.bias, self.linear_transformation_influence.bias), dim=0)
        # batch size * seq length * num of nodes * feature dimension
        new_inputs = inputs @ aspect_weight + aspect_bias
        # init GRU output
        outputs = list()
        for i in range(seq_dim):
            output, hidden_state = self.gru_cell(new_inputs[:, i, :, :], hidden_state)
            output = output.reshape((batch_dim, self._input_dim, self._hidden_dim))
            outputs.append(output)
        last_output = outputs[-1]
        return last_output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--applying_player", action="store_true")
        return parser

    @property
    def hyperparameters(self):
        return {
            "input_dim": self._input_dim, 
            "hidden_dim": self._hidden_dim,
        }
