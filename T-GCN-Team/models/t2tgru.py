import argparse
import torch
import torch.nn as nn


class T2TGRULinear(nn.Module):
    def __init__(self, feature_dim: int, input_dim: int, output_dim: int, bias: float = 0.0):
        super(T2TGRULinear, self).__init__()
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
        batch_dim, feature_dim = inputs.shape
        # batch size * feature dimension
        inputs = inputs.reshape((batch_dim, feature_dim))
        # batch_size * input dimension
        hidden_state = hidden_state.reshape(
            (batch_dim, self._input_dim)
        )
        # [x, h] (batch size * (feature dimension + input dimension))
        concatenation = torch.cat((inputs, hidden_state), dim=1)
        batch_dim_, concat_dim_ = concatenation.shape
        assert concat_dim_ == feature_dim + self._input_dim
        # [x, h] (batch_size * (feature dimension + input dimension))
        concatenation = concatenation.reshape((batch_dim_, concat_dim_))
        # [x, h]W + b (batch_size * output dimension)
        outputs = concatenation @ self.weights + self.biases
        # [x, h]W + b (batch size * output dimension)
        outputs = outputs.reshape((batch_dim, self._output_dim))
        return outputs

    def hyperparameters(self):
        return {
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class T2TGRUCell(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super(T2TGRUCell, self).__init__()
        # hidden dimension
        self._hidden_dim = hidden_dim
        # feature dimension
        self._feature_dim = feature_dim
        # GRU 1
        self.linear1 = T2TGRULinear(self._feature_dim, self._hidden_dim, self._hidden_dim * 2, bias=1.0)
        # GRU 2
        self.linear2 = T2TGRULinear(self._feature_dim, self._hidden_dim, self._hidden_dim)

    def forward(self, inputs, hidden_state):
        # GRU
        # [r, u] = sigmoid([x, h]W + b)
        # batch size * (2 * hidden state dimension))
        concatenation = torch.sigmoid(self.linear1(inputs, hidden_state))
        # r (batch size * hidden state dimension)
        # u (batch size * hidden state dimension)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh([x, (r * h)W + b])
        # c (batch size * hidden state dimension)
        c = torch.tanh(self.linear2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch size * hidden state dimension)
        new_hidden_state = u * hidden_state + (1 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {
            "hidden_dim": self._hidden_dim
        }


class T2TGRU(nn.Module):
    def __init__(self, hidden_dim: int, aspect_dim: int = 16, **kwargs):
        super(T2TGRU, self).__init__()
        # hidden state dimension
        self._hidden_dim = hidden_dim
        # linear transformation
        self._aspect_dim = aspect_dim
        self.linear_transformation_offense = nn.Linear(80, self._aspect_dim)
        self.linear_transformation_defend = nn.Linear(48, self._aspect_dim)
        self.linear_transformation_error = nn.Linear(32, self._aspect_dim)
        self.linear_transformation_influence = nn.Linear(32, self._aspect_dim)
        self._feature_dim = self._aspect_dim * 4
        # GRU cell
        self.gru_cell = T2TGRUCell(self._feature_dim, self._hidden_dim)

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
        # batch size * seq length * feature dimension
        batch_dim, seq_dim, feature_dim = inputs.shape
        # batch size * hidden state dimension)
        hidden_state = torch.zeros(batch_dim, self._hidden_dim).type_as(
            inputs
        )
        # origin feature dimension * aspect dimension
        offense_weight = self.mask_aspect(feature_dim, self.linear_transformation_offense.weight, [2, 5, 8, 9, 12, 22, 25, 28, 29, 32, 42, 45, 48, 49, 52, 62, 65, 68, 69, 72, 82, 85, 88, 89, 92, 102, 105, 108, 109, 112, 122, 125, 128, 129, 132, 142, 145, 148, 149, 152, 162, 165, 168, 169, 172, 182, 185, 188, 189, 192, 202, 205, 208, 209, 212, 222, 225, 228, 229, 232, 242, 245, 248, 249, 252, 262, 265, 268, 269, 272, 282, 285, 288, 289, 292, 302, 305, 308, 309, 312])
        defend_weight = self.mask_aspect(feature_dim, self.linear_transformation_defend.weight, [10, 14, 15, 30, 34, 35, 50, 54, 55, 70, 74, 75, 90, 94, 95, 110, 114, 115, 130, 134, 135, 150, 154, 155, 170, 174, 175, 190, 194, 195, 210, 214, 215, 230, 234, 235, 250, 254, 255, 270, 274, 275, 290, 294, 295, 310, 314, 315])
        error_weight = self.mask_aspect(feature_dim, self.linear_transformation_error.weight, [13, 17, 33, 37, 53, 57, 73, 77, 93, 97, 113, 117, 133, 137, 153, 157, 173, 177, 193, 197, 213, 217, 233, 237, 253, 257, 273, 277, 293, 297, 313, 317])
        influence_weight = self.mask_aspect(feature_dim, self.linear_transformation_influence.weight, [18, 19, 38, 39, 58, 59, 78, 79, 98, 99, 118, 119, 138, 139, 158, 159, 178, 179, 198, 199, 218, 219, 238, 239, 258, 259, 278, 279, 298, 299, 318, 319])
        # origin feature dimension * feature dimension
        aspect_weight = torch.cat((offense_weight, defend_weight, error_weight, influence_weight), dim=1)
        # feature dimension
        aspect_bias = torch.cat((self.linear_transformation_offense.bias, self.linear_transformation_defend.bias, self.linear_transformation_error.bias, self.linear_transformation_influence.bias), dim=0)
        # batch size * seq length * feature dimension
        new_inputs = inputs @ aspect_weight + aspect_bias
        # init GRU output
        outputs = list()
        for i in range(seq_dim):
            output, hidden_state = self.gru_cell(new_inputs[:, i, :], hidden_state)
            output = output.reshape((batch_dim, self._hidden_dim))
            outputs.append(output)
        last_output = outputs[-1]
        return last_output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {
            "hidden_dim": self._hidden_dim
        }
