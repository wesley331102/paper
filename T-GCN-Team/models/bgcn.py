import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph_conv import calculate_laplacian_with_self_loop
from utils.dict_processing import dict_processing
import numpy as np

class GraphConvolutionLayer(nn.Module):
    def __init__(self, adj: np.ndarray, feature_dim: int, input_dim: int, output_dim: int, bias: float = 0.0):
        super(GraphConvolutionLayer, self).__init__()
        # feature dimension
        self._feature_dim = feature_dim
        # input dimension
        self._input_dim = input_dim
        # output dimension
        self._output_dim = output_dim
        # bias initial value
        self._bias_init_value = bias
        # laplacian matrix
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
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
        # batch size * num of team nodes * feature dimension
        inputs = inputs.reshape((batch_dim, num_nodes, feature_dim))
        # batch_size * num of team nodes * input dimension
        hidden_state = hidden_state.reshape(
            (batch_dim, num_nodes, self._input_dim)
        )
        # [x, h] (batch size * num of team nodes * (feature dimension + input dimension))
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num of team nodes * (feature dimension + input dimension) * batch size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num of team nodes * ((feature dimension + input dimension) * batch size))
        concatenation = concatenation.reshape(
            (num_nodes, (self._feature_dim + self._input_dim) * batch_dim)
        )
        # A[x, h] (num of team nodes * ((feature dimension + input dimension) * batch size))
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num of team nodes * (feature dimension + input dimension) * batch size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, (self._feature_dim + self._input_dim), batch_dim)
        )
        # A[x, h] (batch size * num of team nodes * (feature dimension + input dimension))
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] ((batch size * num of team nodes) * (feature dimension + input dimension))
        a_times_concat = a_times_concat.reshape(
            (batch_dim * num_nodes, (self._feature_dim + self._input_dim))
        )
        # A[x, h]W + b ((batch size * num of team nodes) * output dimension)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch size * num of team nodes * output dimension)
        outputs = outputs.reshape((batch_dim, num_nodes, self._output_dim))
        # A[x, h]W + b (batch size * (num of team nodes * output dimension))
        outputs = outputs.reshape((batch_dim, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            # "num_gru_units": self._input_dim,
            # "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }

class RelationalGraphConvLayer(nn.Module):
    def __init__(
        self, adj_1: np.ndarray, adj_2: np.ndarray, feature_dim: int, input_dim: int, output_dim: int, bias: float = 0.0
    ):
        super(RelationalGraphConvLayer, self).__init__()
        # feature dimension
        self._feature_dim = feature_dim
        # input dimension
        self._input_dim = input_dim
        # output dimension
        self._output_dim = output_dim
        # bias initial value
        self._bias_init_value = bias
        # num of bases
        self._num_bases = 30
        # num of relationships
        self._num_rel = 2
        # laplacian matrices
        self.register_buffer(
            "laplacian_1", calculate_laplacian_with_self_loop(torch.FloatTensor(adj_1))
        )
        self.register_buffer(
            "laplacian_2", calculate_laplacian_with_self_loop(torch.FloatTensor(adj_2))
        )
        # weight bases (num of bases * (feature dimension + input dimension) * output dimension)
        self.w_bases = nn.Parameter(
            torch.FloatTensor(self._num_bases, self._feature_dim + self._input_dim, self._output_dim)
        )
        # weight relationships (num of relationships * num of bases)
        self.w_rel = nn.Parameter(torch.FloatTensor(self._num_rel, self._num_bases))
        # biases (output dimension)
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_bases)
        nn.init.xavier_uniform_(self.w_rel)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_dim, num_nodes, feature_dim = inputs.shape
        # batch size * num of player nodes * feature dimension
        inputs = inputs.reshape((batch_dim, num_nodes, feature_dim))
        # batch_size * num of team nodes * input dimension
        hidden_state = hidden_state.reshape(
            (batch_dim, num_nodes, self._input_dim)
        )
        # [x, h] (batch size * num of player nodes * (feature dimension + input dimension))
        a_times_concat = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num of player nodes * (feature dimension + input dimension) * batch size)
        a_times_concat = a_times_concat.transpose(0, 1).transpose(1, 2)
        # [x, h] (num of player nodes * ((feature dimension + input dimension) * batch size))
        a_times_concat = a_times_concat.reshape(
            (num_nodes, (self._feature_dim + self._input_dim) * batch_dim)
        )
        supports = []
        # A[x, h] (num of player nodes * ((feature dimension + input dimension) * batch size))
        supports.append(self.laplacian_1 @ a_times_concat)
        # num of relationships * A[x, h] num of relationships * (num of player nodes * ((feature dimension + input dimension) * batch size))
        supports.append(self.laplacian_2 @ a_times_concat)
        # A[x, h] (num of player nodes * (num of relationships * (feature dimension + input dimension) * batch size))
        a_times_concat = torch.cat(supports, dim=1)
        # A[x, h] (num of player nodes * (num of relationships * (feature dimension + input dimension)) * batch size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_rel * (self._feature_dim + self._input_dim), batch_dim)
        )
        # A[x, h] (batch size * num of player nodes * (num of relationships * (feature dimension + input dimension)))
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] ((batch size * num of player nodes) * (num of relationships * (feature dimension + input dimension)))
        a_times_concat = a_times_concat.reshape(
            (batch_dim * num_nodes, self._num_rel * (self._input_dim + self._feature_dim))
        )
        # w (num of relationships * (feature dimension + input dimension) * output dimension)
        w = (
            torch.einsum("rb, bio -> rio", (self.w_rel, self.w_bases))
        )
        # weights ((num of relationships * (feature dimension + input dimension)) * output dimension)
        weights = w.view(
            w.shape[0] * w.shape[1], w.shape[2]
        )  
        # A[x, h]W + b ((batch size * num of player nodes) * output dimension)
        outputs = a_times_concat @ weights + self.biases
        # A[x, h]W + b (batch size * num of player nodes * output dimension)
        outputs = outputs.reshape((batch_dim, num_nodes, self._output_dim))
        # A[x, h]W + b (batch size * (num of player nodes * output dimension))
        outputs = outputs.reshape((batch_dim, num_nodes * self._output_dim))
        return outputs

class ParallelCoAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_of_aspect: int, co_attention_dim: int):
        super(ParallelCoAttentionLayer, self).__init__()
        # num of aspect
        self._num_of_aspect = num_of_aspect
        # aspect dimension
        self._aspect_dim = hidden_dim // num_of_aspect
        # co-attention dimension
        self._co_attention_dim = co_attention_dim
        # weight (aspect dimension * aspect dimension)
        self.w_b = nn.Parameter(torch.FloatTensor(self._aspect_dim, self._aspect_dim))
        # weight (co-attention dimension * aspect dimension)
        self.w_t = nn.Parameter(torch.FloatTensor(self._co_attention_dim, self._aspect_dim))
        # weight (co-attention dimension * aspect dimension)
        self.w_p = nn.Parameter(torch.FloatTensor(self._co_attention_dim, self._aspect_dim))
        # weight (co-attention dimension * aspect dimension)
        self.w_ht = nn.Parameter(torch.FloatTensor(self._co_attention_dim, self._num_of_aspect))
        # weight (co-attention dimension * aspect dimension)
        self.w_hp = nn.Parameter(torch.FloatTensor(self._co_attention_dim, self._num_of_aspect))
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_b)
        nn.init.xavier_uniform_(self.w_t)
        nn.init.xavier_uniform_(self.w_p)
        nn.init.xavier_uniform_(self.w_ht)
        nn.init.xavier_uniform_(self.w_hp)
    
    def forward(self, team_hidden_state, player_hidden_state):
        team_batch_dim, team_hidden_dim  = team_hidden_state.shape
        player_batch_dim, player_num_nodes, player_hidden_dim = player_hidden_state.shape
        assert team_batch_dim == player_batch_dim and team_hidden_dim == player_hidden_dim
        # batch size * num of players * (number of aspect * aspect dimension)
        team_hidden_state_t = team_hidden_state.reshape((team_batch_dim, 1, team_hidden_dim)).repeat(1, player_num_nodes, 1)
        # batch size * num of players * (number of aspect * aspect dimension)
        player_hidden_state_p = player_hidden_state.reshape(player_batch_dim, player_num_nodes, player_hidden_dim)
        # batch size * num of players * aspect dimension * number of aspect
        team_hidden_state_t = team_hidden_state_t.reshape((team_batch_dim, player_num_nodes, self._num_of_aspect, self._aspect_dim)).permute(0, 1, 3, 2)
        # batch size * num of players * number of aspect * aspect dimension
        player_hidden_state_p = player_hidden_state_p.reshape(player_batch_dim, player_num_nodes, self._num_of_aspect, self._aspect_dim)
        # batch size * num of players * num of aspect * num of aspect
        C = torch.tanh(torch.matmul(player_hidden_state_p, torch.matmul(self.w_b, team_hidden_state_t)))
        # batch size * num of players * co-attention dimension * num of aspect
        team_co_attention_hidden_state = torch.tanh(torch.matmul(self.w_t, team_hidden_state_t) + torch.matmul(torch.matmul(self.w_p, player_hidden_state_p.permute(0, 1, 3, 2)), C))
        # batch size * num of players * co-attention dimension * num of aspect
        player_co_attention_hidden_state = torch.tanh(torch.matmul(self.w_p, player_hidden_state_p.permute(0, 1, 3, 2)) + torch.matmul(torch.matmul(self.w_t, team_hidden_state_t), C.permute(0, 1, 3, 2)))
        # batch size * num of players * num of aspect * num of aspect
        team_attention = F.softmax(torch.matmul(torch.t(self.w_ht), team_co_attention_hidden_state), dim=3)
        # batch size * num of players * num of aspect * num of aspect
        player_attention = F.softmax(torch.matmul(torch.t(self.w_hp), player_co_attention_hidden_state), dim=3)
        # batch size * num of players * number of aspect * aspect dimension
        team_hidden_state_output = torch.squeeze(torch.matmul(team_attention, team_hidden_state_t.permute(0, 1, 3, 2)))        
        # batch size * num of players * number of aspect * aspect dimension
        player_hidden_state_output = torch.squeeze(torch.matmul(player_attention, player_hidden_state_p)) 
        # batch size * num of players * (number of aspect * aspect dimension)
        team_hidden_state_output = team_hidden_state_output.reshape((team_batch_dim, player_num_nodes, team_hidden_dim))       
        # batch size * num of players * (number of aspect * aspect dimension)
        player_hidden_state_output = player_hidden_state_output.reshape((player_batch_dim, player_num_nodes, player_hidden_dim))

        return torch.mean(team_hidden_state_output, dim=1), player_hidden_state_output

class BGCNCell(nn.Module):
    def __init__(self, adj: np.ndarray, adj_1: np.ndarray, adj_2: np.ndarray, team_2_player: dict, input_dim_t: int, input_dim_p: int, aspect_num: int, feature_dim: int, hidden_dim: int, co_attention_dim: int, applying_player: bool):
        super(BGCNCell, self).__init__()
        # applying RGCN
        self._applying_player = applying_player
        # num of nodes for team
        self._input_dim_t = input_dim_t
        # num of nodes for player
        if self._applying_player:
            self._input_dim_p = input_dim_p
        # aspect dimension
        self._aspect_num = aspect_num
        # feature dimension
        self._feature_dim = feature_dim
        # hidden dimension
        self._hidden_dim = hidden_dim
        # co-attention dimension
        self._co_attention_dim = co_attention_dim
        # adjacency matrices
        self.register_buffer("adj", torch.FloatTensor(adj))
        if self._applying_player:
            self.register_buffer("adj_1", torch.FloatTensor(adj_1))
            self.register_buffer("adj_2", torch.FloatTensor(adj_2))
        # team to player dictionary
        if self._applying_player:
            self._team_2_player = dict_processing(team_2_player, self._input_dim_t, self._input_dim_p)
        # GCN 1
        self.graph_conv1 = GraphConvolutionLayer(
            self.adj, self._feature_dim, self._hidden_dim, self._hidden_dim*2, bias=1.0
        )
        # GCN 2
        self.graph_conv2 = GraphConvolutionLayer(
            self.adj, self._feature_dim, self._hidden_dim, self._hidden_dim
        )
        if self._applying_player:
            # RGCN 1
            self.r_graph_conv1 = RelationalGraphConvLayer(
                self.adj_1, self.adj_2, self._feature_dim, self._hidden_dim, self._hidden_dim*2, bias=1.0
            )
            # RGCN 2
            self.r_graph_conv2 = RelationalGraphConvLayer(
                self.adj_1, self.adj_2, self._feature_dim, self._hidden_dim, self._hidden_dim
            )
            # Co-attention
            self.co_attention = ParallelCoAttentionLayer(
                self._hidden_dim, self._aspect_num, self._co_attention_dim
            )

    def forward(self, inputs, hidden_state):
        if self._applying_player:
            # batch size * num of team nodes * feature dimension
            team_inputs = inputs[:, :self._input_dim_t, :]
            # batch size * num of player nodes * feature dimension
            player_inputs = inputs[:, self._input_dim_t:, :]
            # batch size * (num of team nodes * hidden state dimension)
            team_hidden_state = hidden_state[:, :self._input_dim_t*self._hidden_dim]
            # batch size * (num of player nodes * hidden state dimension)
            player_hidden_state = hidden_state[:, self._input_dim_t*self._hidden_dim:]
        else:
            # batch size * num of team nodes * feature dimension
            team_inputs = inputs
            # batch size * (num of team nodes * hidden state dimension)
            team_hidden_state = hidden_state

        # gcn
        # [r, u] = sigmoid(A[x, h]W + b)
        # batch size * (num of team nodes * (2 * hidden state dimension))
        concatenation_t = torch.sigmoid(self.graph_conv1(team_inputs, team_hidden_state))
        # r (batch size * (num of team nodes * hidden state dimension))
        # u (batch size * (num of team nodes * hidden state dimension))
        r_t, u_t = torch.chunk(concatenation_t, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch size * (num of team nodes * hidden state dimension))
        c_t = torch.tanh(self.graph_conv2(team_inputs, r_t * team_hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch size, (num of team nodes * hidden state dimension))
        new_hidden_state_t = u_t * team_hidden_state + (1.0 - u_t) * c_t

        if self._applying_player:
            # rgcn
            # [r, u] = sigmoid(A[x, h]W + b)
            # batch size * (num of player nodes * (2 * hidden state dimension))
            concatenation_p = torch.sigmoid(self.r_graph_conv1(player_inputs, player_hidden_state))
            # r (batch size * (num of player nodes * hidden state dimension))
            # u (batch size * (num of player nodes * hidden state dimension))
            r_p, u_p = torch.chunk(concatenation_p, chunks=2, dim=1)
            # c = tanh(A[x, (r * h)W + b])
            # c (batch size * (num of player nodes * hidden state dimension))
            c_p = torch.tanh(self.r_graph_conv2(player_inputs, r_p * player_hidden_state))
            # h := u * h + (1 - u) * c
            # h (batch size, (num of player nodes * hidden state dimension))
            new_hidden_state_p = u_p * player_hidden_state + (1.0 - u_p) * c_p
        
        if self._applying_player:
            # batch size * (num of team nodes * hidden state dimension)
            batch_dim_t, team_nodes_hidden_dim = new_hidden_state_t.shape
            # batch size * (num of player nodes * hidden state dimension)
            batch_dim_p, player_nodes_hidden_dim = new_hidden_state_p.shape
            assert batch_dim_t == batch_dim_p
            # batch size * num of team nodes * hidden state dimension
            co_attention_hidden_state_t = new_hidden_state_t.reshape((batch_dim_t, self._input_dim_t, self._hidden_dim))
            # batch size * num of player nodes * hidden state dimension
            co_attention_hidden_state_p = new_hidden_state_p.reshape((batch_dim_p, self._input_dim_p, self._hidden_dim))
            # batch size * num of team nodes * hidden state dimension
            for team in self._team_2_player:
                co_attention_hidden_state_t[:, team, :], co_attention_hidden_state_p[:, self._team_2_player[team], :] = self.co_attention(co_attention_hidden_state_t[:, team, :], co_attention_hidden_state_p[:, self._team_2_player[team], :])
            # batch size * (num of nodes * hidden state dimension)
            new_hidden_state = torch.cat((co_attention_hidden_state_t, co_attention_hidden_state_p), 1).reshape(batch_dim_t, (team_nodes_hidden_dim + player_nodes_hidden_dim))
        else:
            # batch size, (num of nodes * hidden state dimension)
            new_hidden_state = new_hidden_state_t
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"hidden_dim": self._hidden_dim}

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        # hidden dimension
        self._hidden_dim = hidden_dim
        # weight (hidden dimension * hidden dimension)
        self.w = nn.Parameter(torch.FloatTensor(self._hidden_dim, self._hidden_dim))
        # biases (hidden dimension)
        self.u = nn.Parameter(torch.FloatTensor(self._hidden_dim, 1))
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.u)
        
    def forward(self, inputs):
        # batch size * seq length * all hidden dimension
        inputs = inputs.transpose(0, 1)
        # batch size * seq length * all hidden dimension
        u = torch.tanh(torch.matmul(inputs, self.w))
        # batch size * seq length * 1
        attn = torch.matmul(u, self.u)
        # batch size * seq length * 1
        attn_score = F.softmax(attn, dim=1)
        # batch size * seq length * all hidden dimension
        scored_inputs = inputs * attn_score
        # batch size * all hidden dimension
        aggr_inputs = torch.sum(scored_inputs, dim=1)
        return aggr_inputs

class BGCN(nn.Module):
    def __init__(self, adj: np.ndarray, adj_1: np.ndarray, adj_2: np.ndarray, feat: np.ndarray, team_2_player: dict, aspect_num: int, hidden_dim: int, co_attention_dim: int, linear_transformation: bool, applying_player: bool, applying_attention: bool, **kwargs):
        super(BGCN, self).__init__()
        # applying RGCN
        self._applying_player = applying_player
        # num of nodes for team
        self._input_dim_t = adj.shape[0]
        # num of nodes for player
        self._input_dim_p = adj_1.shape[0]
        # num of aspect
        self._aspect_num = aspect_num
        # hidden state dimension
        self._hidden_dim = hidden_dim
        # co-attention_dim dimension
        self._co_attention_dim = co_attention_dim
        # using linear transformation or not
        self._linear_transformation = linear_transformation
        # applying attention
        self._applying_attention = applying_attention
        # adjacency matrices
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.register_buffer("adj_1", torch.FloatTensor(adj_1))
        self.register_buffer("adj_2", torch.FloatTensor(adj_2))

        # team to player dictionary
        self._team_2_player = team_2_player

        # feature dimension
        if self._linear_transformation:
            # tunable
            self._aspect_dim = 16
            self.linear_transformation_offense = nn.Linear(5, self._aspect_dim)
            self.linear_transformation_defend = nn.Linear(3, self._aspect_dim)
            self.linear_transformation_error = nn.Linear(2, self._aspect_dim)
            self.linear_transformation_influence = nn.Linear(2, self._aspect_dim)
        self._feature_dim = self._aspect_dim*4 if self._linear_transformation else feat.shape[2]

        # BGCN cell
        self.tgcn_cell = BGCNCell(self.adj, self.adj_1, self.adj_2, self._team_2_player, self._input_dim_t, self._input_dim_p, self._aspect_num, self._feature_dim, self._hidden_dim, self._co_attention_dim, self._applying_player)
        if self._applying_attention:
            self.attention = AttentionLayer(self._input_dim_t*self._hidden_dim)

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
        # num of nodes = num of team nodes + num of player nodes
        assert input_dim == self._input_dim_t + self._input_dim_p
        # batch size * (num of nodes * hidden state dimension)
        if self._applying_player:
            hidden_state = torch.zeros(batch_dim, input_dim * self._hidden_dim).type_as(
                inputs
            )
        else:
            hidden_state = torch.zeros(batch_dim, self._input_dim_t * self._hidden_dim).type_as(
                inputs
            )
        
        # initial input if using linear transformation
        new_inputs = None
        new_input_dim = input_dim if self._applying_player else self._input_dim_t

        # linear transforamtion
        if self._linear_transformation:
            # origin feature dimension * aspect dimension
            offense_weight = self.mask_aspect(feature_dim, self.linear_transformation_offense.weight, [2, 5, 8, 9, 12])
            defend_weight = self.mask_aspect(feature_dim, self.linear_transformation_defend.weight, [10, 14, 15])
            error_weight = self.mask_aspect(feature_dim, self.linear_transformation_error.weight, [13, 17])
            influence_weight = self.mask_aspect(feature_dim, self.linear_transformation_influence.weight, [19, 20])
            # origin feature dimension * feature dimension
            aspect_weight = torch.cat((offense_weight, defend_weight, error_weight, influence_weight), dim=1)
            # feature dimension
            aspect_bias = torch.cat((self.linear_transformation_offense.bias, self.linear_transformation_defend.bias, self.linear_transformation_error.bias, self.linear_transformation_influence.bias), dim=0)
            # batch size * seq length * num of nodes * feature dimension
            new_inputs = inputs @ aspect_weight + aspect_bias
        else:
             # batch size * seq length * num of nodes * feature dimension
            new_inputs = inputs[:, :, :new_input_dim, :]

        # initial output
        output = None
        if self._applying_attention:
            # seq size * batch size * (num of nodes * hidden dimension)
            # TODO output size
            attention_output = torch.zeros(seq_dim, batch_dim, self._input_dim_t*self._hidden_dim)
        for i in range(seq_dim):
            output, hidden_state = self.tgcn_cell(new_inputs[:, i, :new_input_dim, :], hidden_state)
            if self._applying_attention:
                attention_output[i] = output[:, :self._input_dim_t*self._hidden_dim]
            # batch size * num of nodes * hidden dimension
            output = output.reshape((batch_dim, new_input_dim, self._hidden_dim))

        # attention
        if self._applying_attention:
            output = self.attention(attention_output).reshape((batch_dim, self._input_dim_t, self._hidden_dim))
            
        return output[:, :self._input_dim_t, :]

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--aspect_num", type=int, default=4)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--co_attention_dim", type=int, default=32)
        parser.add_argument("--linear_transformation", action="store_true")
        parser.add_argument("--applying_player", action="store_true")
        parser.add_argument("--applying_attention", action="store_true")
        return parser

    @property
    def hyperparameters(self):
        return {
            "aspect_num": self._aspect_num,
            "hidden_dim": self._hidden_dim,
            "co_attention_dim": self._co_attention_dim,
            "linear_transformation": self._linear_transformation,
            "applying_player": self._applying_player,
            "applying_attention": self._applying_attention
        }
