import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph_conv import calculate_laplacian_with_self_loop
import numpy as np
from itertools import combinations

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
        self, adj_1: np.ndarray, adj_2: np.ndarray, adj_3: np.ndarray, adj_4: np.ndarray, adj_5: np.ndarray, feature_dim: int, input_dim: int, output_dim: int, bias: float = 0.0
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
        # tunable
        self._num_rel = 5
        # laplacian matrices
        self.register_buffer(
            "laplacian_1", calculate_laplacian_with_self_loop(torch.FloatTensor(adj_1))
        )
        self.register_buffer(
            "laplacian_2", calculate_laplacian_with_self_loop(torch.FloatTensor(adj_2))
        )
        self.register_buffer(
            "laplacian_3", calculate_laplacian_with_self_loop(torch.FloatTensor(adj_3))
        )
        self.register_buffer(
            "laplacian_4", calculate_laplacian_with_self_loop(torch.FloatTensor(adj_4))
        )
        self.register_buffer(
            "laplacian_5", calculate_laplacian_with_self_loop(torch.FloatTensor(adj_5))
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

    def forward(self, inputs, hidden_state, team_graph_list, oppo_graph_list):
        batch_dim, num_nodes, feature_dim = inputs.shape
        # batch size * num of player nodes * feature dimension
        inputs = inputs.reshape((batch_dim, num_nodes, feature_dim))
        # batch size * num of player nodes * input dimension
        hidden_state = hidden_state.reshape(
            (batch_dim, num_nodes, self._input_dim)
        )
        # batch size * num of player nodes * num of player nodes
        dynamic_laplacian_1 = self.laplacian_1.reshape((1, num_nodes, num_nodes)).repeat(batch_dim, 1, 1) * torch.stack(team_graph_list)
        # batch size * num of player nodes * num of player nodes
        dynamic_laplacian_2 = self.laplacian_2.reshape((1, num_nodes, num_nodes)).repeat(batch_dim, 1, 1) * torch.stack(team_graph_list)
        # batch size * num of player nodes * num of player nodes
        dynamic_laplacian_3 = self.laplacian_3.reshape((1, num_nodes, num_nodes)).repeat(batch_dim, 1, 1) * torch.stack(oppo_graph_list)
        # batch size * num of player nodes * num of player nodes
        dynamic_laplacian_4 = self.laplacian_4.reshape((1, num_nodes, num_nodes)).repeat(batch_dim, 1, 1) * torch.stack(oppo_graph_list)
        # batch size * num of player nodes * num of player nodes
        dynamic_laplacian_5 = self.laplacian_5.reshape((1, num_nodes, num_nodes)).repeat(batch_dim, 1, 1) * torch.stack(oppo_graph_list)

        # [x, h] (batch size * num of player nodes * (feature dimension + input dimension))
        a_times_concat = torch.cat((inputs, hidden_state), dim=2)
        
        supports = []
        # A[x, h] (num of player nodes * ((feature dimension + input dimension) * batch size))
        # tunable
        # num of relationships * A[x, h] num of player nodes * ((feature dimension + input dimension) * batch size)
        supports.append((dynamic_laplacian_1 @ a_times_concat).transpose(0, 1).transpose(1, 2))
        # num of relationships * A[x, h] num of player nodes * ((feature dimension + input dimension) * batch size)
        supports.append((dynamic_laplacian_2 @ a_times_concat).transpose(0, 1).transpose(1, 2))
        # num of relationships * A[x, h] num of player nodes * ((feature dimension + input dimension) * batch size)
        supports.append((dynamic_laplacian_3 @ a_times_concat).transpose(0, 1).transpose(1, 2))
        # num of relationships * A[x, h] num of player nodes * ((feature dimension + input dimension) * batch size)
        supports.append((dynamic_laplacian_4 @ a_times_concat).transpose(0, 1).transpose(1, 2))
        # num of relationships * A[x, h] num of player nodes * ((feature dimension + input dimension) * batch size)
        supports.append((dynamic_laplacian_5 @ a_times_concat).transpose(0, 1).transpose(1, 2))
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
        team_hidden_dim  = team_hidden_state.shape[0]
        player_num_nodes, player_hidden_dim = player_hidden_state.shape
        assert team_hidden_dim == player_hidden_dim
        # num of players * (number of aspect * aspect dimension)
        team_hidden_state_t = team_hidden_state.reshape((1, team_hidden_dim)).repeat(player_num_nodes, 1)
        # num of players * (number of aspect * aspect dimension)
        player_hidden_state_p = player_hidden_state.reshape(player_num_nodes, player_hidden_dim)
        # num of players * aspect dimension * number of aspect
        team_hidden_state_t = team_hidden_state_t.reshape((player_num_nodes, self._num_of_aspect, self._aspect_dim)).permute(0, 2, 1)
        # num of players * number of aspect * aspect dimension
        player_hidden_state_p = player_hidden_state_p.reshape(player_num_nodes, self._num_of_aspect, self._aspect_dim)
        # num of players * num of aspect * num of aspect
        C = torch.tanh(torch.matmul(player_hidden_state_p, torch.matmul(self.w_b, team_hidden_state_t)))
        # num of players * co-attention dimension * num of aspect
        team_co_attention_hidden_state = torch.tanh(torch.matmul(self.w_t, team_hidden_state_t) + torch.matmul(torch.matmul(self.w_p, player_hidden_state_p.permute(0, 2, 1)), C))
        # num of players * co-attention dimension * num of aspect
        player_co_attention_hidden_state = torch.tanh(torch.matmul(self.w_p, player_hidden_state_p.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.w_t, team_hidden_state_t), C.permute(0, 2, 1)))
        # num of players * num of aspect * num of aspect
        team_attention = F.softmax(torch.matmul(torch.t(self.w_ht), team_co_attention_hidden_state), dim=2)
        # num of players * num of aspect * num of aspect
        player_attention = F.softmax(torch.matmul(torch.t(self.w_hp), player_co_attention_hidden_state), dim=2)
        # num of players * number of aspect * aspect dimension
        team_hidden_state_output = torch.squeeze(torch.matmul(team_attention, team_hidden_state_t.permute(0, 2, 1)))        
        # num of players * number of aspect * aspect dimension
        player_hidden_state_output = torch.squeeze(torch.matmul(player_attention, player_hidden_state_p)) 
        # num of players * (number of aspect * aspect dimension)
        team_hidden_state_output = team_hidden_state_output.reshape((player_num_nodes, team_hidden_dim))       
        # num of players * (number of aspect * aspect dimension)
        player_hidden_state_output = player_hidden_state_output.reshape((player_num_nodes, player_hidden_dim))

        return torch.mean(team_hidden_state_output, dim=0), player_hidden_state_output

class BGCNCell(nn.Module):
    def __init__(self, adj: np.ndarray, adj_1: np.ndarray, adj_2: np.ndarray, adj_3: np.ndarray, adj_4: np.ndarray, adj_5: np.ndarray, team_2_player: dict, input_dim_t: int, input_dim_p: int, aspect_num: int, feature_dim: int, hidden_dim: int, co_attention_dim: int, applying_team: bool, applying_player: bool):
        super(BGCNCell, self).__init__()
        # applying GCN
        self._applying_team = applying_team
        # applying RGCN
        self._applying_player = applying_player
        # num of nodes for team
        if  self._applying_team:
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
        if  self._applying_team:
            self.register_buffer("adj", torch.FloatTensor(adj))
        if self._applying_player:
            self.register_buffer("adj_1", torch.FloatTensor(adj_1))
            self.register_buffer("adj_2", torch.FloatTensor(adj_2))
            self.register_buffer("adj_3", torch.FloatTensor(adj_3))
            self.register_buffer("adj_4", torch.FloatTensor(adj_4))
            self.register_buffer("adj_5", torch.FloatTensor(adj_5))
        # team to player dictionary
        if self._applying_player:
            self._team_2_player = team_2_player
            # dynamic graph dict
            self._team_graph_dict = self.mask_graph()

        if self._applying_team:
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
                self.adj_1, self.adj_2, self.adj_3, self.adj_4, self.adj_5, self._feature_dim, self._hidden_dim, self._hidden_dim*2, bias=1.0
            )
            # RGCN 2
            self.r_graph_conv2 = RelationalGraphConvLayer(
                self.adj_1, self.adj_2, self.adj_3, self.adj_4, self.adj_5, self._feature_dim, self._hidden_dim, self._hidden_dim
            )
        if self._applying_team and self._applying_player:
            # Co-attention
            self.co_attention = ParallelCoAttentionLayer(
                self._hidden_dim, self._aspect_num, self._co_attention_dim
            )
    
    def mask_graph(self):
        team_graph_dict = dict()
        for i in self._team_2_player.keys():
            team_dict = dict()
            for team in self._team_2_player[i]:
                team_combination = list(combinations(self._team_2_player[i][team], 2))
                team_dict[team] = team_combination
            team_graph_dict[i] = team_dict
        return team_graph_dict
    
    def get_dynamic_graph(self, seq_list):
        team_graph_list = list()
        oppo_graph_list = list()
        for batch in seq_list:
            team_dict = dict()
            team_graph = torch.zeros(self._input_dim_p, self._input_dim_p)
            oppo_graph = torch.ones(self._input_dim_p, self._input_dim_p)
            for team in self._team_2_player[batch]:
                team_combination = self._team_graph_dict[batch][team]
                team_dict[team] = team_combination
                for com in team_combination:
                    team_graph[com[0]][com[1]] = 1
                    team_graph[com[1]][com[0]] = 1
            oppo_graph = oppo_graph - team_graph
            team_graph_list.append(team_graph)
            oppo_graph_list.append(oppo_graph)
        return team_graph_list, oppo_graph_list

    def forward(self, inputs, hidden_state, seq_index):
        if self._applying_team and self._applying_player:
            seq_list = [s.item() for s in seq_index]
            # batch size * num of team nodes * feature dimension
            team_inputs = inputs[:, :self._input_dim_t, :]
            # batch size * num of player nodes * feature dimension
            player_inputs = inputs[:, self._input_dim_t:, :]
            # batch size * (num of team nodes * hidden state dimension)
            team_hidden_state = hidden_state[:, :self._input_dim_t*self._hidden_dim]
            # batch size * (num of player nodes * hidden state dimension)
            player_hidden_state = hidden_state[:, self._input_dim_t*self._hidden_dim:]
            # gcn
            concatenation_t = self.graph_conv1(team_inputs, team_hidden_state)
            # rgcn
            team_graph_list, oppo_graph_list = self.get_dynamic_graph(seq_list)
            concatenation_p = self.r_graph_conv1(player_inputs, player_hidden_state, team_graph_list, oppo_graph_list)

            # gcn
            # [r, u] = sigmoid(A[x, h]W + b)
            # batch size * (num of team nodes * (2 * hidden state dimension))
            concatenation_t = torch.sigmoid(concatenation_t)
            # r (batch size * (num of team nodes * hidden state dimension))
            # u (batch size * (num of team nodes * hidden state dimension))
            r_t, u_t = torch.chunk(concatenation_t, chunks=2, dim=1)
            # c = tanh(A[x, (r * h)W + b])
            # c (batch size * (num of team nodes * hidden state dimension))
            c_t = self.graph_conv2(team_inputs, r_t * team_hidden_state)

            # rgcn
            # [r, u] = sigmoid(A[x, h]W + b)
            # batch size * (num of player nodes * (2 * hidden state dimension))
            concatenation_p = torch.sigmoid(concatenation_p)
            # r (batch size * (num of player nodes * hidden state dimension))
            # u (batch size * (num of player nodes * hidden state dimension))
            r_p, u_p = torch.chunk(concatenation_p, chunks=2, dim=1)
            # c = tanh(A[x, (r * h)W + b])
            # c (batch size * (num of player nodes * hidden state dimension))
            c_p = self.r_graph_conv2(player_inputs, r_p * player_hidden_state, team_graph_list, oppo_graph_list)

            # co-attention
            # batch size * (num of team nodes * hidden state dimension))
            batch_dim_t, _ = c_t.shape
            # batch size * (num of player nodes * hidden state dimension))
            batch_dim_p, _ = c_p.shape
            assert batch_dim_t == batch_dim_p and batch_dim_t == len(seq_list)
            # batch size * num of team nodes * hidden state dimension
            co_attention_hidden_state_t = c_t.reshape((batch_dim_t, self._input_dim_t, self._hidden_dim))
            # batch size * num of player nodes * hidden state dimension
            co_attention_hidden_state_p = c_p.reshape((batch_dim_p, self._input_dim_p, self._hidden_dim))
            for i in range(batch_dim_t):
                team_list = self._team_2_player[seq_list[i]].keys()
                for team in team_list:
                    co_attention_hidden_state_t[i, team, :], co_attention_hidden_state_p[i, self._team_2_player[seq_list[i]][team], :] = self.co_attention(co_attention_hidden_state_t[i, team, :], co_attention_hidden_state_p[i, self._team_2_player[seq_list[i]][team], :])            
            # batch size * (num of team nodes * hidden state dimension)
            co_attention_hidden_state_t = co_attention_hidden_state_t.reshape((batch_dim_p, self._input_dim_t * self._hidden_dim))
            # batch size * (num of player nodes * hidden state dimension)
            co_attention_hidden_state_p = co_attention_hidden_state_p.reshape((batch_dim_p, self._input_dim_p * self._hidden_dim))

            # gcn
            # h := u * h + (1 - u) * c
            # h (batch size, (num of team nodes * hidden state dimension))
            new_hidden_state_t = u_t * team_hidden_state + (1.0 - u_t) * torch.tanh(co_attention_hidden_state_t)

            # rgcn
            # h := u * h + (1 - u) * c
            # h (batch size, (num of player nodes * hidden state dimension))
            new_hidden_state_p = u_p * player_hidden_state + (1.0 - u_p) * torch.tanh(co_attention_hidden_state_p)
            
            # batch size * (num of nodes * hidden state dimension)
            new_hidden_state = torch.cat((new_hidden_state_t, new_hidden_state_p), 1).reshape(batch_dim_t, ((self._input_dim_t + self._input_dim_p) * self._hidden_dim))
        elif self._applying_team:
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
            new_hidden_state = u_t * team_hidden_state + (1.0 - u_t) * c_t
        elif self._applying_player:
            seq_list = [s.item() for s in seq_index]
            # batch size * num of player nodes * feature dimension
            player_inputs = inputs
            # batch size * (num of player nodes * hidden state dimension)
            player_hidden_state = hidden_state
            # rgcn
            team_graph_list, oppo_graph_list = self.get_dynamic_graph(seq_list)
            # [r, u] = sigmoid(A[x, h]W + b)
            # batch size * (num of player nodes * (2 * hidden state dimension))
            concatenation_p = torch.sigmoid(self.r_graph_conv1(player_inputs, player_hidden_state, team_graph_list, oppo_graph_list))
            # r (batch size * (num of player nodes * hidden state dimension))
            # u (batch size * (num of player nodes * hidden state dimension))
            r_p, u_p = torch.chunk(concatenation_p, chunks=2, dim=1)
            # c = tanh(A[x, (r * h)W + b])
            # c (batch size * (num of player nodes * hidden state dimension))
            c_p = torch.tanh(self.r_graph_conv2(player_inputs, r_p * player_hidden_state, team_graph_list, oppo_graph_list))
            # h := u * h + (1 - u) * c
            # h (batch size, (num of player nodes * hidden state dimension))
            new_hidden_state = u_p * player_hidden_state + (1.0 - u_p) * c_p
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
        # weight (hidden dimension)
        self.u = nn.Parameter(torch.FloatTensor(self._hidden_dim, 1))
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.u)
        
    def forward(self, inputs):
        # batch size * num of nodes * seq length * hidden dimension
        inputs = inputs.permute(1, 2, 0, 3)
        # batch size * num of nodes * seq length * hidden dimension
        u = torch.tanh(torch.matmul(inputs, self.w))
        # batch size * num of nodes * seq length * 1
        attn = torch.matmul(u, self.u)
        # batch size * num of nodes * seq length * 1
        attn_score = F.softmax(attn, dim=2)
        # batch size * num of nodes * seq length * hidden dimension
        scored_inputs = inputs * attn_score
        # batch size * num of nodes * hidden dimension
        aggr_inputs = torch.sum(scored_inputs, dim=2)
        return aggr_inputs

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        # hidden dimension
        self._hidden_dim = hidden_dim
        # query (hidden dimension * hidden dimension)
        self.qw = nn.Parameter(torch.FloatTensor(self._hidden_dim, self._hidden_dim))
        # key (hidden dimension * hidden dimension)
        self.kw = nn.Parameter(torch.FloatTensor(self._hidden_dim, self._hidden_dim))
        # value (hidden dimension * hidden dimension)
        self.vw = nn.Parameter(torch.FloatTensor(self._hidden_dim, self._hidden_dim))
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qw)
        nn.init.xavier_uniform_(self.kw)
        nn.init.xavier_uniform_(self.vw)
        
    def forward(self, inputs, unit_dim = 8):
        # num of nodes * hidden dimension
        inputs = inputs.reshape((unit_dim, self._hidden_dim))
        # num of nodes * hidden dimension
        q = torch.tanh(torch.matmul(inputs, self.qw))
        # num of nodes * hidden dimension
        k = torch.tanh(torch.matmul(inputs, self.kw))
        # num of nodes * hidden dimension
        v = torch.tanh(torch.matmul(inputs, self.vw))
        # num of nodes * num of nodes
        attn = torch.matmul(q, k.transpose(0, 1))
        # num of nodes * num of nodes
        attn_score = F.softmax(attn, dim=1)
        # num of nodes * hidden dimension
        scored_outputs = torch.matmul(attn_score, v)
        # (num of nodes * hidden dimension)
        scored_outputs = torch.flatten(scored_outputs)
        return scored_outputs
    
class OutputAttentionV1Layer(nn.Module):
    def __init__(self, hidden_dim: int, **kwargs):
        super(OutputAttentionV1Layer, self).__init__(**kwargs)
        # hidden dimension
        self._hidden_dim = hidden_dim
        # weight (hidden dimension)
        self.w1 = nn.Parameter(torch.FloatTensor(self._hidden_dim, 1))
        # weight (hidden dimension * hidden dimension)
        self.w2 = nn.Parameter(torch.FloatTensor(self._hidden_dim, self._hidden_dim))
        # weight (hidden dimension * hidden dimension)
        self.w3 = nn.Parameter(torch.FloatTensor(self._hidden_dim, self._hidden_dim))
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w3)
        
    def forward(self, inputs, hidden_state, unit_dim = 7):
        # num of nodes * hidden dimension
        inputs = inputs.reshape((unit_dim, self._hidden_dim))
        # num of nodes * hidden dimension
        hidden_state = hidden_state.reshape((1, self._hidden_dim)).repeat((unit_dim, 1)).reshape((unit_dim, self._hidden_dim))
        # num of nodes * hidden dimension
        u = torch.tanh(torch.matmul(inputs, self.w2) + torch.matmul(hidden_state, self.w3))
        # num of nodes * 1
        attn = torch.matmul(u, self.w1)
        # num of nodes * 1
        attn_score = F.softmax(attn, dim=1)
        # num of nodes * hidden dimension
        scored_outputs = inputs * attn_score
        # num of nodes * hidden dimension
        aggr_outputs = torch.flatten(scored_outputs)
        return aggr_outputs
    
class OutputAttentionV2Layer(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int, attention_mul: bool = False, **kwargs):
        super(OutputAttentionV2Layer, self).__init__(**kwargs)
        # hidden dimension
        self._hidden_dim = hidden_dim
        # attention dimension
        self._attention_dim = attention_dim
        # attention dimension
        self._attention_mul = attention_mul
        # weight (hidden dimension * attention dimension)
        self.w1 = nn.Parameter(torch.FloatTensor(self._hidden_dim, self._attention_dim))
        if self._attention_mul == False:
            # weight (attention dimension * attention dimension)
            self.w2 = nn.Parameter(torch.FloatTensor(self._attention_dim, self._attention_dim))
            # weight (attention dimension * 1)
            self.w3 = nn.Parameter(torch.FloatTensor(self._attention_dim, 1))
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        if self._attention_mul == False:
            nn.init.xavier_uniform_(self.w2)
            nn.init.xavier_uniform_(self.w3)
        
    def forward(self, inputs, hidden_state, unit_dim = 7):
        # num of nodes * hidden dimension
        inputs = inputs.reshape((unit_dim, self._hidden_dim))
        if self._attention_mul:
            # num of nodes * 1
            attn = torch.tanh(torch.matmul(torch.matmul(inputs, self.w1), hidden_state))
            attn = attn.reshape((unit_dim, 1))
        else:
            # num of nodes * attention dimension
            hidden_state = hidden_state.reshape((1, self._attention_dim)).repeat((unit_dim, 1)).reshape((unit_dim, self._attention_dim))
            # num of nodes * attention dimension
            u = torch.tanh(torch.matmul(inputs, self.w1) + torch.matmul(hidden_state, self.w2))
            # num of nodes * 1
            attn = torch.matmul(u, self.w3)
        # num of nodes * 1
        attn_score = F.softmax(attn, dim=1)
        # num of nodes * hidden dimension
        scored_outputs = inputs * attn_score
        # num of nodes * hidden dimension
        aggr_outputs = torch.flatten(scored_outputs)
        return aggr_outputs

class OutputCoAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, history_hidden_size: int, hidden_dim: int, co_attention_dim: int = 32):
        super(OutputCoAttentionLayer, self).__init__()
        # hidden state size
        self._hidden_size = hidden_size
        # history hidden state size
        self._history_hidden_size = history_hidden_size
        # hidden state dimension
        self._hidden_dim = hidden_dim
        # co-attention dimension
        self._co_attention_dim = co_attention_dim
        # weight (hidden state dimension * hidden state dimension)
        self.w_c = nn.Parameter(torch.FloatTensor(self._hidden_dim, self._hidden_dim))
        # weight (co-attention dimension * hidden state dimension)
        self.w_t = nn.Parameter(torch.FloatTensor(self._co_attention_dim, self._hidden_dim))
        # weight (co-attention dimension * hidden state dimension)
        self.w_p = nn.Parameter(torch.FloatTensor(self._co_attention_dim, self._hidden_dim))
        # weight (co-attention dimension * hidden state size)
        self.w_ht = nn.Parameter(torch.FloatTensor(self._co_attention_dim, self._hidden_size))
        # weight (co-attention dimension * history hidden state size)
        self.w_hp = nn.Parameter(torch.FloatTensor(self._co_attention_dim, self._history_hidden_size))
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_c)
        nn.init.xavier_uniform_(self.w_t)
        nn.init.xavier_uniform_(self.w_p)
        nn.init.xavier_uniform_(self.w_ht)
        nn.init.xavier_uniform_(self.w_hp)
    
    def forward(self, hidden_state, history_hidden_state):
        # hidden state size * hidden state dimension
        hidden_state = hidden_state.reshape((self._hidden_size, self._hidden_dim))
        # history hidden state size * hidden state dimension
        history_hidden_state = history_hidden_state.reshape((self._history_hidden_size, self._hidden_dim))
        # hidden state size * history hidden state size
        C = torch.tanh(torch.matmul(hidden_state, torch.matmul(self.w_c, torch.t(history_hidden_state))))
        # co-attention dimension * hidden state size
        co_attention_hidden_state = torch.tanh(torch.matmul(self.w_t, torch.t(hidden_state)) + torch.matmul(self.w_p, torch.matmul(torch.t(history_hidden_state), torch.t(C))))
        # co-attention dimension * history hidden state size
        co_attention_history_hidden_state = torch.tanh(torch.matmul(self.w_p, torch.t(history_hidden_state)) + torch.matmul(self.w_t, torch.matmul(torch.t(hidden_state), C)))
        # hidden state size * hidden state size
        co_attention_hidden_state_score = F.softmax(torch.matmul(torch.t(self.w_ht), co_attention_hidden_state), dim=1)
        # history hidden state size * history hidden state size
        co_attention_history_hidden_state_score = F.softmax(torch.matmul(torch.t(self.w_hp), co_attention_history_hidden_state), dim=1)
        # hidden state size * hidden state dimension
        co_attention_hidden_state_output = torch.matmul(co_attention_hidden_state_score, hidden_state)
        # history hidden state size * hidden state dimension
        co_attention_history_hidden_state_output = torch.matmul(co_attention_history_hidden_state_score, history_hidden_state)
        # hidden state size * hidden state dimension
        co_attention_hidden_state_output = co_attention_hidden_state_output.reshape((self._hidden_size, self._hidden_dim))       
        # history hidden state size * hidden state dimension
        co_attention_history_hidden_state_output = co_attention_history_hidden_state_output.reshape((self._history_hidden_size, self._hidden_dim))

        return co_attention_hidden_state_output, co_attention_history_hidden_state_output

class BGCN(nn.Module):
    def __init__(self, adj: np.ndarray, adj_1: np.ndarray, adj_2: np.ndarray, adj_3: np.ndarray, adj_4: np.ndarray, adj_5: np.ndarray, team_2_player: dict, aspect_num: int, hidden_dim: int, co_attention_dim: int, applying_team: bool, applying_player: bool, applying_attention: bool, aspect_dim: int = 16, **kwargs):
        super(BGCN, self).__init__()
        # applying GCN
        self._applying_team = applying_team
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
        # applying attention
        self._applying_attention = applying_attention
        # adjacency matrices
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.register_buffer("adj_1", torch.FloatTensor(adj_1))
        self.register_buffer("adj_2", torch.FloatTensor(adj_2))
        self.register_buffer("adj_3", torch.FloatTensor(adj_3))
        self.register_buffer("adj_4", torch.FloatTensor(adj_4))
        self.register_buffer("adj_5", torch.FloatTensor(adj_5))

        # team to player dictionary
        self._team_2_player = team_2_player

        # linear transformation
        self._aspect_dim = aspect_dim
        self.linear_transformation_offense = nn.Linear(5, self._aspect_dim)
        self.linear_transformation_defend = nn.Linear(3, self._aspect_dim)
        self.linear_transformation_error = nn.Linear(2, self._aspect_dim)
        self.linear_transformation_influence = nn.Linear(4, self._aspect_dim)
        self._feature_dim = self._aspect_dim * 4

        # BGCN cell
        self.tgcn_cell = BGCNCell(self.adj, self.adj_1, self.adj_2, self.adj_3, self.adj_4, self.adj_5, self._team_2_player, self._input_dim_t, self._input_dim_p, self._aspect_num, self._feature_dim, self._hidden_dim, self._co_attention_dim, self._applying_team, self._applying_player)
        if self._applying_attention:
            self.attention = AttentionLayer(self._hidden_dim)

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
        # batch size * seq length * num of nodes + 1 * feature dimension
        batch_dim, seq_dim, input_dim_with_mem, feature_dim = inputs.shape
        # num of nodes
        input_dim = input_dim_with_mem - 1
        # batch size * seq length
        seq_index = inputs[:, :, -1, 0]
        # batch size * seq length * num of nodes * feature dimension
        inputs = inputs[:, :, :-1, :]
        # num of nodes = num of team nodes + num of player nodes
        assert input_dim == self._input_dim_t + self._input_dim_p
        # batch size * (num of nodes * hidden state dimension)
        
        # initial input
        # new_inputs = None
        if self._applying_player and self._applying_team:
            new_input_dim = input_dim
            from_index = 0
            to_index = input_dim
            hidden_state = torch.zeros(batch_dim, input_dim * self._hidden_dim).type_as(
                    inputs
                )
        elif self._applying_team:
            new_input_dim = self._input_dim_t
            from_index = 0
            to_index = self._input_dim_t
            hidden_state = torch.zeros(batch_dim, self._input_dim_t * self._hidden_dim).type_as(
                inputs
            )
        elif self._applying_player:
            new_input_dim = self._input_dim_p
            from_index = self._input_dim_t
            to_index = input_dim
            hidden_state = torch.zeros(batch_dim, self._input_dim_p * self._hidden_dim).type_as(
                inputs
            )

        # linear transforamtion (origin feature dimension * aspect dimension)
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

        # initial output
        output = None
        if self._applying_attention:
            # seq size * batch size * (num of nodes * hidden dimension)
            attention_output = torch.zeros(seq_dim, batch_dim, new_input_dim*self._hidden_dim)
        for i in range(seq_dim):
            output, hidden_state = self.tgcn_cell(new_inputs[:, i, from_index:to_index, :], hidden_state, seq_index[:, i])

            if self._applying_attention:
                attention_output[i] = output[:, :]
            # batch size * num of nodes * hidden dimension
            output = output.reshape((batch_dim, new_input_dim, self._hidden_dim))
        
        # attention
        if self._applying_attention:
            # attention_output = attention_output.reshape((seq_dim, batch_dim, new_input_dim, self._hidden_dim))
            # output = self.attention(attention_output)

            attention_output = attention_output.reshape((seq_dim, batch_dim, new_input_dim, self._hidden_dim))
            return attention_output
        
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--aspect_num", type=int, default=4)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--co_attention_dim", type=int, default=32)
        parser.add_argument("--applying_player", action="store_true")
        parser.add_argument("--applying_team", action="store_true")
        parser.add_argument("--applying_attention", action="store_true")
        return parser

    @property
    def hyperparameters(self):
        return {
            "aspect_num": self._aspect_num,
            "hidden_dim": self._hidden_dim,
            "co_attention_dim": self._co_attention_dim,
            "applying_player": self._applying_player,
            "applying_team": self._applying_team,
            "applying_attention": self._applying_attention
        }
