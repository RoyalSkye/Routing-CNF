import sys
import os
sys.path.append(os.pardir)

import torch
from torch import nn
from utils.utils_func import construct_graph_batch
from torch_geometric.utils import to_dense_batch
from torch.distributions import Categorical
import torch_geometric as pyg
from torch_scatter import scatter
from itertools import chain


def matrix_list_to_graphs(lower_left_matrices, device):
    graphs = []
    #edge_candidates = []
    for b, lower_left_m in enumerate(lower_left_matrices):
        edge_indices = [[], []]
        edge_attrs = [] ###############################
        x = torch.ones(len(lower_left_m), 1)
        #edge_cand = {x: set() for x in range(len(lower_left_m))}
        for row, cols in enumerate(lower_left_m):
            for col, weight in enumerate(cols):
                # if weight == 0 or weight >= 2:
                #     pass
                # else:
                #     edge_indices[0].append(row)
                #     edge_indices[1].append(col)
                #     edge_attrs.append(weight)
                #     x[row] += weight
                #     x[col] += weight
                #     #edge_cand[row].add(col)
                #     #edge_cand[col].add(row)
                edge_indices[0].append(row)
                edge_indices[1].append(col)
                edge_attrs.append(weight)
                x[row] += weight
                x[col] += weight
        edge_indices = torch.tensor(edge_indices)
        edge_attrs = torch.Tensor(edge_attrs).to(device)  ####################
        #x = (x) / torch.std(x)
        # graphs.append(pyg.data.Data(x=x, edge_index=edge_indices)) #, edge_attrs=edge_attrs)) ############################
        graphs.append(pyg.data.Data(x=x, edge_index=edge_indices, edge_attrs=edge_attrs))  ##########################
        #edge_candidates.append(edge_cand)
    return graphs #, edge_candidates


class GCN(nn.Module):
    def __init__(self, num_in_feats, num_out_feats, num_layers=3, batch_norm=True):
        super(GCN, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_layers = num_layers

        for l in range(self.num_layers):
            if l == 0:
                conv = pyg.nn.GCNConv(self.num_in_feats, self.num_out_feats)
            else:
                conv = pyg.nn.GCNConv(self.num_out_feats, self.num_out_feats)
            if batch_norm:
                norm = nn.BatchNorm1d(self.num_out_feats)
            else:
                norm = nn.Identity()
            self.add_module('conv_{}'.format(l), conv)
            self.add_module('norm_{}'.format(l), norm)

        self.init_parameters()

    def init_parameters(self):
        for l in range(self.num_layers):
            # nn.init.xavier_uniform_(getattr(self, 'conv_{}'.format(l)).weight)
            getattr(self, 'conv_{}'.format(l)).reset_parameters()

    def forward(self, *args):
        if len(args) == 1 and isinstance(args[0], (pyg.data.Data, pyg.data.Batch)):
            x, edge_index = args[0].x, args[0].edge_index
        elif len(args) == 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            x, edge_index = args
        else:
            raise ValueError('Unknown combination of data types: {}'.format(','.join([type(x) for x in args])))
        for l in range(self.num_layers):
            conv = getattr(self, 'conv_{}'.format(l))
            norm = getattr(self, 'norm_{}'.format(l))
            x = conv(x, edge_index)
            x = nn.functional.relu(norm(x))
        return x


class GraphAttentionPooling(nn.Module):
    """
    Attention module to extract global feature of a graph.
    """
    def __init__(self, feat_dim):
        """
        :param feat_dim: number dimensions of input features.
        """
        super(GraphAttentionPooling, self).__init__()
        self.feat_dim = feat_dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.feat_dim, self.feat_dim))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, batch, size=None):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: Result of the GNN.
        :param batch: Batch vector, which assigns each node to a specific example
        :return representation: A graph level representation matrix.
        """
        size = batch[-1].item() + 1 if size is None else size
        mean = scatter(x, batch, dim=0, dim_size=size, reduce='mean')
        transformed_global = torch.tanh(torch.mm(mean, self.weight_matrix))

        coefs = torch.sigmoid((x * transformed_global[batch] * 10).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x

        return scatter(weighted, batch, dim=0, dim_size=size, reduce='add')

    def get_coefs(self, x):
        mean = x.mean(dim=0)
        transformed_global = torch.tanh(torch.matmul(mean, self.weight_matrix))

        return torch.sigmoid(torch.matmul(x, transformed_global))


class ResNetBlock(nn.Module):
    def __init__(self, num_in_feats, num_out_feats, num_layers=3, batch_norm=True):
        super(ResNetBlock, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_layers = num_layers

        self.first_linear = None
        self.last_linear = None
        self.sequential = []
        self.output_seq = []

        for l in range(self.num_layers):
            if l == 0:
                self.first_linear = nn.Linear(self.num_in_feats, self.num_out_feats)
                if batch_norm: self.sequential.append(nn.BatchNorm1d(self.num_out_feats))
                self.sequential.append(nn.ReLU())
            elif l == self.num_layers - 1:
                self.last_linear = nn.Linear(self.num_out_feats, self.num_out_feats)
                if batch_norm: self.output_seq.append(nn.BatchNorm1d(self.num_out_feats))
            else:
                self.sequential.append(nn.Linear(self.num_out_feats, self.num_out_feats))
                if batch_norm: self.sequential.append(nn.BatchNorm1d(self.num_out_feats))
                self.sequential.append(nn.ReLU())

        self.sequential = nn.Sequential(*self.sequential)
        self.output_seq = nn.Sequential(*self.output_seq)

        self.init_parameters()

    def init_parameters(self):
        for mod in chain(self.sequential, self.output_seq):
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)

    def forward(self, inp):
        x1 = self.first_linear(inp)
        x2 = self.sequential(x1) + x1
        return self.output_seq(x2)


class GraphEncoder(torch.nn.Module):
    def __init__(
            self,
            node_feature_dim,
            node_output_size,
            batch_norm,
            one_hot_degree,
            num_layers=10
    ):
        super(GraphEncoder, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.node_output_size = node_output_size
        self.one_hot_degree = one_hot_degree
        self.batch_norm = batch_norm
        self.num_layers = num_layers

        one_hot_dim = self.one_hot_degree + 1 if self.one_hot_degree > 0 else 0
        self.siamese_gcn = GCN(self.node_feature_dim + one_hot_dim, self.node_output_size, num_layers=self.num_layers, batch_norm=self.batch_norm)
        self.att = GraphAttentionPooling(self.node_output_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, inp_lower_matrix):
        # construct graph batches
        batched_graphs = construct_graph_batch(matrix_list_to_graphs(inp_lower_matrix, self.device), self.one_hot_degree, self.device)

        # forward pass
        batched_node_feat = self.siamese_gcn(batched_graphs)
        node_feat_reshape, _ = to_dense_batch(batched_node_feat, batched_graphs.batch)
        graph_feat = self.att(batched_node_feat, batched_graphs.batch)
        state_feat = torch.cat(
            (node_feat_reshape, graph_feat.unsqueeze(1).expand(-1, node_feat_reshape.shape[1], -1)), dim=-1)

        return state_feat


class ActorNet(torch.nn.Module):
    def __init__(
            self,
            state_feature_size,
            batch_norm,
    ):
        super(ActorNet, self).__init__()
        self.state_feature_size = state_feature_size
        self.batch_norm = batch_norm

        self.act1_resnet = ResNetBlock(self.state_feature_size, 1, batch_norm=self.batch_norm)
        #self.act2_resnet = ResNetBlock(self.state_feature_size * 2, 1, batch_norm=self.batch_norm)
        self.act2_query = nn.Linear(self.state_feature_size, self.state_feature_size, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_feat, edge_candidates, known_action=None):
        return self._act(input_feat, edge_candidates, known_action)

    def _act(self, input_feat, edge_candidates, known_action=None):
        if known_action is None:
            known_action = (None, None)
        # roll-out 2 acts
        mask1, ready_nodes1 = self._get_mask1(input_feat.shape[0], input_feat.shape[1], edge_candidates)
        act1, log_prob1, entropy1 = self._select_node(input_feat, mask1, known_action[0])
        mask2, ready_nodes2 = self._get_mask2(input_feat.shape[0], input_feat.shape[1], edge_candidates, act1)
        act2, log_prob2, entropy2 = self._select_node(input_feat, mask2, known_action[1], act1)
        return torch.stack((act1, act2)), torch.stack((log_prob1, log_prob2)), entropy1 + entropy2

    def _select_node(self, state_feat, mask, known_cur_act=None, prev_act=None, greedy_sel_num=0):
        # neural net prediction
        if prev_act is None:  # for act 1
            act_scores = self.act1_resnet(state_feat).squeeze(-1)
        else:  # for act 2
            prev_node_feat = state_feat[torch.arange(len(prev_act)), prev_act, :]
            #state_feat = torch.cat(
            #    (state_feat, prev_node_feat.unsqueeze(1).expand(-1, state_feat.shape[1], -1)), dim=-1)
            #act_scores = self.act2_resnet(state_feat).squeeze(-1)
            act_query = torch.tanh(self.act2_query(prev_node_feat))
            act_scores = (act_query.unsqueeze(1) * state_feat).sum(dim=-1)

        # select action
        if greedy_sel_num > 0:
            act_probs = nn.functional.softmax(act_scores + mask, dim=1)
            argsort_prob = torch.argsort(act_probs, dim=-1, descending=True)
            acts = argsort_prob[:, :greedy_sel_num]
            return acts, act_probs[torch.arange(acts.shape[0]).unsqueeze(-1), acts]
        else:
            act_log_probs = nn.functional.log_softmax(act_scores + mask, dim=1)
            dist = Categorical(logits=act_log_probs)
            if known_cur_act is None:
                act = dist.sample()
                return act, dist.log_prob(act), dist.entropy()
            else:
                return known_cur_act, dist.log_prob(known_cur_act), dist.entropy()

    def _get_mask1(self, batch_size, num_nodes, edge_candidates):
        masks = torch.full((batch_size, num_nodes), -float('inf'), device=self.device)
        ready_nodes = []
        for b in range(batch_size):
            ready_nodes.append([])
            for node, candidates in edge_candidates[b].items():
                if len(candidates) == 0:
                    pass
                else:
                    masks[b, node] = 0
                    ready_nodes[b].append(node)
        return masks, ready_nodes

    def _get_mask2(self, batch_size, num_nodes, edge_candidates, act1):
        masks = torch.full((batch_size, num_nodes), -float('inf'), device=self.device)
        ready_nodes = []
        for b in range(batch_size):
            ready_nodes.append([])
            candidates = edge_candidates[b][act1[b].item()]
            for index in candidates:
                masks[b, index] = 0.0
                ready_nodes[b].append(index)
        return masks, ready_nodes


class CriticNet(torch.nn.Module):
    def __init__(
            self,
            state_feature_size,
            batch_norm,
    ):
        super(CriticNet, self).__init__()
        self.state_feature_size = state_feature_size
        self.batch_norm = batch_norm

        self.critic_resnet = ResNetBlock(self.state_feature_size, 1, batch_norm=self.batch_norm)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, state_feat):
        return self._eval(state_feat)

    def _eval(self, state_feat):
        # get global features
        state_feat = torch.max(state_feat, dim=1).values
        state_value = self.critic_resnet(state_feat).squeeze(-1)
        return state_value


class ActorCritic(nn.Module):
    def __init__(self, node_feature_dim, node_output_size, batch_norm, one_hot_degree, gnn_layers):
        super(ActorCritic, self).__init__()
        self.state_encoder = GraphEncoder(node_feature_dim, node_output_size, batch_norm, one_hot_degree, gnn_layers)
        self.actor_net = ActorNet(node_output_size * 2, batch_norm)
        self.value_net = CriticNet(node_output_size * 2, batch_norm)

    def forward(self):
        raise NotImplementedError

    def act(self, inp_lower_matrix, edge_candidates, memory):
        state_feat = self.state_encoder(inp_lower_matrix)
        actions, action_logits, entropy = self.actor_net(state_feat, edge_candidates)

        memory.states.append(inp_lower_matrix)
        memory.edge_candidates.append(edge_candidates)
        memory.actions.append(actions)
        memory.logprobs.append(action_logits)

        return actions

    def evaluate(self, inp_lower_matrix, edge_candidates, action):
        state_feat = self.state_encoder(inp_lower_matrix)
        _, action_logits, entropy = self.actor_net(state_feat, edge_candidates, action)
        state_value = self.value_net(state_feat)
        return action_logits, state_value, entropy
