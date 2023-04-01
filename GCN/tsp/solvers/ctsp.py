"""
File taken from:
https://github.com/chaitjo/graph-convnet-tsp
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tsp.utils import *
from sklearn.utils.class_weight import compute_class_weight
from tsp.data import DotDict
from tsp.utils import adj_from_coordinates


def get_convtsp_model(seed):
    config = {'voc_nodes_in': 2, 'voc_nodes_out': 2, 
              'voc_edges_in': 3, 'voc_edges_out': 2, 
              'beam_size': 1280, 'hidden_dim': 300, 
              'num_layers': 30, 'mlp_layers': 3, 
              'aggregation': 'mean', 'num_nodes': 100,
              'node_dim': 2, 'num_neighbors': -1, 'batch_size': 20}
    model = ConvTSP(config, torch.cuda.FloatTensor, torch.cuda.LongTensor, seed)
    return model, config


class ConvTSP(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.

    References:
        Paper: https://arxiv.org/pdf/1711.07553v2.pdf
        Code: https://github.com/xbresson/spatial_graph_convnets
    """

    def __init__(self, config, dtypeFloat, dtypeLong, seed):
        super(ConvTSP, self).__init__()
        torch.manual_seed(seed)
        self.name = "ConvTSP"
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        # Define net parameters
        self.num_nodes = config['num_nodes']
        self.node_dim = config['node_dim']
        self.voc_nodes_in = config['voc_nodes_in']
        self.voc_nodes_out = config['num_nodes']  # config['voc_nodes_out']
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = config['voc_edges_out']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.mlp_layers = config['mlp_layers']
        self.aggregation = config['aggregation']
        # Node and edge embedding layers/lookups
        self.nodes_coord_embedding = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim//2, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim//2)
        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)
        # self.mlp_nodes = MLP(self.hidden_dim, self.voc_nodes_out, self.mlp_layers)

    def forward(self, batch, return_loss=True):
        if not isinstance(batch.edges_values, list):
            x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, _ = unroll(batch)
            x_edges = list(torch.split(x_edges, 1))
            x_edges_values = list(torch.split(x_edges_values, 1))
            x_nodes_coord = list(torch.split(x_nodes_coord, 1))
            x_nodes = list(torch.split(x_nodes, 1))
            y_edges = list(torch.split(y_edges, 1))
        else:
            x_edges, x_edges_values, y_edges = batch.edges, batch.edges_values, batch.edges_target
            x_nodes, x_nodes_coord = batch.nodes, batch.nodes_coord

        og_num_nodes = [x.size(1) for x in x_nodes_coord]
        max_nodes = max(og_num_nodes) + 1
        cw, mask = [], []
        for sample_id in range(len(x_edges)):
            nn = x_edges[sample_id].size(1)
            assert x_edges_values[sample_id].size(1) == nn
            assert x_nodes_coord[sample_id].size(1) == nn
            assert y_edges[sample_id].size(1) == y_edges[sample_id].size(2) == nn
            edge_labels = y_edges[sample_id].cpu().numpy().flatten()
            cw.append(compute_class_weight("balanced", classes=np.array([0.0, 1.0]), y=edge_labels))
            m = torch.zeros(max_nodes)
            m[:nn] = 1
            mask.append(m)
            if nn < max_nodes:
                add = max_nodes - nn
                x_edges[sample_id] = torch.cat((torch.cat((x_edges[sample_id].cuda(), torch.full((1, nn, add), 2).cuda()), dim=2),
                                                torch.full((1, add, max_nodes), 2).cuda()), dim=1)
                x_edges_values[sample_id] = torch.cat((torch.cat((x_edges_values[sample_id], torch.full((1, nn, add), 2).cuda()), dim=2), 
                                                    torch.full((1, add, max_nodes), 2).cuda()), dim=1)
                x_nodes_coord[sample_id] = torch.cat((x_nodes_coord[sample_id], torch.zeros(1, add, 2).cuda()), dim=1)
                y_edges[sample_id] = torch.cat((torch.cat((y_edges[sample_id], torch.full((1, nn, add), 0).cuda()), dim=2), 
                                                torch.full((1, add, max_nodes), 0).cuda()), dim=1)
        mask = torch.stack(mask).cuda()
        x_edges = torch.cat(x_edges, dim=0)
        x_edges_values = torch.cat(x_edges_values, dim=0)
        x_nodes_coord = torch.cat(x_nodes_coord, dim=0)
        y_edges = torch.cat(y_edges, dim=0)

        y_pred, loss = self.forward_unpadded(x_edges.long(), x_edges_values, x_nodes, x_nodes_coord, y_edges, cw, mask=mask)
        ys = []
        for sample_id in range(len(x_edges)):
            ys.append(y_pred[sample_id][mask[sample_id].bool()][:, mask[sample_id].bool(), ...])
        if return_loss:
            return ys, loss
        else:
            return ys

    def forward_unpadded(self, x_edges, x_edges_values, x_nodes, x_nodes_coord,
                y_edges, edge_cw, postprocess=False, mask=None):
        """
        Args:
            x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
            x_nodes: Input nodes (batch_size, num_nodes)
            x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim)
            y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
            edge_cw: Class weights for edges loss
            # y_nodes: Targets for nodes (batch_size, num_nodes, num_nodes)
            # node_cw: Class weights for nodes loss

        Returns:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
            # y_pred_nodes: Predictions for nodes (batch_size, num_nodes)
            loss: Value of loss function
        """
        if mask is None:
            mask = torch.ones_like(y_edges)

        # Node and edge embedding
        x = self.nodes_coord_embedding(x_nodes_coord)  # B x V x H
        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # B x V x V x H
        e_tags = self.edges_embedding(x_edges)  # B x V x V x H
        e = torch.cat((e_vals, e_tags), dim=3)
        # GCN layers
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x, e, mask=mask)  # B x V x H, B x V x V x H
        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # B x V x V x voc_edges_out
        # y_pred_nodes = self.mlp_nodes(x)  # B x V x voc_nodes_out
        if postprocess:
            return y_pred_edges
        # Compute loss
        edge_cw = torch.Tensor(edge_cw).type(self.dtypeFloat)  # Convert to tensors
        loss = loss_edges(y_pred_edges, y_edges.long(), edge_cw, mask)
        return y_pred_edges, loss

    def reconstruct(self, all_coords, batch, routes, num_neighbors=20):
        edges_values = [adj_from_coordinates(xc.unsqueeze(0)).cuda() for xc in all_coords]
        nodes = [torch.ones(xc.shape[:2]).int() for xc in all_coords]

        if num_neighbors == -1:
            edges = [(torch.ones(xe.shape) + torch.eye(xe.shape[1])).cuda() for xe in edges_values]
        else:
            # construct adjacency matrix use KNN (if num_neighbors != -1)
            edges = [np.zeros((xe.size(-1), xe.size(-1))) for xe in edges_values]
            for i, edge in enumerate(edges):
                knns = np.argpartition(edges_values[i].cpu().detach().squeeze(0).numpy(), kth=num_neighbors, axis=-1)[:, num_neighbors::-1]
                for idx in range(edges_values[i].size(-1)):
                    edge[idx][knns[idx]] = 1
                np.fill_diagonal(edge, 2)  # Special token for self-connections
            edges = [torch.Tensor(edge).unsqueeze(0).cuda() for edge in edges]

        y_edges = []
        for i, route in enumerate(routes):
            edges_target = torch.zeros(edges_values[i].size(1), edges_values[i].size(1)).cuda()
            for idx in range(len(route) - 1):
                i = route[idx]
                j = route[idx + 1]
                edges_target[i][j] = 1
                edges_target[j][i] = 1
            
            # Add final connection of tour in edge target
            edges_target[j][route[0]] = 1
            edges_target[route[0]][j] = 1
            y_edges.append(edges_target.unsqueeze(0))

        batch = DotDict()
        batch.edges = edges
        batch.edges_values = edges_values
        batch.edges_target = y_edges
        batch.nodes = nodes
        batch.tour_nodes = routes
        batch.nodes_coord = [a.unsqueeze(0).cuda() for a in all_coords]
            
        return batch


def loss_edges(y_pred_edges, y_edges, edge_cw, mask=None):
    """
    Loss function for edge predictions.

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
        edge_cw: Class weights for edges loss

    Returns:
        loss_edges: Value of loss function
    
    """
    # Edge loss
    if mask is not None:
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    else:
        square_mask = torch.ones_like(y_edges)
    square_cw = edge_cw[torch.arange(y_pred_edges.shape[0])[:, None, None], y_edges]
    # Edge loss
    y = y_pred_edges.permute(0, 3, 1, 2)  # B x voc_edges x V x V
    loss_edges = square_mask * nn.CrossEntropyLoss(reduction='none')(y, y_edges)  # assumes  B x C x d1 x d2 x ... x dk
    loss_edges = square_cw * loss_edges
    return torch.sum(loss_edges, dim=(1, 2)) / (square_cw * square_mask).sum([-1, -2])


class BatchNormNode(nn.Module):
    """Batch normalization for node features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x, mask):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)

        Returns:
            x_bn: Node features after batch normalization (batch_size, num_nodes, hidden_dim)
        """
        e_bn_1d = x.clone()
        e_bn_1d[torch.where(mask)] = self.batch_norm(e_bn_1d[torch.where(mask)])
        return e_bn_1d


class BatchNormEdge(nn.Module):
    """Batch normalization for edge features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        # self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, e, mask):
        """
        Args:
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_bn: Edge features after batch normalization (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        e_bn_1d = e.clone()
        e_bn_1d[torch.where(square_mask)] = self.batch_norm(e_bn_1d[torch.where(square_mask)])
        return e_bn_1d


class NodeFeatures(nn.Module):
    """Convnet features for nodes.
    
    Using `sum` aggregation:
        x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]
    
    Using `mean` aggregation:
        x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
    """
    
    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, edge_gate, mask):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
        Ux = mask.unsqueeze(-1) * self.U(x)  # B x V x H
        Vx = mask.unsqueeze(-1) * self.V(x)  # B x V x H
        Vx = Vx.unsqueeze(1)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        if mask is not None:
            edge_gate = edge_gate * mask.unsqueeze(-1).unsqueeze(-1)
            edge_gate = edge_gate * mask.unsqueeze(-1).unsqueeze(1)
        gateVx = edge_gate * Vx  # B x V x V x H
        if self.aggregation == "mean":
            x_new = Ux + torch.sum(gateVx, dim=2) / (1e-20 + torch.sum(edge_gate, dim=2))  # B x V x H
        elif self.aggregation == "sum":
            x_new = Ux + torch.sum(gateVx, dim=2)  # B x V x H
        return x_new


class EdgeFeatures(nn.Module):
    """Convnet features for edges.

    e_ij = U*e_ij + V*(x_i + x_j)
    """

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)
        
    def forward(self, x, e, mask):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        Ue = self.U(e)
        Vx = mask.unsqueeze(-1) * self.V(x)
        Wx = Vx.unsqueeze(1)  # Extend Vx from "B x V x H" to "B x V x 1 x H"
        Vx = Vx.unsqueeze(2)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        e_new = Ue + Vx + Wx
        return e_new


class ResidualGatedGCNLayer(nn.Module):
    """Convnet layer with gating and residual connection.
    """

    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e, mask=None):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_in = e
        x_in = x
        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in, mask)  # B x V x V x H
        # Compute edge gates
        edge_gate = torch.sigmoid(e_tmp)
        # Node convolution
        x_tmp = self.node_feat(x_in, edge_gate, mask)
        # Batch normalization
        e_tmp = self.bn_edge(e_tmp, mask)
        x_tmp = self.bn_node(x_tmp, mask)
        # ReLU Activation
        e = F.relu(e_tmp)
        x = F.relu(x_tmp)
        # Residual connection
        x_new = x_in + x
        e_new = e_in + e
        return x_new, e_new


class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)

        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)  # B x H
            Ux = F.relu(Ux)  # B x H
        y = self.V(Ux)  # B x O
        return y
