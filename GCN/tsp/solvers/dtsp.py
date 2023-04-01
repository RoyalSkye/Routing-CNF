import copy
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear, ReLU


class DTSP(torch.nn.Module):
    def __init__(self, dim=64, seed=0, num_mp=32, device="cuda"):
        super(DTSP, self).__init__() 
        
        torch.manual_seed(seed)
        self.name = "DTSP"
        self.device = device
        self.dim = dim
        self.num_mp = num_mp

        self.init_mlp = Seq(Linear(2, int(dim/8)), ReLU(), Linear(int(dim/8), int(dim/4)), 
                            ReLU(), Linear(int(dim/4), int(dim/2)), ReLU(), Linear(int(dim/2), dim))

        self.edge_message_mlp = Seq(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(), Linear(dim, dim))             
        self.vertex_message_mlp = Seq(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.edge_vote_mlp = Seq(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(), Linear(dim, 1))
        
        self.edge_vote_mlp.apply(initialize)
        self.edge_message_mlp.apply(initialize)
        self.vertex_message_mlp.apply(initialize)
        self.init_mlp.apply(initialize)
        
        self.v_init = torch.nn.Parameter(torch.normal(0, 1, (1, dim)))
        self.lstm_v = torch.nn.LSTM(dim, dim)
        self.lstm_e = torch.nn.LSTM(dim, dim)

    def forward(self, batch, return_loss=False):
        EV = batch['EV'].to(self.device)
        E = self.init_mlp(torch.cat([batch['W'].to(self.device), batch['C'].to(self.device)], dim=1)).unsqueeze(0)
        V = torch.div(self.v_init.to(self.device), 
                      torch.sqrt(torch.Tensor([self.dim]).to(self.device))).repeat(
                                                         (EV.shape[1], 1)).unsqueeze(0)
        
        c_state_e = torch.zeros_like(E).to(self.device)
        c_state_v = torch.zeros_like(V).to(self.device)
        
        for _ in range(self.num_mp):
            # input to LSTM has size: seq_len, batch, input_size
            _, (V, c_state_v) = self.lstm_v((EV.T @ self.edge_message_mlp(E)), (V, c_state_v))
            _, (E, c_state_e) = self.lstm_e((EV @ self.vertex_message_mlp(V)), (E, c_state_e))
            
        votes = self.edge_vote_mlp(E.squeeze(0))
        votes = torch.split(votes, batch['n_edges'].tolist())
        votes = torch.cat([torch.mean(x).unsqueeze(0) for x in votes])
        
        if return_loss:
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
            loss = loss_fn(votes, batch['target'].cuda())
            return votes, loss
        return votes


    def reconstruct(self, all_coords, batch, new_routes):
        new_batch = copy.deepcopy(batch)

        # recalculate adjacency from new points and reformat
        adjs = [adj_from_coordinates(all_coords[k].unsqueeze(0)).squeeze(0) for k in range(len(all_coords))]
        assert all([adjs[k].shape[0] == len(new_routes[k]) for k in range(len(adjs))])
        adjs = [(torch.triu((a > 0).float()), a, new_routes[k].tolist(), None, None) for k, a in enumerate(adjs)]

        # Create "batch" and calc loss
        new_batch['EV'], new_batch['W'], new_batch['C'] = create_batch(adjs, labels=batch['target'])
        new_batch['n_edges'] = np.array([a[0].sum().item() for a in adjs]).astype(int)
        return new_batch


    def collate_fn(self, instances_single):
        # every item exists twice, but with different label
        instances = []
        for item in instances_single:
            instances.extend([item, item])
        n_instances = len(instances)
            
        # n_vertices[i]: number of vertices in the i-th instance
        n_vertices = np.array([x[0].shape[0] for x in instances])
        # n_edges[i]: number of edges in the i-th instance
        n_edges = np.array([len(np.nonzero(x[0])[0]) for x in instances])

        # Even index instances are UNSAT, odd are SAT
        route_exists = np.array([i % 2 for i in range(n_instances)])
        routes, fnames, positions = [], [], []

        for (i, (_, _, route, pos, fname)) in enumerate(instances):
            routes.append(route)
            fnames.append(fname)
            positions.append([float(x) for x in pos])
            positions[i] = torch.reshape(torch.Tensor(positions[i]), (-1, 2))

        EV, W, C = create_batch(instances, route_exists)

        batch = {
            "EV": EV,
            "W": W,
            "C": C,
            "target": torch.Tensor(route_exists),
            "n_vertices": n_vertices,
            "n_edges": n_edges, 
            "routes": routes, 
            "fnames": fnames, 
            "coords": positions
        }
        return batch


def initialize(l):
    if type(l) == Linear:
        torch.nn.init.xavier_uniform_(l.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.zeros_(l.bias)


def create_batch(instances, labels, dev=0.02):
    if not isinstance(instances[0], torch.Tensor):
        instances = [(torch.Tensor(Ma), torch.Tensor(Mw), r, p, f) for (Ma, Mw, r, p, f) in instances]

    n_vertices = torch.Tensor([x[0].shape[0] for x in instances]).long()
    n_edges = torch.Tensor([len(torch.nonzero(x[0])) for x in instances]).long()
    total_vertices = sum(n_vertices).int().item()
    total_edges = sum(n_edges).int().item()
    
    EV = list(torch.split(torch.zeros((total_edges, total_vertices)).cuda(), n_edges.int().tolist()))
    W = list(torch.split(torch.zeros((total_edges, 1)).cuda(), n_edges.int().tolist()))
    C = torch.zeros((total_edges, 1)).cuda()

    for (i, (Ma, Mw, route, _, _)) in enumerate(instances):
        assert isinstance(route, list)
        n, m = n_vertices[i], n_edges[i]
        n_acc = sum(n_vertices[0:i])
        m_acc = sum(n_edges[0:i])

        # Get the list of edges in this graph
        edges = torch.Tensor(list(zip(torch.nonzero(Ma)[:, 0], torch.nonzero(Ma)[:, 1]))).long().cuda()

        # Populate EV, W and edges_mask
        W[i] = Mw[edges[:, 0], edges[:, 1]].unsqueeze(-1)
        replacement = torch.ones(edges.shape).cuda()
        EV[i].scatter_(1, edges+n_acc, replacement)
        assert torch.all(EV[i].sum(dim=1) == 2.0)
        
        cost = sum([Mw[min(x, y), max(x, y)] for (x, y) in zip(route, route[1:]+[route[0]])]) / n
        assert len(list(zip(route, route[1:]+[route[0]]))) == len(route)
        C[m_acc:m_acc+m.int().item(), 0] = (1+dev)*cost if labels[i] else (1-dev)*cost
        
    EV = torch.cat(EV)
    W = torch.cat(W)
    return EV, W, C

        
def adj_from_coordinates(coo):
    row_vec = coo.unsqueeze(1).repeat(1, coo.shape[1], 1, 1)
    col_vec = coo.unsqueeze(-1).repeat(1, 1, 1, coo.shape[1]).transpose(-1, -2)
    res = row_vec - col_vec
    adjs = torch.norm(res, p=2, dim=-1)
    return adjs
