import torch
import random
import numpy as np
import torch.nn.functional as F


def padded_get_stats(y_preds, batch):
    _, x_edges_values, _ = batch.edges, batch.edges_values, batch.edges_target
    y_nodes = batch.tour_nodes
    pred_tour_len, gt_tour_len = [], []
    for i in range(len(y_preds)):
        bs_nodes = greedy_search(y_preds[i].clone().detach().unsqueeze(0))
        pred_tour_len.append(get_gt_tour(bs_nodes, x_edges_values[i]))
        if y_nodes[i].ndim == 2:
            gt_tour_len.append(get_gt_tour(torch.tensor(y_nodes[i]).long(), x_edges_values[i]))
        else:
            gt_tour_len.append(get_gt_tour(torch.tensor(y_nodes[i]).unsqueeze(0).long(), x_edges_values[i]))
    return torch.cat(pred_tour_len), torch.cat(gt_tour_len)


def mean_tour_len_edges(x_edges_values, y_pred_edges):
    """
    Computes mean tour length for given batch prediction as edge adjacency matrices.
    """
    y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
    y = y.argmax(dim=3)  # B x V x V
    # Divide by 2 because edges_values is symmetric
    tour_lens = (y.float() * x_edges_values.float()).sum(dim=1).sum(dim=1) / 2
    mean_tour_len = tour_lens.sum().to(dtype=torch.float).item() / tour_lens.numel()
    return mean_tour_len


def unroll(batch):
    x_edges = torch.LongTensor(batch.edges).type(torch.cuda.LongTensor)
    x_edges_values = torch.FloatTensor(batch.edges_values).type(torch.cuda.FloatTensor)
    x_nodes = torch.LongTensor(batch.nodes).type(torch.cuda.LongTensor)
    x_nodes_coord = torch.FloatTensor(batch.nodes_coord).type(torch.cuda.FloatTensor)
    y_edges = torch.LongTensor(batch.edges_target).type(torch.cuda.LongTensor)
    y_nodes = torch.LongTensor(batch.nodes_target).type(torch.cuda.LongTensor)

    # careful! y_nodes is the position of node i in the final tour solution
    route_decoded = []
    for i in range(len(y_nodes)):
        values, idx = torch.sort(y_nodes[i])
        route_decoded.append(torch.arange(len(y_nodes[i]))[idx])
    route_decoded = torch.stack(route_decoded)

    return x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, route_decoded


def update_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def greedy_search(y_pred_edges):
    y = F.softmax(y_pred_edges, dim=3)
    y = y[:, :, :, 1]
    # make self-loop infeasible
    y = y - torch.eye(y[0].shape[0]).unsqueeze(0).cuda() * 10
    visited = torch.zeros((y.shape[0], y.shape[1])).bool().cuda()
    visited[:, 0] = True
    rows = y[:, 0]
    tours = torch.zeros((y.shape[0], 1)).cuda()
    
    while not torch.all(visited):
        vals, idx = torch.sort(rows, descending=True)
        mask_resorted = torch.gather(visited, 1, idx)
        next_ids = torch.reshape(idx[~mask_resorted], (y.shape[0], -1))[:, 0]

        for i, ix in enumerate(next_ids):
            assert not visited[i, ix]
            visited[i, ix] = True
            rows[i] = y[i][ix]
        tours = torch.cat((tours, next_ids.unsqueeze(-1)), dim=1)
    return tours.int()


def get_gt_tour(routes, adj):
    costs = []
    for i in range(len(routes)):
        adj = adj[i] if adj.ndim == 3 else adj
        route_pairs = list(zip(routes[i].tolist(), routes[i][1:].tolist()+routes[i][:1].tolist()))
        cost = 0
        for (x, y) in route_pairs:
            cost += adj[x, y]
        costs.append(cost)
    return torch.Tensor(costs)


def check_constraints(adv_coo, coordinates, x, y, subspaces=None, return_difference=False):
    n_points = None
    if isinstance(x, list):
        n_points = [len(x_i) for x_i in x]
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)

    # 1. we assume inserting z between p and q: w(p, z) + w(q, z) - w(p, q)
    detour = (torch.norm(x - adv_coo, dim=-1) + torch.norm(y - adv_coo, dim=-1) - torch.norm(x - y, dim=-1))

    # 2. get all the node pairs (a and b) to be compared with (p and q), following proposition 2
    counter, ap = 0, []
    for i, c in enumerate(coordinates):
        if subspaces is None:
            all_pairs = torch.tril_indices(len(c), len(c), offset=-1)
        else:
            triu = torch.triu(torch.ones(len(c), len(c)), diagonal=1)  # upper triangular matrix

            # Remove X and Y (captured by `detour` variable)
            if subspaces is not None:
                ss_exclude_idx = torch.tensor(subspaces[i][2]).T.sort(dim=0).values
                triu[ss_exclude_idx[0], ss_exclude_idx[1]] = 0

            all_pairs = triu.nonzero().T

        ap.append(all_pairs + counter)
        counter += len(c)

    lengths = [len(a[0]) for a in ap]

    if n_points is not None:
        lengths = [l for l, n in zip(lengths, n_points) for _ in range(n)]
        ap = [a for a, n in zip(ap, n_points) for _ in range(n)]
    ap = torch.cat(ap, dim=1)
    coordinates = torch.cat([torch.Tensor(c) for c in coordinates])
    adv_coo_rep = torch.repeat_interleave(adv_coo, torch.Tensor(lengths).long(), dim=0)

    # 3. calculate w(a, z) + w(b, z) - w(a, b)
    # - only for common (a, b) e.g., for TSP100 with 5 inserted adv nodes, have (4950-5) common node pairs instead of 4949
    all_detours = (
      torch.norm(coordinates[ap[0]] - adv_coo_rep, dim=1)
      + torch.norm(coordinates[ap[1]] - adv_coo_rep, dim=1)
      - torch.norm(coordinates[ap[0]] - coordinates[ap[1]], dim=1)
    )
    
    all_detours = torch.split(all_detours, lengths)
    constraint_violated = [(all_detours[i] <= detour[i]).any().item() for i in range(len(adv_coo))]

    # 4. check missing (a, b) in the last step - (5-1) node paris
    if n_points is not None:
        chained_range = [i for n in n_points for i in range(n)]
        extra_indices = [j - torch.tril_indices(n+1, n+1, offset=-1) for j, n in enumerate(chained_range)]
        lengths = [idx.shape[1] for idx in extra_indices]
        adv_coo_rep_2 = torch.repeat_interleave(adv_coo, torch.Tensor(lengths).long(), dim=0)
        extra_indices = torch.cat(extra_indices, dim=1)

        extra_detours = (
            torch.norm(x[extra_indices[0]] - adv_coo_rep_2, dim=1)
            + torch.norm(y[extra_indices[1]] - adv_coo_rep_2, dim=1)
            - torch.norm(x[extra_indices[0]] - y[extra_indices[1]], dim=1)
        )
        extra_detours = torch.split(extra_detours, lengths)

        extra_constraint_violated = [(extra_detours[i] <= detour[i]).any().item() for i in range(len(adv_coo))]
        constraint_violated = [c1 or c2 for c1, c2 in zip(constraint_violated, extra_constraint_violated)]

    is_satisfied = [not cv for cv in constraint_violated]
    if not return_difference:
        return is_satisfied
    else:
        if n_points is None or len(extra_detours[i]) == 0:
            return is_satisfied, [detour[i] - all_detours[i].min() for i in range(len(adv_coo))]
        else:
            return is_satisfied, [
                (detour[i] - torch.min(all_detours[i].min(), extra_detours[i].min()) 
                 if len(extra_detours[i])
                 else detour[i] - all_detours[i].min())
                for i in range(len(adv_coo))
            ]


def add_to_route(route, pairs):
    found_count = 0
    if not isinstance(pairs[0], list):
        pairs = [pairs]
    route = route[:]
    for pair in pairs:
        for k, elem in enumerate(route):
            if elem == pair[0] and route[k-1] == pair[1]:
                route = route[:k] + [(max(route)+1)] + route[k:]
                found_count += 1
            elif elem == pair[1] and route[k-1] == pair[0]:
                route = route[:k] + [(max(route)+1)] + route[k:]
                found_count += 1
    assert found_count == len(pairs)
    return route


def find_subspace(points, routes, n_subspaces=1):
    possible_solutions = []
    for r in range(0, len(routes)):
        possible_solutions.append([routes[r-1], routes[r]])
    possible_solutions = np.array(possible_solutions)
    solset = possible_solutions.tolist()
    assert len(solset) > 0
    random.shuffle(solset)
    solset = solset[0: min(n_subspaces, len(solset))]
    point_1 = [points[s[0]] for s in solset]
    point_2 = [points[s[1]] for s in solset]
    return point_1, point_2, solset


def scale_coordinates(coos, sqrt=False):
    coos_resh = torch.flatten(coos, start_dim=1, end_dim=2)
    minimum, _ = torch.min(coos_resh, dim=-1)
    maximum, _ = torch.max(coos_resh, dim=-1)
    coos = (coos - minimum.view(-1, 1, 1)) / (maximum.view(-1, 1, 1) - minimum.view(-1, 1, 1))
    assert torch.all(coos <= 1)
    assert torch.all(coos >= 0)
    if sqrt:
        coos = coos * torch.sqrt(torch.Tensor([2]))/2
        assert torch.all(coos <= (torch.sqrt(torch.Tensor([2]))/2)[0].item())
        assert torch.all(coos >= 0)
    return coos


def adj_from_coordinates(coo):
    # shape of adj: batch x nodes x nodes 
    row_vec = coo.unsqueeze(1).repeat(1, coo.shape[1], 1, 1)
    col_vec = coo.unsqueeze(-1).repeat(1, 1, 1, coo.shape[1]).transpose(-1, -2)
    res = row_vec - col_vec
    adjs = torch.norm(res, p=2, dim=-1)
    return adjs   
