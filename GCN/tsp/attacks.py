import copy
import torch
import itertools
import numpy as np
import torch.nn as nn
from tsp.utils import *


def attack_rand(batch, model, num_neighbors=20, allowed_tries=50, **attackargs):
    """
    Random attack on TSP coordinates

    Keyword arguments:
    batch -- TSP data w/ coordinates of graph
    model -- neural TSP solver
    allowed_tries -- max. number of samples to find valid point
    """
    return attack_opt(batch, model, steps=1, allowed_tries=allowed_tries, **attackargs)


def attack_opt(batch, model, num_neighbors=20, steps=500, lr=0.01, sqrt=False, allowed_tries=50,
               gd_project_lr=2e-3, gd_project_steps=3, n_subspaces=5, **attackargs):
    """
    Optimized attack on TSP coordinates

    Keyword arguments:
    batch -- TSP data w/ coordinates of graph
    model -- neural TSP solver
    steps -- number of attack steps
    lr -- attack learning rate
    sqrt -- scale coordinates to unit square or 1/sqrt(2) square
    allowed_tries -- max. number of samples to find valid inital point
    gd_project_lr, gd_project_steps -- constraint GD parameters
    n_subspaces -- max. number of nodes to add during attack
    """
    model.train()

    # collect subspaces and keep track of instances where the route never meets the convex hull
    subspaces = []
    pos = copy.deepcopy(batch['coords'])
    for i, x in enumerate(pos):
        pos[i] = torch.reshape(torch.Tensor(x), (-1, 2))
    for k in range(len(pos)):
        p1, p2, sidx = find_subspace(pos[k], batch['routes'][k], n_subspaces=n_subspaces)
        assert len(sidx) > 0
        subspaces.append((p1, p2, sidx))
    p1s = [torch.stack(s[0]) for s in subspaces]
    p2s = [torch.stack(s[1]) for s in subspaces]
    
    # consider only samples where route passes convex hull
    coordinates = copy.deepcopy(batch['coords'])
    orig_routes = copy.deepcopy(batch['routes'])
    if 'og_batch' in batch.keys():
        batch = batch['og_batch']

    # init early stopping with clean instances
    # logits: list [[V x V x voc_edges_out], ...] loss: [B]
    logits, loss = model(batch, return_loss=True)
    best_loss = -loss
    best_coos = [torch.empty((0, 2)) for _ in coordinates]
    best_point_found = torch.tensor([False for _ in range(sum([len(s[0]) for s in subspaces]))])
    best_point_found_for_route = torch.tensor([False for _ in range(len(coordinates))])

    # sample points
    adv_coos, point_found = find_random_points(coordinates, p1s, p2s, allowed_tries=allowed_tries, subspaces=subspaces)
    points_per_route = [len(s[0]) for s in subspaces]
    pfr_idx = torch.cat((torch.tensor([0]), torch.tensor(points_per_route).cumsum(dim=0)))
    point_found_for_route = [m.any().item() for m in torch.split(point_found, points_per_route)]

    adv_coos.requires_grad = True  # [B x n_subspaces, 2]
    optimizer = torch.optim.Adam([adv_coos], lr=lr)

    for _ in range(steps):
        # print(point_found)
        # add new point to route where valid point was found
        routes = copy.deepcopy(orig_routes)
        pf_split = torch.split(point_found, points_per_route)
        for k, r in enumerate(routes):
            if point_found_for_route[k]:
                new_waypoints = list(itertools.compress(subspaces[k][2], pf_split[k]))
                routes[k] = np.array(add_to_route(r, new_waypoints))
            else:
                routes[k] = np.array(routes[k])

        # add optimized points and scale coordinates
        # Note that: only new nodes that satisfy the constraint have gradient!
        new_points = torch.split(adv_coos[point_found], [pfs.sum().item() for pfs in pf_split])
        all_coords = [torch.cat((po, np)) for po, np in zip(coordinates, new_points)]
        all_coords = [scale_coordinates(co.unsqueeze(0), sqrt=sqrt).squeeze(0) for co in all_coords]  # do we need this line?

        new_batch = model.reconstruct(all_coords, batch, routes, num_neighbors)
        logits, loss = model(new_batch, return_loss=True)
        loss = -loss
        # print(best_loss, loss)

        # early stopping - cache before optimization since the optimized nodes may even violate the constraint
        with torch.no_grad():
            new_best_loss = best_loss > loss
            if new_best_loss.any():
                advc_split = torch.split(adv_coos, points_per_route)
                best_coos = [advc_split[k].clone() if new_best_loss[k] else b for k, b in enumerate(best_coos)]
                mask_bpf = new_best_loss.repeat_interleave(torch.Tensor(points_per_route).long().cuda())
                best_point_found[mask_bpf] = point_found[mask_bpf]
                best_point_found_for_route[new_best_loss] = torch.tensor(point_found_for_route)[new_best_loss]
                best_loss[new_best_loss] = loss[new_best_loss]

        # optimize new nodes that satisfy the constraint
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # optimize new nodes that violate the constraint
        if gd_project_lr:
            adv_coos.data, point_found = gradient_projection(adv_coos, coordinates, p1s, p2s, subspaces, lr=gd_project_lr, max_steps=gd_project_steps)
            point_found_for_route = [m.any().item() for m in torch.split(point_found, points_per_route)]

    # create final output
    with torch.no_grad():
        # add new point to route where valid point was found
        routes = [r for r in orig_routes]
        pf_split = torch.split(best_point_found, points_per_route)
        for k, r in enumerate(routes):
            if best_point_found_for_route[k]:
                new_waypoints = list(itertools.compress(subspaces[k][2], pf_split[k]))
                routes[k] = np.array(add_to_route(r, new_waypoints))
            else:
                routes[k] = np.array(routes[k])
        
        # add optimized points and scale coordinates
        best_coordinates = []
        for k in range(len(coordinates)):
            if best_point_found_for_route[k]:
                new_coordinates = torch.stack([best_coos[k][p] for p in range(points_per_route[k]) if best_point_found[pfr_idx[k] + p]])
                coos = torch.cat((coordinates[k], new_coordinates)).squeeze(0)
                coos = scale_coordinates(coos.unsqueeze(0), sqrt=False).squeeze(0) 
            else:
                coos = coordinates[k]
            best_coordinates.append(coos)
        new_batch = model.reconstruct(best_coordinates, batch, routes, num_neighbors)

    return new_batch


def gradient_projection(opt_coos, coordinates, p1s, p2s, subspaces, lr=2e-3, max_steps=3):
    opt_coos = opt_coos.detach()

    for _ in range(max_steps):
        opt_coos.requires_grad = True
        const_check, differences = check_constraints(opt_coos, coordinates, p1s, p2s, subspaces=subspaces, return_difference=True)
        const_check = torch.tensor(const_check).bool()

        if const_check.all():
            break

        grad = torch.autograd.grad(differences, opt_coos, retain_graph=False, create_graph=False)[0]
        opt_coos = opt_coos.detach()
        opt_coos[~const_check] = opt_coos[~const_check] - lr * grad[~const_check]

    return opt_coos, const_check


def find_random_points(coordinates, p1s, p2s, allowed_tries=10, subspaces=None, keep_coo=None, keep_mask=None):
    if isinstance(p1s, list):
        p1s_tensor = torch.cat(p1s, dim=0)
        p2s_tensor = torch.cat(p2s, dim=0)
        inits = torch.zeros_like(p1s_tensor)
        inits[:, 0] = (p1s_tensor[:, 0] + p2s_tensor[:, 0]) / 2
        inits[:, 1] = (p1s_tensor[:, 1] + p2s_tensor[:, 1]) / 2
    else:
        inits = torch.zeros_like(p1s)
        inits[:, 0] = (p1s[:, 0] + p2s[:, 0]) / 2
        inits[:, 1] = (p1s[:, 1] + p2s[:, 1]) / 2
    point_found = torch.zeros((inits.shape[0],)).bool()
    add_points = torch.zeros((inits.shape))

    if keep_coo is not None and keep_mask is not None:
        point_found = keep_mask
        add_points[keep_mask] = keep_coo[keep_mask]
        
    num_tries = 0
    while not torch.all(point_found):
        num_tries += 1
        if num_tries >= allowed_tries:
            break
            
        noise = (torch.rand(inits[~point_found].shape) - 0.5) / 5
        add_points[~point_found] = inits[~point_found] + noise
        point_found = torch.Tensor(check_constraints(add_points, coordinates, p1s, p2s, subspaces=subspaces)).bool()
    return add_points, point_found
