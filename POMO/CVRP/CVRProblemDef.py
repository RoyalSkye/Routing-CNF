
import torch
import numpy as np


def get_random_problems(batch_size, problem_size):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 200:
        demand_scaler = 70
    else:
        raise NotImplementedError

    node_demand = torch.Tensor(np.random.randint(1, 10, size=(batch_size, problem_size)))  # (unnormalized) shape: (batch, problem)
    capacity = torch.Tensor(np.full(batch_size, demand_scaler))

    # node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand, capacity


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data


def generate_x_adv(model, nat_data, eps=10.0, num_steps=1, perturb_demand=False):
    """
        Generate adversarial data based on the current model.
        Note: We only modify the continuous variable (i.e., coordinate) in CVRP.
        Note: data should include depot_xy, node_xy, normalized node_demand.
        data = (depot_xy, node_xy, node_demand)
        See also: "Learning to Solve Travelling Salesman Problem with Hardness-adaptive Curriculum" in AAAI 2022.
    """
    from CVRPEnv import CVRPEnv as Env
    from torch.autograd import Variable
    demand_scaler = {20: 30, 50: 40, 100: 50, 200: 70}
    def minmax(xy_):
        # min_max normalization: [b, n, 2]
        xy_ = (xy_ - xy_.min(dim=1, keepdims=True)[0]) / (xy_.max(dim=1, keepdims=True)[0] - xy_.min(dim=1, keepdims=True)[0])
        return xy_

    if eps == 0: return nat_data
    # generate x_adv
    model.eval()
    model.set_eval_type("softmax")
    depot_xy, node_xy, node_demand = nat_data
    depot_xy, node_xy, node_demand = depot_xy.clone().detach(), node_xy.clone().detach(), node_demand.clone().detach()
    aug_factor, batch_size = 1, node_xy.size(0)
    env = Env(**{'problem_size': node_xy.size(1), 'pomo_size': node_xy.size(1)})
    with torch.enable_grad():
        for i in range(num_steps):
            # depot_xy.requires_grad_()
            if perturb_demand:
                node_demand.requires_grad_()
            node_xy.requires_grad_()
            env.load_problems(batch_size, problems=(depot_xy, node_xy, node_demand), aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            prob_list = torch.zeros(size=(aug_factor * batch_size, env.pomo_size, 0))
            state, reward, done = env.pre_step()
            while not done:
                selected, prob = model(state)
                state, reward, done = env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size).permute(1, 0, 2).view(batch_size, -1)
            baseline_reward = aug_reward.float().mean(dim=1, keepdims=True)
            log_prob = prob_list.log().sum(dim=2).reshape(aug_factor, batch_size, env.pomo_size).permute(1, 0, 2).view(batch_size, -1)

            # May cause gradient nan problem.
            # delta = torch.autograd.grad(eps * ((aug_reward / baseline_reward) * log_prob).mean(), depot_xy, retain_graph=True)[0]  # original with baseline
            # depot_xy = depot_xy.detach() + delta
            if perturb_demand:
                delta = torch.autograd.grad(eps * ((aug_reward / baseline_reward) * log_prob).mean(), node_demand, retain_graph=True)[0]
                node_demand = node_demand.detach() + delta
            delta = torch.autograd.grad(eps * ((aug_reward / baseline_reward) * log_prob).mean(), node_xy)[0]  # original with baseline
            # delta = torch.autograd.grad(eps * (aug_reward * log_prob).mean(), node_xy)[0]  # original without baseline
            node_xy = node_xy.detach() + delta

            # data = minmax(torch.cat((depot_xy, node_xy), dim=1))
            # depot_xy, node_xy = data[:, :1, :], data[:, 1:, :]
            # depot_xy = Variable(depot_xy, requires_grad=False)
            if perturb_demand:
                node_demand = torch.clamp(torch.ceil(minmax(node_demand) * 9), min=1, max=9) / demand_scaler[node_xy.size(1)]
                # node_demand = minmax(node_demand)
                node_demand = Variable(node_demand, requires_grad=False)
            node_xy = minmax(node_xy)
            node_xy = Variable(node_xy, requires_grad=False)

    return depot_xy, node_xy, node_demand
