
import torch
import numpy as np


def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    return problems


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems


def generate_x_adv(model, nat_data, eps=10.0, num_steps=1, return_opt=False):
    """
        Generate adversarial data based on the current model.
        See also: "Learning to Solve Travelling Salesman Problem with Hardness-adaptive Curriculum" in AAAI 2022.
    """
    from TSPEnv import TSPEnv as Env
    from TSP_gurobi import solve_all_gurobi
    from torch.autograd import Variable
    def minmax(xy_):
        # min_max normalization: [b, n, 2]
        xy_ = (xy_ - xy_.min(dim=1, keepdims=True)[0]) / (xy_.max(dim=1, keepdims=True)[0] - xy_.min(dim=1, keepdims=True)[0])
        return xy_

    data = nat_data.clone().detach()
    if eps == 0: return data
    # generate x_adv
    model.eval()
    model.set_eval_type("softmax")
    aug_factor, batch_size = 1, data.size(0)
    env = Env(**{'problem_size': data.size(1), 'pomo_size': data.size(1)})
    with torch.enable_grad():
        for i in range(num_steps):
            data.requires_grad_()
            env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
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

            delta = torch.autograd.grad(eps * ((aug_reward / baseline_reward) * log_prob).mean(), data)[0]  # original with baseline
            # delta = torch.autograd.grad(eps * (aug_reward * log_prob).mean(), data)[0]  # original without baseline
            data = data.detach() + delta
            data = minmax(data)
            data = Variable(data, requires_grad=False)

    # generate opt sol
    if return_opt:
        return data, solve_all_gurobi(data)

    return data
