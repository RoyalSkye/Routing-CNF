import os
import torch


def write_atsp(data, dir="tmp", scaler=1000*1000):
    if not os.path.exists(dir):
        os.makedirs(dir)

    for i in range(data.size(0)):
        with open(os.path.join(dir, "problem_{}_0_1000000_0_{}.atsp".format(data.size(-1), i)), 'w') as f:
            f.write("\n".join([
                "{} : {}".format(k, v)
                for k, v in (
                    ("TYPE", "ATSP"),
                    ("DIMENSION", data.size(-1)),
                    ("EDGE_WEIGHT_TYPE", "EXPLICIT"),
                    ("EDGE_WEIGHT_FORMAT", "FULL_MATRIX"),
                )
            ]))
            f.write("\n")
            f.write("EDGE_WEIGHT_SECTION\n")
            for i, ll in enumerate(data[i].cpu().tolist()):
                for j, l in enumerate(ll):
                    if j == i:
                        f.write("9999999\t")
                    else:
                        f.write("{}\t".format(int(l*scaler)))
                f.write("\n")
            f.write("EOF\n")


def get_random_problems(batch_size, node_cnt, problem_gen_params):

    ################################
    # "tmat" type
    ################################

    int_min = problem_gen_params['int_min']
    int_max = problem_gen_params['int_max']
    scaler = problem_gen_params['scaler']

    problems = torch.randint(low=int_min, high=int_max, size=(batch_size, node_cnt, node_cnt))
    # shape: (batch, node, node)
    problems[:, torch.arange(node_cnt), torch.arange(node_cnt)] = 0

    while True:
        old_problems = problems.clone()

        problems, _ = (problems[:, :, None, :] + problems[:, None, :, :].transpose(2,3)).min(dim=3)
        # shape: (batch, node, node)

        if (problems == old_problems).all():
            break

    # Scale
    scaled_problems = problems.float() / scaler

    return scaled_problems
    # shape: (batch, node, node)


def load_single_problem_from_file(filename, node_cnt, scaler):

    ################################
    # "tmat" type
    ################################

    problem = torch.empty(size=(node_cnt, node_cnt), dtype=torch.long)
    # shape: (node, node)

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as err:
        print(str(err))

    line_cnt = 0
    for line in lines:
        linedata = line.split()

        if linedata[0].startswith(('TYPE', 'DIMENSION', 'EDGE_WEIGHT_TYPE', 'EDGE_WEIGHT_FORMAT', 'EDGE_WEIGHT_SECTION', 'EOF')):
            continue

        integer_map = map(int, linedata)
        integer_list = list(integer_map)

        problem[line_cnt] = torch.tensor(integer_list, dtype=torch.long)
        line_cnt += 1

    # Diagonals to 0
    problem[torch.arange(node_cnt), torch.arange(node_cnt)] = 0

    # Scale
    scaled_problem = problem.float() / scaler

    return scaled_problem
    # shape: (node, node)


def generate_x_adv(model, nat_data, eps=10.0, num_steps=1, return_opt=False):
    """
        Generate adversarial data based on the current model.
        See also: "Learning to Solve Travelling Salesman Problem with Hardness-adaptive Curriculum" in AAAI 2022.
    """
    from ATSPEnv import ATSPEnv as Env
    from torch.autograd import Variable
    def minmax(xy_):
        # min_max normalization: [b, n, n]
        # Note: be careful with which projection operator to use!
        batch_size, n = xy_.size(0), xy_.size(1)
        xy_ = xy_.view(batch_size, -1)
        xy_ = (xy_ - xy_.min(dim=1, keepdims=True)[0]) / (xy_.max(dim=1, keepdims=True)[0] - xy_.min(dim=1, keepdims=True)[0])
        return xy_.view(batch_size, n, n)

    data = nat_data.clone().detach()
    if eps == 0: return data
    # generate x_adv
    model.eval()
    model.set_eval_type("softmax")
    aug_factor, batch_size = 1, data.size(0)
    env = Env(**{'node_cnt': data.size(1), 'problem_gen_params': {'int_min': 0, 'int_max': 1000*1000, 'scaler': 1000*1000}, 'pomo_size': data.size(1)})
    with torch.enable_grad():
        for i in range(num_steps):
            data.requires_grad_()
            env.load_problems_manual(data)
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
        raise NotImplementedError

    return data
