import os, time, copy
import itertools
import torch
import numpy as np
from attacker_env import AttackerEnv


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

        if linedata[0].startswith(
                ('TYPE', 'DIMENSION', 'EDGE_WEIGHT_TYPE', 'EDGE_WEIGHT_FORMAT', 'EDGE_WEIGHT_SECTION', 'EOF')):
            continue

        integer_map = map(int, linedata)
        integer_list = list(integer_map)

        problem[line_cnt] = torch.tensor(integer_list, dtype=torch.long)
        line_cnt += 1

    # Diagonals to 0
    problem[torch.arange(node_cnt), torch.arange(node_cnt)] = 0

    # Scale
    scaled_problem = problem.float() / scaler

    return scaled_problem  # shape: (node, node)


def get_lkh_executable(url="http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.8.tgz"):
    from subprocess import check_call, check_output, CalledProcessError
    from urllib.parse import urlparse

    cwd = os.path.abspath("../")
    os.makedirs(cwd, exist_ok=True)

    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    filedir = os.path.splitext(file)[0]

    if not os.path.isdir(filedir):
        print("{} not found, downloading and compiling".format(filedir))

        check_call(["wget", url], cwd=cwd)
        assert os.path.isfile(file), "Download failed, {} does not exist".format(file)
        check_call(["tar", "xvfz", file], cwd=cwd)

        assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
        check_call("make", cwd=filedir)
        os.remove(file)

    executable = os.path.join(filedir, "LKH")
    assert os.path.isfile(executable)
    return os.path.abspath(executable)


def write_atsp(data, dir="tmp", scaler=1000*1000):
    if not os.path.exists(dir):
        os.makedirs(dir)

    for i in range(data.size(0)):
        with open(os.path.join(dir, "problem_{}_0_1000000_0_{}.atsp".format(data.size(-1), i+1000)), 'w') as f:
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


def repeat_interleave(inp_list, repeat_num):
    return list(itertools.chain.from_iterable(zip(*itertools.repeat(inp_list, repeat_num))))


def beam_search_step_kernel(idx, act_n_sel,acts1, acts2, probs1, probs2, ready_nodes1, ready_nodes2_flat, graph_list, act_list, prob_list, orig_greedy, atsp_env, defense=False):
    beam_idx = idx // act_n_sel ** 2
    act1_idx = idx // act_n_sel % act_n_sel
    act2_idx = idx % act_n_sel
    act1, prob1 = acts1[beam_idx, act1_idx].item(), probs1[beam_idx, act1_idx].item()
    act2, prob2 = acts2[beam_idx, act1_idx, act2_idx].item(), probs2[beam_idx, act1_idx, act2_idx].item()
    ready_nodes_1 = ready_nodes1[beam_idx]
    ready_nodes_2 = ready_nodes2_flat[beam_idx * act_n_sel + act1_idx]

    if act1 in ready_nodes_1 and act2 in ready_nodes_2:
        assert prob1 > 0
        assert prob2 > 0
        reward, new_lower_matrix, edge_candidates, new_greedy, done = atsp_env.step(graph_list[beam_idx], (act1, act2), orig_greedy, defense)
        return (
                new_lower_matrix,
                edge_candidates,
                reward,
                act_list[beam_idx] + [(act1, act2)],
                prob_list[beam_idx] + [(prob1, prob2)],
                done
        )
    else:
        return None


def beam_search(base_model, policy_model, atsp_env, inp_lower_matrix, edge_candidates, greedy_cost, max_actions, beam_size=5, attack=True, defense=False, global_adv=False):
    start_time = time.time()

    if not global_adv:
        state_encoder = policy_model.state_encoder
        actor_net = policy_model.actor_net

    orig_greedy = greedy_cost
    best_tuple = (
        copy.deepcopy(inp_lower_matrix),  # input lower-left adjacency matrix
        edge_candidates,  # edge candidates
        -100,  # accumulated reward
        [],  # actions
        [],  # probabilities
        False,
    )
    topk_graphs = [best_tuple]

    act_n_sel = beam_size
    best_reward_each_step = np.zeros(max_actions + 1)
    for step in range(1, max_actions + 1):
        if global_adv:
            from ATSPTester import ATSPTester as Tester
            env_params = {'node_cnt': atsp_env.node_dimension, 'problem_gen_params': {'int_min': 0, 'int_max': 1000 * 1000, 'scaler': 1000 * 1000}, 'pomo_size': atsp_env.node_dimension}
            tester_params = {"augmentation_enable": True, "aug_factor": 16, "use_cuda": True, "cuda_device_num": torch.cuda.current_device()}
            # eval on M models
            best_model, best_attacker, best_length = None, None, 10**8
            for i in range(len(policy_model)):
                tester = Tester(env_params, None, tester_params, model=base_model[i])
                tour, length = tester.attacker_run((torch.tensor(inp_lower_matrix) / 1e4).to(torch.float32).to(device))
                if length < best_length:
                    best_length = length
                    best_model = base_model[i]
                    best_attacker = policy_model[i]
            # attack best model
            state_encoder = best_attacker.state_encoder
            actor_net = best_attacker.actor_net
            # recreate env
            tester = Tester(env_params, None, tester_params, model=best_model)
            atsp_env = AttackerEnv(solver_type="MatNet", node_dimension=atsp_env.node_dimension, is_attack=True, tester=tester, path="./data/train_n20", printinfo=False)

        lower_matrix_list, edge_cand_list, reward_list, act_list, prob_list = [], [], [], [], []
        for lower_matrix, edge_cand, reward, acts, probs, done in topk_graphs:
            lower_matrix_list.append(lower_matrix)
            edge_cand_list.append(edge_cand)
            reward_list.append(reward)
            act_list.append(acts)
            prob_list.append(probs)
            if done:
                ret_solution = orig_greedy + reward if (attack and not defense) else orig_greedy - reward
                return {
                    'reward': reward,
                    'solution': ret_solution,
                    'acts': acts,
                    'probs': probs,
                    'time': time.time() - start_time,
                }

        state_feat = state_encoder(lower_matrix_list)

        # mask1: (beam_size, max_num_nodes)
        mask1, ready_nodes1 = actor_net._get_mask1(state_feat.shape[0], state_feat.shape[1], edge_cand_list)
        # acts1, probs1: (beam_size, act_n_sel)
        acts1, probs1 = actor_net._select_node(state_feat, mask1, greedy_sel_num=act_n_sel)
        # acts1_flat, probs1_flat: (beam_size x act_n_sel,)
        acts1_flat, probs1_flat = acts1.reshape(-1), probs1.reshape(-1)
        # mask2_flat: (beam_size x act_n_sel, max_num_nodes)
        mask2_flat, ready_nodes2_flat = actor_net._get_mask2(state_feat.shape[0] * act_n_sel, state_feat.shape[1], repeat_interleave(edge_cand_list, act_n_sel), acts1_flat)
        # acts2_flat, probs2_flat: (beam_size x act_n_sel, act_n_sel)
        acts2_flat, probs2_flat = actor_net._select_node(state_feat.repeat_interleave(act_n_sel, dim=0), mask2_flat, prev_act=acts1_flat, greedy_sel_num=act_n_sel)
        # acts2, probs2: (beam_size, act_n_sel, act_n_sel)
        acts2, probs2 = acts2_flat.reshape(-1, act_n_sel, act_n_sel), probs2_flat.reshape(-1, act_n_sel, act_n_sel)

        acts1, acts2, probs1, probs2 = acts1.cpu(), acts2.cpu(), probs1.cpu(), probs2.cpu()

        def kernel_func_feeder(max_idx):
            for idx in range(max_idx):
                yield (
                    idx, act_n_sel,
                    acts1, acts2, probs1, probs2, ready_nodes1, ready_nodes2_flat,
                    lower_matrix_list, act_list, prob_list,
                    orig_greedy, atsp_env, defense
                )

        tmp_graphs = [beam_search_step_kernel(*x) for x in kernel_func_feeder(len(lower_matrix_list) * act_n_sel ** 2)]
        searched_graphs = []
        for graph_tuple in tmp_graphs:
            if graph_tuple is not None:
                searched_graphs.append(graph_tuple)

        # find the best action
        searched_graphs.sort(key=lambda x: x[2], reverse=True)
        if searched_graphs[0][2] > best_tuple[2]:
            best_tuple = searched_graphs[0]
        # print(searched_graphs[0], '\n\n')
        best_reward_each_step[step] = best_tuple[2]
        # find the topk expandable actions
        topk_graphs = searched_graphs[:beam_size]

    ret_solution = orig_greedy + best_tuple[2] if (attack and not defense) else orig_greedy - best_tuple[2]
    best_solution_each_step = orig_greedy + best_reward_each_step if (attack and not defense) else orig_greedy - best_reward_each_step

    return {
        'inp_lower_matrix': best_tuple[0],
        'reward': best_tuple[2],
        'solution': ret_solution,
        'acts': best_tuple[3],
        'probs': best_tuple[4],
        'time': time.time() - start_time,
        'best_reward_each_step': best_reward_each_step,
        'best_solution_each_step': best_solution_each_step,
    }


def generate_x_adv(model, attacker, nat_data, global_adv=False):
    """
    Generate adversarial data based on the attacker model.
    See also: "ROCO: A General Framework for Evaluating Robustness of Combinatorial Optimization Solvers on Graphs" in ICLR 2023.
    """
    from ATSPTester import ATSPTester as Tester
    env_params = {'node_cnt': nat_data.size(-1), 'problem_gen_params': {'int_min': 0, 'int_max': 1000 * 1000, 'scaler': 1000 * 1000}, 'pomo_size': nat_data.size(-1)}
    tester_params = {"augmentation_enable": True, "aug_factor": 16, "use_cuda": True, "cuda_device_num": torch.cuda.current_device()}
    if global_adv:
        for i in range(len(attacker)):
            model[i].eval()
            attacker[i].eval()
        tester = Tester(env_params, None, tester_params, model=model[0])
    else:
        model.eval()
        attacker.eval()
        tester = Tester(env_params, None, tester_params, model=model)

    atsp_env = AttackerEnv(solver_type="MatNet", node_dimension=nat_data.size(-1), is_attack=True, tester=tester, path="./data/train_n20")
    _, transform_data = atsp_env.generate_tuples(0, nat_data.size(0), rand_id=0, defense=True)
    print(len(transform_data))
    adv_data = nat_data.clone().detach()

    for id, (inp_lower_matrix, edge_candidates, ori_greedy, baselines, _, tsp_path) in enumerate(transform_data):
        print(id)
        bs_result = beam_search(model, attacker, atsp_env, inp_lower_matrix, edge_candidates, ori_greedy, max_actions=10, beam_size=3, attack=True, defense=False, global_adv=global_adv)
        actions = bs_result["acts"]
        # generate adv instances - half the cost of selected edges
        for action in actions:
            adv_data[id][action[0], action[1]] /= 2

    return adv_data


if __name__ == "__main__":
    import argparse, glob
    from attacker_model import ActorCritic
    from utils.utils import display_num_param
    from ATSPModel import ATSPModel as Model

    parser = argparse.ArgumentParser(description='attacker_test')
    parser.add_argument('--num_expert', default=3, type=int)
    parser.add_argument('--node_cnt', default=20, type=int)
    parser.add_argument('--one_hot_degree', default=0, type=int)
    parser.add_argument('--node_feature_dim', default=1, type=int)
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--node_output_size', default=16, type=int)
    parser.add_argument('--gnn_layers', default=3, type=int, help='number of GNN layers')
    parser.add_argument('--attacker_path_0', default='./pretrained/CNF/PPO_MatNet_node20_0_beam_3_ratio0.0065.pt', type=str, help='path of pretrained attacker model')
    parser.add_argument('--attacker_path_1', default='./pretrained/CNF/PPO_MatNet_node20_1_beam_3_ratio0.0060.pt', type=str, help='path of pretrained attacker model')
    parser.add_argument('--attacker_path_2', default='./pretrained/CNF/PPO_MatNet_node20_2_beam_3_ratio0.0067.pt', type=str, help='path of pretrained attacker model')
    parser.add_argument('--model_path', default='./pretrained/CNF/checkpoint-20-10.pt', type=str, help='path of pretrained model')
    parser.add_argument('--test_dataset_path', default='./data/train_n20', type=str, help='path of pretrained model')
    parser.add_argument('--global_attack', action='store_true')
    args = parser.parse_args()

    attack_params = args.node_feature_dim, args.node_output_size, args.batch_norm, args.one_hot_degree, args.gnn_layers
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    get_lkh_executable()

    print(">> Generating Local_attack - {}, Global_attack - {}".format(not args.global_attack, args.global_attack))

    # load model
    model_params = {
        'embedding_dim': 256,
        'sqrt_embedding_dim': 256 ** (1 / 2),
        'encoder_layer_num': 5,
        'qkv_dim': 16,
        'sqrt_qkv_dim': 16 ** (1 / 2),
        'head_num': 16,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'ms_hidden_dim': 16,
        'ms_layer1_init': (1 / 2) ** (1 / 2),
        'ms_layer2_init': (1 / 16) ** (1 / 2),
        'eval_type': 'argmax',
        'one_hot_seed_cnt': 20,  # must be >= node_cnt
    }
    if args.global_attack:
        pre_model = [Model(**model_params) for _ in range(args.num_expert)]
        checkpoint = torch.load(args.model_path, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
        for i in range(args.num_expert):
            model = pre_model[i]
            model.load_state_dict(model_state_dict[i])
            display_num_param(model)
    else:
        pre_model = Model(**model_params)
        checkpoint = torch.load(args.model_path, map_location=device)
        pre_model.load_state_dict(checkpoint["model_state_dict"][0])
        display_num_param(pre_model)

    # load attacker
    if args.global_attack:
        attacker = [ActorCritic(*attack_params) for _ in range(args.num_expert)]
        attacker_path = {0: args.attacker_path_0, 1: args.attacker_path_1, 2: args.attacker_path_2}
        for i in range(args.num_expert):
            model = attacker[i]
            checkpoint = torch.load(attacker_path[i], map_location=device)
            model.load_state_dict(checkpoint, strict=True)
            display_num_param(model)
    else:
        attacker = ActorCritic(*attack_params)
        checkpoint = torch.load(args.attacker_path_0, map_location=device)
        attacker.load_state_dict(checkpoint, strict=True)
        # print(attacker)
        display_num_param(attacker)

    # generate nat data
    if args.test_dataset_path is None:
        problem_gen_params = {
            'int_min': 0,
            'int_max': 1000 * 1000,
            'scaler': 1000 * 1000
        }
        nat_data = get_random_problems(64, args.node_cnt, problem_gen_params)
        print(nat_data.size())  # (batch_size, n, n)
        write_atsp(nat_data, dir="./tmp")
    else:
        nat_data = torch.zeros(0, args.node_cnt, args.node_cnt)
        for fp in sorted(glob.iglob(os.path.join(args.test_dataset_path, "*.atsp"))):
            data = load_single_problem_from_file(fp, node_cnt=args.node_cnt, scaler=1000*1000)
            nat_data = torch.cat((nat_data, data.unsqueeze(0)), dim=0)
        nat_data = nat_data.to(device)
        print(">> load {} data from {}".format(nat_data.size(0), args.test_dataset_path))

    # generate adv data
    adv_data = generate_x_adv(pre_model, attacker, nat_data, global_adv=args.global_attack)
    print(adv_data.size())
    write_atsp(adv_data, dir="adv_{}".format(os.path.split(args.test_dataset_path)[-1]))
