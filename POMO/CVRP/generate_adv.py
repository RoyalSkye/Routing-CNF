import os, sys
import pickle
import argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
import torch

from CVRPModel import CVRPModel as Model
from CVRProblemDef import generate_x_adv
from CVRP_baseline import *
from utils.utils import *
from utils.functions import *


def generate_adv_dataset(model, data, eps_min=1, eps_max=100, num_steps=1):
    """
        generate adversarial dataset (ins and sol).
    """
    eps = iter([i for i in range(eps_min, eps_max+1, 1)])
    depot_xy, node_xy, node_demand = data
    episode, batch_size, test_num_episode = 0, 10, depot_xy.size(0)
    # adv_depot_xy = torch.zeros(0, 1, 2)
    adv_node_xy = torch.zeros(0, node_xy.size(1), 2)
    while episode < test_num_episode:
        remaining = test_num_episode - episode
        batch_size = min(batch_size, remaining)
        nat_data = (depot_xy[episode: episode + batch_size], node_xy[episode: episode + batch_size], node_demand[episode: episode + batch_size])
        _, node, _ = generate_x_adv(model, nat_data, eps=next(eps), num_steps=num_steps)
        # adv_depot_xy = torch.cat((adv_depot_xy, depot), dim=0)
        adv_node_xy = torch.cat((adv_node_xy, node), dim=0)
        episode += batch_size

    return depot_xy, adv_node_xy, node_demand


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_expert", type=int, default=1, help="Number of experts")
    parser.add_argument("--model_path", type=str, default='../../pretrained/POMO-CVRP/checkpoint-30500.pt', help="Path of the checkpoint to load")
    parser.add_argument("--test_set_path", type=str, default='../../data/CVRP/cvrp100_uniform.pkl', help="Filename of the dataset(s) to evaluate")
    parser.add_argument('--test_episodes', type=int, default=1000, help="Number of instances to process")
    parser.add_argument('--eps_min', type=int, default=1, help="Min attack budget")
    parser.add_argument('--eps_max', type=int, default=100, help="Max attack budget")
    parser.add_argument('--num_steps', type=int, default=1, help="Number of steps to generate adversarial examples")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID")
    opts = parser.parse_args()

    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** (1 / 2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'softmax',
        'norm': 'instance',
    }
    torch.cuda.set_device(opts.gpu_id)
    device = torch.device('cuda', opts.gpu_id)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    checkpoint = torch.load(opts.model_path, map_location=device)

    # load data & preprocessing
    data = load_dataset(opts.test_set_path)[: opts.test_episodes]
    depot_xy, node_xy, ori_node_demand, capacity = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data]
    depot_xy, node_xy, ori_node_demand, capacity = torch.Tensor(depot_xy), torch.Tensor(node_xy), torch.Tensor(ori_node_demand), torch.Tensor(capacity)
    node_demand = ori_node_demand / capacity.view(-1, 1)
    test_data = (depot_xy, node_xy, node_demand)  # [batch_size, 1, 2], [batch_size, problems, 2], [batch_size, problems]

    if opts.num_expert == 1:
        model = Model(**model_params)
        model.load_state_dict(checkpoint['model_state_dict'])
        models = [model]
    else:
        models = [Model(**model_params) for _ in range(opts.num_expert)]
        model_state_dict = checkpoint['model_state_dict']
        models = [models[i].load_state_dict(model_state_dict[i]) for i in range(opts.num_expert)]

    # generate adversarial examples (only coordinates of nods are adversarially updated)
    # adv_depot_xy = torch.zeros(0, 1, 2)
    adv_node_xy = torch.zeros(0, test_data[1].size(1), 2)
    for i in range(opts.num_expert):
        _, node, _ = generate_adv_dataset(models[i], test_data, eps_min=opts.eps_min, eps_max=opts.eps_max, num_steps=opts.num_steps)
        # adv_depot_xy = torch.cat((adv_depot_xy, depot), dim=0)
        adv_node_xy = torch.cat((adv_node_xy, node), dim=0)
    dir, filename = os.path.split(opts.test_set_path)

    adv_data = (torch.cat([depot_xy] * opts.num_expert, dim=0), adv_node_xy, torch.cat([ori_node_demand] * opts.num_expert, dim=0), torch.cat([capacity] * opts.num_expert, dim=0))
    # save_dataset(adv_data, "{}/adv_{}".format(dir, filename))
    with open("{}/adv_{}".format(dir, filename), "wb") as f:
        pickle.dump(list(zip(adv_data[0].tolist(), adv_data[1].tolist(), adv_data[2].tolist(), adv_data[3].tolist())), f, pickle.HIGHEST_PROTOCOL)  # [(depot_xy, node_xy, node_demand, capacity), ...]

    # obtain (sub-)opt solution using HGS
    params = argparse.ArgumentParser()
    params.cpus, params.n, params.progress_bar_mininterval = None, None, 0.1
    dataset = [attr.cpu().tolist() for attr in adv_data]
    dataset = [(dataset[0][i][0], dataset[1][i], [int(d) for d in dataset[2][i]], int(dataset[3][i])) for i in range(adv_data[0].size(0))]
    executable = get_hgs_executable()
    def run_func(args):
        return solve_hgs_log(executable, *args, runs=1, disable_cache=True)  # otherwise it directly loads data from dir

    results, _ = run_all_in_pool(run_func, "./HGS_result", dataset, params, use_multiprocessing=False)
    os.system("rm -rf ./HGS_result")

    results = [(i[0], i[1]) for i in results]
    save_dataset(results, "{}/concorde_adv_{}".format(dir, filename))
    print(">> Adversarial Dataset (len: {}) Generated Finished!".format(len(results)))
