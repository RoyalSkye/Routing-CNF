import os, sys
import time
import argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
import torch
import numpy as np
from datetime import timedelta

from TSPModel import TSPModel as Model
from TSProblemDef import generate_x_adv
from TSP_baseline import *
from utils.utils import *
from utils.functions import *


def generate_adv_dataset(model, data, eps_min=1, eps_max=100, num_steps=1):
    """
        generate adversarial dataset (ins and sol).
    """
    eps = iter([i for i in range(eps_min, eps_max+1, 1)])
    episode, batch_size, test_num_episode = 0, 10, data.size(0)
    adv_data, adv_opt = torch.zeros(0, data.size(1), data.size(2)), []
    while episode < test_num_episode:
        remaining = test_num_episode - episode
        batch_size = min(batch_size, remaining)
        nat_data = data[episode: episode + batch_size]
        x_adv = generate_x_adv(model, nat_data, eps=next(eps), num_steps=num_steps, return_opt=False)
        adv_data = torch.cat((adv_data, x_adv), dim=0)
        # adv_opt.extend(sol)
        episode += batch_size

    return adv_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_expert", type=int, default=3, help="Number of experts")
    parser.add_argument("--model_path", type=str, default='../../pretrained/POMO-TSP/checkpoint-3000.pt', help="Path of the checkpoint to load")
    parser.add_argument("--test_set_path", type=str, default='../../data/TSP/tsp100_uniform.pkl', help="Filename of the dataset(s) to evaluate")
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
    test_data = torch.Tensor(load_dataset(opts.test_set_path)[: opts.test_episodes])
    checkpoint = torch.load(opts.model_path, map_location=device)

    if opts.num_expert == 1:
        model = Model(**model_params)
        model.load_state_dict(checkpoint['model_state_dict'])
        models = [model]
    else:
        models = [Model(**model_params) for _ in range(opts.num_expert)]
        model_state_dict = checkpoint['model_state_dict']
        for i in range(opts.num_expert):
            models[i].load_state_dict(model_state_dict[i])

    # generate adversarial examples
    start_time = time.time()
    adv_data = torch.zeros(0, test_data.size(1), 2)
    for i in range(opts.num_expert):
        data = generate_adv_dataset(models[i], test_data, eps_min=opts.eps_min, eps_max=opts.eps_max, num_steps=opts.num_steps)
        adv_data = torch.cat((adv_data, data), dim=0)
    dir, filename = os.path.split(opts.test_set_path)
    save_dataset(adv_data.tolist(), "{}/adv_{}".format(dir, filename))
    print(">> Adversarial dataset generation finished within {:.2f}s".format(time.time() - start_time))

    # obtain (sub-)opt solution using Concorde
    start_time = time.time()
    params = argparse.ArgumentParser()
    params.cpus, params.n, params.progress_bar_mininterval = None, None, 0.1
    dataset = [(instance.cpu().numpy(),) for instance in adv_data]
    executable = os.path.abspath(os.path.join('concorde', 'concorde', 'TSP', 'concorde'))
    def run_func(args):
        return solve_concorde_log(executable, *args, disable_cache=True)
    results, parallelism = run_all_in_pool(run_func, "./Concorde_result", dataset, params, use_multiprocessing=False)
    os.system("rm -rf ./Concorde_result")

    costs, tours, durations = zip(*results)
    print(">> Solving adversarial dataset finished using Concorde within {:.2f}s".format(time.time() - start_time))
    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    print("Average serial duration: {} +- {}".format(np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

    results = [(i[0], i[1]) for i in results]
    save_dataset(results, "{}/concorde_adv_{}".format(dir, filename))
