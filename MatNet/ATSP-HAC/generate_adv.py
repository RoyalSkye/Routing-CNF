import os, sys
import glob
import time
import argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
import torch
import numpy as np
from datetime import timedelta

from ATSPModel import ATSPModel as Model
from ATSProblemDef import *
from utils.utils import *
from utils.functions import *


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
    parser.add_argument("--node_cnt", type=int, default=20, help="Problem size")
    parser.add_argument("--num_expert", type=int, default=1, help="Number of experts")
    parser.add_argument("--model_path", type=str, default='../../pretrained/MatNet-ATSP/checkpoint-20-5000.pt', help="Path of the checkpoint to load")
    parser.add_argument("--test_set_path", type=str, default='../../data/ATSP-HAC/test_n20', help="Filename of the dataset(s) to evaluate")
    parser.add_argument('--test_episodes', type=int, default=1000, help="Number of instances to process")
    parser.add_argument('--eps_min', type=int, default=1, help="Min attack budget")
    parser.add_argument('--eps_max', type=int, default=100, help="Max attack budget")
    parser.add_argument('--num_steps', type=int, default=1, help="Number of steps to generate adversarial examples")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID")
    opts = parser.parse_args()

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
        'eval_type': 'softmax',
        'one_hot_seed_cnt': 20,  # must be >= node_cnt
        'norm': "instance"
    }
    torch.cuda.set_device(opts.gpu_id)
    device = torch.device('cuda', opts.gpu_id)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # test_data = torch.Tensor(load_dataset(opts.test_set_path)[: opts.test_episodes])
    checkpoint = torch.load(opts.model_path, map_location=device)

    # load dataset
    test_data = torch.zeros(0, opts.node_cnt, opts.node_cnt)
    for fp in sorted(glob.iglob(os.path.join(opts.test_set_path, "*.atsp"))):
        data = load_single_problem_from_file(fp, node_cnt=opts.node_cnt, scaler=1000 * 1000)
        test_data = torch.cat((test_data, data.unsqueeze(0)), dim=0)
    print(test_data.size())
    test_data = test_data[: opts.test_episodes].to(device)

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
    # Note: Could further tune the parameters for generating adversarial instances (e.g., eps_max, num_steps)
    print("\nNote: Could further tune the parameters for generating adversarial instances (e.g., eps_max, num_steps)!\n")
    start_time = time.time()
    adv_data = torch.zeros(0, test_data.size(1), test_data.size(1))
    for i in range(opts.num_expert):
        data = generate_adv_dataset(models[i], test_data, eps_min=opts.eps_min, eps_max=opts.eps_max, num_steps=opts.num_steps)
        adv_data = torch.cat((adv_data, data), dim=0)

    # save to files
    dir, filename = os.path.split(opts.test_set_path)
    save_dir = os.path.join(dir, "adv_{}".format(filename))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    write_atsp(adv_data, dir=save_dir, scaler=1000*1000)
    # save_dataset(adv_data.tolist(), "{}/adv_{}".format(dir, filename))
    print(">> Saved {} data to {}".format(adv_data.size(0), save_dir))
    print(">> Adversarial dataset generation finished within {:.2f}s".format(time.time() - start_time))

    # TODO: obtain (sub-)opt solution using LKH
    # start_time = time.time()
    # params = argparse.ArgumentParser()
    # params.cpus, params.n, params.progress_bar_mininterval = None, None, 0.1
    # dataset = [(instance.cpu().numpy(),) for instance in adv_data]
    # executable = os.path.abspath(os.path.join('concorde', 'concorde', 'TSP', 'concorde'))
    # def run_func(args):
    #     return solve_concorde_log(executable, *args, disable_cache=True)
    # results, parallelism = run_all_in_pool(run_func, "./Concorde_result", dataset, params, use_multiprocessing=False)
    # os.system("rm -rf ./Concorde_result")
    #
    # costs, tours, durations = zip(*results)
    # print(">> Solving adversarial dataset finished using Concorde within {:.2f}s".format(time.time() - start_time))
    # print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    # print("Average serial duration: {} +- {}".format(np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    # print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    # print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))
    #
    # results = [(i[0], i[1]) for i in results]
    # save_dataset(results, "{}/lkh_adv_{}".format(dir, filename))

    print("\nNot Implemented: obtain (sub-)opt solution using LKH!")
    raise NotImplementedError
