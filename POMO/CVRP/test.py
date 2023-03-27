import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
import torch
import logging
from multiprocessing import Pool
from utils.utils import create_logger, copy_all_src
from utils.functions import seed_everything, check_null_hypothesis
from CVRPTester import CVRPTester as Tester

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0
PARALLEL = True

##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'norm': 'instance',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'seed': 2023,
    'num_expert': 1,
    'model_load': {
        'path': '../../pretrained/POMO-CVRP',
        'epoch': 30500,
    },
    'test_episodes': 1000,
    'test_batch_size': 1000,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 100,
    'test_set_path': '../../data/CVRP/cvrp100_uniform.pkl',
    'test_set_opt_sol_path': '../../data/CVRP/hgs_cvrp100_uniform.pkl'
}

if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test_cvrp',
        'filename': 'log.txt'
    }
}


def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    avg_gaps, avg_aug_gaps = [], []
    all_scores, all_aug_scores = torch.zeros(tester_params['test_episodes'], 0), torch.zeros(tester_params['test_episodes'], 0)
    if not PARALLEL:
        # a. test all experts in sequential (on the same GPU)
        for i in range(tester_params['num_expert']):
            seed_everything(tester_params['seed'])
            tester_params['cuda_device_num'] = CUDA_DEVICE_NUM
            tester = Tester(env_params=env_params, model_params=model_params, tester_params=tester_params)
            copy_all_src(tester.result_folder)
            scores, aug_scores, opt_sol, _ = tester.run(i)
            all_scores = torch.cat((all_scores, scores.unsqueeze(1)), dim=1)
            all_aug_scores = torch.cat((all_aug_scores, aug_scores.unsqueeze(1)), dim=1)
            gaps = [(s - opt_sol[j]) / opt_sol[j] * 100 for j, s in enumerate(scores.tolist())]
            aug_gaps = [(s - opt_sol[j]) / opt_sol[j] * 100 for j, s in enumerate(aug_scores.tolist())]
            avg_gaps.append(sum(gaps)/len(gaps))
            avg_aug_gaps.append(sum(aug_gaps)/len(aug_gaps))
            print(">> Model {}: Scores {:.4f} -> x8 Aug Scores {:.4f}; Gaps {:.4f}% -> x8 Aug Gaps {:.4f}%".format(
                i, scores.mean().item(), aug_scores.mean().item(), sum(gaps)/len(gaps), sum(aug_gaps)/len(aug_gaps)))
    else:
        # b. test all experts in parallel (on multiple GPUs)
        res_list = []
        pool = Pool(processes=tester_params['num_expert'])
        for i in range(tester_params['num_expert']):
            seed_everything(tester_params['seed'])
            tester_params['cuda_device_num'] = i
            tester = Tester(env_params=env_params, model_params=model_params, tester_params=tester_params)
            copy_all_src(tester.result_folder)
            res = pool.apply_async(tester.run, args=(i, ))
            res_list.append(res)
        pool.close()
        pool.join()
        for r in res_list:
            scores, aug_scores, opt_sol, i = r.get()
            all_scores = torch.cat((all_scores, scores.unsqueeze(1)), dim=1)
            all_aug_scores = torch.cat((all_aug_scores, aug_scores.unsqueeze(1)), dim=1)
            gaps = [(s - opt_sol[j]) / opt_sol[j] * 100 for j, s in enumerate(scores.tolist())]
            aug_gaps = [(s - opt_sol[j]) / opt_sol[j] * 100 for j, s in enumerate(aug_scores.tolist())]
            avg_gaps.append(sum(gaps) / len(gaps))
            avg_aug_gaps.append(sum(aug_gaps) / len(aug_gaps))
            print(">> Model {}: Scores {:.4f} -> x8 Aug Scores {:.4f}; Gaps {:.4f}% -> x8 Aug Gaps {:.4f}%".format(
                i, scores.mean().item(), aug_scores.mean().item(), sum(gaps) / len(gaps), sum(aug_gaps) / len(aug_gaps)))

    best_scores, _ = all_scores.min(1)
    best_aug_scores, _ = all_aug_scores.min(1)
    avg_scores, avg_aug_scores = all_scores.mean(0).tolist(), all_aug_scores.mean(0).tolist()
    if opt_sol is not None:
        best_gaps = [(s - opt_sol[j]) / opt_sol[j] * 100 for j, s in enumerate(best_scores.tolist())]
        best_aug_gaps = [(s - opt_sol[j]) / opt_sol[j] * 100 for j, s in enumerate(best_aug_scores.tolist())]

        print(">> Val Score on {}: NO_AUG_Score: {} -> Min {:.4f} Col {:.4f}, AUG_Score: {} -> Min {:.4f} -> Col {:.4f}".format(
            os.path.split(tester_params['test_set_path'])[-1], avg_scores, min(avg_scores), best_scores.mean().item(),
            avg_aug_scores, min(avg_aug_scores), best_aug_scores.mean().item()))
        print(">> Val Score on {}: NO_AUG_Gap: {} -> Min {:.4f}% -> Col {:.4f}%, AUG_Gap: {} -> Min {:.4f}% -> Col {:.4f}%".format(
            os.path.split(tester_params['test_set_path'])[-1], avg_gaps, min(avg_gaps), sum(best_gaps)/len(best_gaps),
            avg_aug_gaps, min(avg_aug_gaps), sum(best_aug_gaps)/len(best_aug_gaps)))
    else:
        print(best_scores)
        print(best_aug_scores)


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


def t_test(path1, path2):
    """
        Conduct T-test to check the null hypothesis. If p < 0.05, the null hypothesis is rejected.
    """
    import pickle
    with open(path1, 'rb') as f1:
        results1 = pickle.load(f1)
    with open(path2, 'rb') as f2:
        results2 = pickle.load(f2)
    check_null_hypothesis(results1["score_list"], results2["score_list"])
    check_null_hypothesis(results1["aug_score_list"], results2["aug_score_list"])


if __name__ == "__main__":
    if PARALLEL:
        torch.multiprocessing.set_start_method('spawn')
    main()
