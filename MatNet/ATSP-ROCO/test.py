import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
import logging
from utils.utils import create_logger, copy_all_src
from ATSPTester import ATSPTester as Tester
from utils.functions import seed_everything

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# parameters

env_params = {
    'node_cnt': 20,
    'problem_gen_params': {
        'int_min': 0,
        'int_max': 1000*1000,
        'scaler': 1000*1000
    },
    'pomo_size': 20  # same as node_cnt
}

model_params = {
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256**(1/2),
    'encoder_layer_num': 5,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16**(1/2),
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'eval_type': 'softmax',
    'one_hot_seed_cnt': 20,  # must be >= node_cnt
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'seed': 2023,
    'num_expert': 1,
    'model_load': {
        'path': './pretrained',  # directory path of pre-trained model and log files saved.
        'epoch': 500,  # epoch version of pre-trained model to load.
        'n': 20,
    },
    'saved_problem_folder': "./data/test_n20",
    'test_set_opt_sol_path': "./data/test_n20.pkl",
    'saved_problem_filename': 'problem_20_0_1000000_{}.atsp',
    'file_count': 1000,
    'test_batch_size': 100,
    'augmentation_enable': False,
    'aug_factor': 16,
    'aug_batch_size': 100,
}

if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'atsp_matnet_test',
        'filename': 'log.txt'
    }
}


def main():

    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    seed_everything(tester_params["seed"])

    tester = Tester(env_params=env_params, model_params=model_params, tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    tester_params['aug_factor'] = 10
    tester_params['file_count'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":
    main()
