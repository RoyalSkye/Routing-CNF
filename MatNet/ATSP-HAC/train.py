import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
import torch
import logging
from utils.utils import create_logger, copy_all_src
from utils.functions import seed_everything
from ATSPTrainer import ATSPTrainer as Trainer
from ATSPTrainer_baseline import ATSPTrainer as Trainer_baseline

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
    'eval_type': 'argmax',
    'one_hot_seed_cnt': 20,  # must be >= node_cnt
    'norm': "instance"
}

optimizer_params = {
    'optimizer': {
        'lr': 4*1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [301, ],  # if further training is needed
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'seed': 1234,
    'method': 'ours',  # choose from ['ours', 'baseline', 'baseline_hac']
    'routing_model': True,
    'epochs': 500,
    'train_episodes': 10 * 1000,
    'num_expert': 3,
    'train_batch_size': 200,
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss.json'
        },
    },
    # load checkpoint for phase 2
    'model_load': {
        'enable': False,
        'path': './result/saved_atsp_model',
        'epoch': 510,

    },
    # load pretrain model for phase 1
    'pretrain_load': {
        'enable': True,
        'path': '../../pretrained/MatNet-ATSP',
        'epoch': 5000,
        'n': 20,
    }
}

adv_params = {
    'eps_min': 1,
    'eps_max': 100,
    'num_steps': 1,
}

logger_params = {
    'log_file': {
        'desc': 'atsp_matnet_train',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    seed_everything(trainer_params['seed'])

    print(">> Starting {} Training, Routing network {}".format(trainer_params['method'], trainer_params['routing_model']))
    if trainer_params['method'] == "ours":
        trainer = Trainer(env_params=env_params, model_params=model_params, optimizer_params=optimizer_params, trainer_params=trainer_params, adv_params=adv_params)
    else:
        trainer = Trainer_baseline(env_params=env_params, model_params=model_params, optimizer_params=optimizer_params, trainer_params=trainer_params, adv_params=adv_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():

    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2
    trainer_params['validate_episodes'] = 4
    trainer_params['validate_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used


def occumpy_mem(cuda_device):
    """
    Occupy GPU memory in advance for size setting.
    """
    torch.cuda.set_device(cuda_device)
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    block_mem = int((total-used) * 0.5)
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x


if __name__ == "__main__":
    # reserve GPU memory
    # occumpy_mem(CUDA_DEVICE_NUM)
    main()
