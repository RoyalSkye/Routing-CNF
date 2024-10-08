import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
import torch
import logging
from utils.utils import create_logger, copy_all_src
from utils.functions import seed_everything
from CVRPTrainer import CVRPTrainer as Trainer
from CVRPTrainer_baseline import CVRPTrainer as Trainer_baseline

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0  # $ CUDA_VISIBLE_DEVICES=0 nohup python -u train.py 2>&1 &

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
    'norm': 'instance'
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [301, ],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'seed': 1234,
    'method': 'ours',  # choose from ['ours', 'baseline', 'baseline_hac']
    'routing_model': True,  # for 'ours'
    'global_attack': True,  # for 'ours' (Note: set adv_params['num_steps'] > 1 for abl study)
    'epochs': 500,
    'pretrain_epochs': 30500,
    'train_episodes': 10 * 1000,
    'num_expert': 3,
    'train_batch_size': 64,
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'general.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    # load checkpoint for phase 2
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': './result/saved_CVRP20_model',  # directory path of pre-trained model and log files saved.
        'epoch': 2000,  # epoch version of pre-trained model to laod.
    },
    # load pretrain model for phase 1
    'pretrain_load': {
        'enable': True,
        'path': '../../pretrained/POMO-CVRP',
        'epoch': 30500,
    }
}

adv_params = {
    'eps_min': 1,
    'eps_max': 100,
    'num_steps': 1,
    'perturb_demand': False
}

logger_params = {
    'log_file': {
        'desc': 'train_cvrp',
        'filename': 'log.txt'
    }
}


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
