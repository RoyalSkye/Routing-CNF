import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
import logging
from utils.utils import create_logger, copy_all_src
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
}

optimizer_params = {
    'optimizer': {
        'lr': 4*1e-5,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [2001, 2101],  # if further training is needed
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'seed': 1234,
    'num_expert': 3,
    "method": 'ours',  # choose from ['ours', 'baseline']
    'routing_model': False,
    'fixed_dataset': True,  # train on fixed dataset
    'epochs': 20,
    'train_episodes': 1000,
    'train_batch_size': 100,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 10,
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
        'path': './pretrained/AT',
        'epoch': 10,
        'n': 20,
    },
    # load pretrain model for phase 1
    'pretrain_load': {
        'enable': True,  # enable loading pre-trained model
        'path': './pretrained',  # directory path of pre-trained model and log files saved.
        'epoch': 500,  # epoch version of pre-trained model to laod.
        'n': 20,
    }
}

adv_params = {
    "one_hot_degree": 0,
    "node_feature_dim": 1,
    "batch_norm": False,
    "node_output_size": 16,
    "gnn_layers": 3,
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

    print(">> Starting {} Training, Routing network {}".format(trainer_params['method'], trainer_params['routing_model']))
    if trainer_params["method"] == "ours":
        trainer = Trainer(env_params=env_params, model_params=model_params, optimizer_params=optimizer_params, trainer_params=trainer_params, adv_params=adv_params)
    elif trainer_params["method"] == "baseline":
        trainer = Trainer_baseline(env_params=env_params, model_params=model_params, optimizer_params=optimizer_params, trainer_params=trainer_params, adv_params=adv_params)
    else:
        raise NotImplementedError

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


##########################################################################################

if __name__ == "__main__":
    main()
