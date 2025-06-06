##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from CVRPTester import CVRPTester as Tester


##########################################################################################
# parameters

env_params = {
    'max_problem_size': 80,
    'min_problem_size': 60,
    'pomo_size': 1000,
    'max_agent_num': 6,
    'min_agent_num': 3,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'softmax',
    'context_decoder': True,
    'context_layer_num': 1,
    'random_choice': True
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': '/home/wanpf/workspace_macbook/ad-hoc-fleet/ad_hoc_cvrp/result/20250224_213522_train_cvrp_context_random',  # directory path of pre-trained model and log files saved.
        'epoch': 2000,  # epoch version of pre-trained model to load.
    },
    'test_episodes': 1000,
    'test_batch_size': 1,
    'test_random_seed': 1234,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 400,
    'test_data_load': {
        'enable': False,
        'filename': '/home/wanpf/workspace_macbook/ad-hoc-fleet/ad_hoc_cvrp/data/test_data_123.pt'
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_cvrp_6',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    # create_logger(**logger_params)
    _print_config()
    start_time = time.time()
    end_time = time.time()
    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)

    # copy_all_src(tester.result_folder)
    no_aug_score, aug_score, _, __ = tester.run()
    print('no_aug_score:', no_aug_score)
    print('aug_score:', aug_score)
    print('time', end_time - start_time)
    print(_)
    print(__)

def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
