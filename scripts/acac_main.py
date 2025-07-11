import argparse
import gym
import torch
import random
import numpy as np
import os
import sys
import json
import datetime
import wandb
from acac.acac_marl.misc.logging_utils import setup_logger
from acac.acac_marl.algs import (
    ACAC, ACAC_Vanilla, ACAC_Micro_GAE
)
from acac.acac_marl import PROJECT_DIR
from acac.env import OvercookedMacEnvWrapper


sys.path.append(os.path.dirname(__file__))
from params.utils import get_env_params, make_env_name, merge_params

algs = {      
    'acac': ACAC,
    'acac_vanilla': ACAC_Vanilla, 
    'acac_micro_gae': ACAC_Micro_GAE,
}

def main(args):

    # set seed
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    env_params = get_env_params(args)
    print(env_params)
    if args.share_encoder:
        args.a_mlp_layer_size[0] = args.c_mlp_layer_size[0]
        args.a_rnn_layer_size = args.c_rnn_layer_size
        args.time_emb_actor = args.time_emb_actor or args.time_emb

    if args.time_emb_actor:
        args.time_emb = True

    print("===========================")
    print("args: ")
    for key, value in vars(args).items():
        print(f"{key}: {value}")    
    print("===========================")

    env = gym.make(args.env_id, **env_params)
    if args.env_id.startswith("Overcooked-MA"):
        env = OvercookedMacEnvWrapper(env)
    args.env_name = env_params['env_name'] = make_env_name(args)
    args.save_dir = os.path.join(args.exp_name, args.env_name)
    print('exp_name: ', args.exp_name)
    print('env_name: ', args.env_name)
    print('save_dir: ', args.save_dir)
    print("===========================")
    
    assert os.path.isdir(PROJECT_DIR), f"PROJECT_DIR {PROJECT_DIR} is not a directory"
    os.makedirs(os.path.join(PROJECT_DIR, 'data', 'performance', args.exp_name, args.env_name, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, 'data', 'policy_nns', args.exp_name, args.env_name), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, 'data', 'mylog', args.exp_name, args.env_name), exist_ok=True)
    
    setup_logger(exp_prefix=args.exp_name, variant=vars(args), snapshot_mode='gap_and_last', snapshot_gap=100, env_name=args.env_name, seed=args.seed)
    
    if args.wandb:
        os.makedirs(os.path.join(PROJECT_DIR, 'data', 'wandb', args.exp_name, args.env_name), exist_ok=True)
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            group=str(args.exp_name),
            name=str(args.seed) + '_' + datetime.datetime.now().strftime('%d%H%M%S'),
            config = {**vars(args),**env_params},
            dir=os.path.join(PROJECT_DIR, 'data', 'wandb', args.exp_name, args.env_name),
        )

    model = algs[args.alg_name](env, **vars(args))
    model.learn()
    if args.wandb:
        wandb.finish()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # Load default params (apply default -> apply alg -> apply env config -> apply cmd args)
    with open(os.path.join(PROJECT_DIR, "scripts", "params", "default.json"), "r") as f:
        default_params = json.load(f)

    # Parse arguments
    parser = argparse.ArgumentParser()
    for key, value in default_params.items():
        if isinstance(value, list):
            parser.add_argument(f'--{key}', nargs='+', type=type(value[0]), default=None)
        else:
            parser.add_argument(f'--{key}', type=type(value), default=None)

    parser.add_argument("--duplicate", action="store_true", help='using critic based on duplicated obs')
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="acac_marl")
    args = parser.parse_args()
    args = merge_params(args, default_params)
    main(args)
