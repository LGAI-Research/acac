import os
import argparse
import json
from acac.acac_marl import PROJECT_DIR

def override_params(params, config_path):
    assert os.path.exists(config_path), f"File {config_path} does not exist"
    with open(config_path, "r") as f:
        new_params = json.load(f)
        print(f"{config_path} is successfully loaded")
    
    for key, value in new_params.items():
        if key in params.keys():
            params[key] = value
        else:
            raise ValueError(f"Key {key} is not in params")
    return params

def merge_params(args, default_params):
    # Override default params with env and alg specific params
    alg_config = os.path.join(PROJECT_DIR, "scripts", "params", "alg", f"{args.alg_name}.json")
    env_config = os.path.join(PROJECT_DIR, "scripts", "params", "env", f"{args.env_name}.json")
    params = override_params(default_params, alg_config)
    if os.path.exists(env_config):
        params = override_params(params, env_config)
    else:
        print(f"File {env_config} does not exist")

    # Override params with command line arguments
    for key, value in vars(args).items():
        if value is not None:
            params[key] = value
    args = argparse.Namespace(**params)
    return args

def get_env_params(args):
    if args.env_id.startswith('BP'):
        env_params = {'grid_dim': args.grid_dim,
                      'big_box_reward': args.big_box_reward,
                      'small_box_reward': args.small_box_reward,
                      'penalty': args.penalty,
                      'n_agent': args.n_agent,
                      'terminate_step': args.env_terminate_step,
                      }
    else:
        TASKLIST = [
                "tomato salad", 
                "lettuce salad", 
                "onion salad", 
                "lettuce-tomato salad", 
                "onion-tomato salad", 
                "lettuce-onion salad", 
                "lettuce-onion-tomato salad",
                ]
        rewardList = {
                "subtask finished": 10, 
                "correct delivery": 200, 
                "wrong delivery": -5, 
                "step penalty": args.step_penalty
                }
        env_params = {
                'grid_dim': args.grid_dim,
                'map_type': args.map_type, 
                'n_agent' : args.n_agent,
                'task': TASKLIST[args.task],
                'rewardList': rewardList,
                'debug': False,
                'rand_start' :args.rand_start
                }
        if args.env_id[-1] == '0':
            env_params["n_knife"] = args.n_knife
            env_params["n_plate"] = args.n_plate
            env_params["obj"] = args.obj
            
    return env_params


def make_env_name(args):
    if args.env_id.startswith('Overcooked-v1') or args.env_id.startswith('Overcooked-MA-v0') or args.env_id.startswith('Overcooked-MA-v1'):
        if args.rand_start:
            map_name = f"{args.map_type}R{args.grid_dim[0]}"
        else:
            map_name = f"{args.map_type}{args.grid_dim[0]}"
        
        env_name = "ovc" + map_name
        if args.env_id.startswith('Overcooked-v1'):
            env_name = env_name + '_micro'
        if args.env_id[-1] == '0':
            env_name = env_name + f"_N{args.n_agent}"
    elif 'BP' in args.env_id:
        env_name = "bp" + str(args.grid_dim[0])
        if args.env_id.startswith('BP-v0'):
            env_name = env_name + f"_micro"
    else:
        raise NotImplementedError(f'No environment named {args.env_id}')
    return env_name


