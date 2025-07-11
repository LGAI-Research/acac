import pickle
import torch
import os
import random
import numpy as np
from copy import deepcopy
from acac.acac_marl import PROJECT_DIR 

def save_checkpoint_cent(run_id, epi_count, eval_returns, controller, learner, envs_runner, save_dir, max_save=2):

    # PATH = './performance/' + save_dir + '/ckpt/' + str(run_id) + '_genric_' + '{}.tar'
    PATH = os.path.join(PROJECT_DIR, 'data', 'performance', save_dir, 'ckpt') + '/' + str(run_id) + '_genric_' + '{}.tar'

    for n in list(range(max_save-1, 0, -1)):
        if os.path.isfile(PATH.format(n)):
            os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
    PATH = PATH.format(1)

    torch.save({
                'epi_count': epi_count,
                'random_state': random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_random_state': torch.random.get_rng_state(),
                'envs_runner_returns': envs_runner.train_returns,
                'eval_returns': eval_returns,
                'joint_critic_net_state_dict': learner.joint_critic_net.state_dict(),
                'joint_critic_tgt_net_state_dict': learner.joint_critic_tgt_net.state_dict(),
                'joint_critic_optimizer_state_dict': learner.joint_critic_optimizer.state_dict() if learner.joint_critic_optimizer is not None else None,
                'actor_critic_optimizer_state_dict': learner.actor_critic_optimizer.state_dict() if getattr(learner, "actor_critic_optimizer", None) is not None else None,
                }, PATH)

    for idx, parent in enumerate(envs_runner.parents):
        parent.send(('get_rand_states', None))
    for idx, parent in enumerate(envs_runner.parents):
        # PATH = './performance/' + save_dir + '/ckpt/' + str(run_id) + '_env_rand_states_' + str(idx) + '{}.tar'
        PATH = os.path.join(PROJECT_DIR, 'data', 'performance', save_dir, 'ckpt') + '/' + str(run_id) + '_env_rand_states_' + str(idx) + '_{}.tar'
        rand_states = parent.recv()
        for n in list(range(max_save-1, 0, -1)):
            if os.path.isfile(PATH.format(n)):
                os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
        PATH = PATH.format(1)
        torch.save(rand_states, PATH)

    for idx, agent in enumerate(controller.agents):
        # PATH = './performance/' + save_dir + '/ckpt/' + str(run_id) + '_agent_' + str(idx) + '{}.tar'
        PATH = os.path.join(PROJECT_DIR, 'data', 'performance', save_dir, 'ckpt') + '/' + str(run_id) + '_agent_' + str(idx) + '_{}.tar'

        for n in list(range(max_save-1, 0, -1)):
            if os.path.isfile(PATH.format(n)):
                os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
        PATH = PATH.format(1)

        torch.save({
                    'actor_net_state_dict': agent.actor_net.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict() if agent.actor_optimizer is not None else None,
                    },PATH)
        print(f"{PATH} is saved successfully")

def load_checkpoint_cent(run_id, save_dir, controller, learner, envs_runner):

    # load generic stuff
    # PATH = './performance/' + save_dir + '/ckpt/' + str(run_id) + '_genric_' + '1.tar'
    PATH = os.path.join(PROJECT_DIR, 'data', 'performance', save_dir, 'ckpt_saved') + '/' + str(run_id) + '_genric_1.tar'
    ckpt = torch.load(PATH)
    epi_count = ckpt['epi_count']
    random.setstate(ckpt['random_state'])
    np.random.set_state(ckpt['np_random_state'])
    torch.set_rng_state(ckpt['torch_random_state'])
    envs_runner.train_returns = ckpt['envs_runner_returns']
    eval_returns = ckpt['eval_returns']
    learner.joint_critic_net.load_state_dict(ckpt['joint_critic_net_state_dict'])
    learner.joint_critic_tgt_net.load_state_dict(ckpt['joint_critic_tgt_net_state_dict'])
    learner.joint_critic_optimizer.load_state_dict(ckpt['joint_critic_optimizer_state_dict'])

    # load random states in all workers
    for idx, parent in enumerate(envs_runner.parents):
        # PATH = './performance/' + save_dir + '/ckpt/' + str(run_id) + '_env_rand_states_' + str(idx) + '1.tar'
        PATH = os.path.join(PROJECT_DIR, 'data', 'performance', save_dir, 'ckpt_saved') + '/' + str(run_id) + '_env_rand_states_' + str(idx) + '_1.tar'
        rand_states = torch.load(PATH)
        parent.send(('load_rand_states', rand_states))

    # load actor and ciritc models
    for idx, agent in enumerate(controller.agents):
        # PATH = './performance/' + save_dir + '/ckpt/' + str(run_id) + '_agent_' + str(idx) + '1.tar'
        PATH = os.path.join(PROJECT_DIR, 'data', 'performance', save_dir, 'ckpt_saved') + '/' + str(run_id) + '_agent_' + str(idx) + '_1.tar'
        ckpt = torch.load(PATH)
        agent.actor_net.load_state_dict(ckpt['actor_net_state_dict'])
        agent.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])

    print(f"{PATH} is loaded successfully")
    return epi_count, eval_returns



def load_policy_cent(run_id, save_dir, controller):
    for idx, agent in enumerate(controller.agents):
        # PATH = './performance/' + save_dir + '/ckpt/' + str(run_id) + '_agent_' + str(idx) + '1.tar'
        PATH = os.path.join(PROJECT_DIR, 'data', 'performance', save_dir, 'ckpt') + '/' + str(run_id) + '_agent_' + str(idx) + '_1.tar'
        ckpt = torch.load(PATH)
        print(f"agent {idx}: {list(ckpt['actor_net_state_dict'].keys())}")

        new_ckpt = {}
        for key in ckpt['actor_net_state_dict'].keys():
            if key.startswith("fc1") or key.startswith("fc2") or key.startswith("gru"):
                new_ckpt[f"encoder.{key}"] = deepcopy(ckpt["actor_net_state_dict"][key])
            else:
                new_ckpt[key] = deepcopy(ckpt["actor_net_state_dict"][key])
        agent.actor_net.load_state_dict(new_ckpt)

        # Check norms
        agent_state_dict = agent.actor_net.state_dict()
        for key in ckpt["actor_net_state_dict"].keys():
            if key.startswith("fc1") or key.startswith("fc2") or key.startswith("gru"):
                agent_key = f"encoder.{key}"
            else:
                agent_key = key
            agent_key_norm = torch.norm(agent_state_dict[agent_key])
            ckpt_key_norm = torch.norm(ckpt["actor_net_state_dict"][key])
            is_same_norm = (agent_key_norm == ckpt_key_norm)
            print(f"agent {idx}/{key} | norm of agent == norm of ckpt: " + "True" if is_same_norm else "False")
            # if not is_same_norm:
            print(f"agent {idx}/{key} | norm of agent: {agent_key_norm} | norm of ckpt: {ckpt_key_norm}")
        print(f"{PATH} is loaded successfully")