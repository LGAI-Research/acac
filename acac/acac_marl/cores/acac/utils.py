import pickle
import torch
import os
import random
import numpy as np
from acac.acac_marl import PROJECT_DIR 

class Agent:

    def __init__(self):
        self.idx = None
        self.encoder = None
        self.encoder_tgt = None
        self.actor_net = None
        self.actor_optimizer = None
        self.actor_loss = None
        self.critic_net = None
        self.critic_tgt_net = None
        self.critic_optimizer = None
        self.critic_loss = None

class Linear_Decay(object):

    def __init__ (self, total_steps, init_value, end_value):
        self.total_steps = total_steps
        self.init_value = init_value
        self.end_value = end_value

    def get_value(self, step):
        frac = min(float(step) / self.total_steps, 1.0)
        return self.init_value + frac * (self.end_value-self.init_value)

def save_policies(run_id, agents, save_dir):
    for agent in agents:
        # PATH = './policy_nns/' + save_dir + '/' + str(run_id) + '_agent_' + str(agent.idx) + '.pt'
        PATH = os.path.join(PROJECT_DIR, 'data', 'policy_nns', save_dir, str(run_id) + '_agent_' + str(agent.idx) + '.pt')
        torch.save(agent.actor_net, PATH)
    PATH = os.path.join(PROJECT_DIR, 'data', 'policy_nns', save_dir, str(run_id) + '_agent_critic.pt')
    torch.save(agent.critic_net, PATH)

def save_train_data(run_id, data, save_dir):
    # with open('./performance/' + save_dir + '/train/train_perform' + str(run_id) + '.pickle', 'wb') as handle:
    with open(os.path.join(PROJECT_DIR, 'data', 'performance', save_dir, 'train', 'train_perform' + str(run_id) + '.pickle'), 'wb') as handle:
        pickle.dump(data, handle)

def save_test_data(run_id, data, save_dir):
    # with open('./performance/' + save_dir + '/test/test_perform' + str(run_id) + '.pickle', 'wb') as handle:
    with open(os.path.join(PROJECT_DIR, 'data', 'performance', save_dir, 'test', 'test_perform' + str(run_id) + '.pickle'), 'wb') as handle:
        pickle.dump(data, handle)
