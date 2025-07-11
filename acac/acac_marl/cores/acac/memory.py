from collections import deque
import numpy as np
import torch

class Memory:

    """Base class of a memory buffer"""

    def __init__(self, 
                 obs_dims, 
                 action_dims, 
                 obs_last_action=False, 
                 size=1,
                 max_len=20):

        self.buf = deque(maxlen=size)
        self.max_len=max_len
        self.n_agent = len(obs_dims)

        if not obs_last_action:
            self.ZERO_OBS = [torch.zeros(dim).view(1,-1) for dim in obs_dims]
        else:
            self.ZERO_OBS = [torch.zeros(o_dim+a_dim).view(1,-1) for o_dim, a_dim in zip(*[obs_dims, action_dims])]
        self.ZERO_ACT = [torch.tensor(0).view(1,-1)] * self.n_agent
        self.ZERO_REWARD = [torch.tensor(0.0).view(1,-1)] * self.n_agent
        self.ZERO_JOINT_REWARD = torch.tensor(0.0).view(1,-1)
        self.ZERO_TERMINATE = torch.tensor(0.0).view(1,-1)
        self.ZERO_AVAIL_ACT = [torch.zeros(a_dim).view(1,-1) for a_dim in action_dims]
        self.ZERO_VALID = [torch.tensor(0, dtype=torch.bool).view(1,-1)] * self.n_agent
        self.ZERO_JOINT_VALID = torch.tensor(0, dtype=torch.bool).view(1,-1)
        self.ZERO_EXPV = [torch.tensor(0.0).view(1,-1)]*self.n_agent

        self.ZERO_PADDING = [(self.ZERO_OBS,
                              self.ZERO_VALID,
                              self.ZERO_AVAIL_ACT,
                              self.ZERO_ACT,
                              self.ZERO_REWARD,
                              self.ZERO_JOINT_REWARD, 
                              self.ZERO_OBS,
                              self.ZERO_AVAIL_ACT,
                              self.ZERO_TERMINATE,
                              self.ZERO_VALID,
                              self.ZERO_JOINT_VALID,
                              self.ZERO_EXPV)]

    def append(self, transition):
        self.scenario_cache.append(transition)

    def flush_buf_cache(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
    
    def _scenario_cache_reset(self):
        raise NotImplementedError

class Memory_epi(Memory):
    def __init__(self, *args, **kwargs):
        super(Memory_epi, self).__init__(*args, **kwargs)
        self._scenario_cache_reset()

    def flush_buf_cache(self):
        self.buf.append(self.scenario_cache)
        self._scenario_cache_reset()
    
    def sample(self, return_padded_batch=True):
        batch = list(self.buf)
        return self._padding_batches(batch) if return_padded_batch else batch

    def _scenario_cache_reset(self):
        self.scenario_cache = []

    def _padding_batches(self, batch):
        epi_len = [len(epi) for epi in batch] 
        max_len = max(epi_len)
        batch = [epi + self.ZERO_PADDING * (max_len - len(epi)) for epi in batch]
        return batch, max_len, epi_len
