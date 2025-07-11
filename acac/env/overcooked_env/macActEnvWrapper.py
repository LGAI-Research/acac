import os
from gym import Wrapper

class MacEnvWrapper(Wrapper):
    def __init__(self, env):
        super(MacEnvWrapper, self).__init__(env)

    @property
    def n_agent(self): 
        return self.env.n_agent

    @property
    def obs_size(self):
        return self.env.obs_size
    
    @property
    def state_size(self):
        return self.env.get_vector_state().shape[0]

    @property
    def n_action(self):
        return self.env.n_action

    @property
    def action_spaces(self):
        return self.env.action_spaces

    @property
    def macroAgent(self):
        return self.env.macroAgent

    def reset(self): 
        return self.env.reset()

    def step(self, macro_actions):
    
        obs, reward, done, info = self.env.run(macro_actions)
          
        return obs, reward, done, info

    def action_space_sample(self):
        return self.env.macro_action_sample() 

    def get_state(self):
        return self.env.get_vector_state()

    def get_avail_actions(self):
        return self.env.get_avail_actions()
    


    
