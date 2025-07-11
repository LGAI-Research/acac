#!/usr/bin/python
from gym.envs.registration import register
from acac.env.overcooked_env.macActEnvWrapper import MacEnvWrapper as OvercookedMacEnvWrapper

# For BoxPushing Env 
### Micro-action version
register(
    id='BP-v0',
    entry_point='acac.env.box_pushing_env.box_pushing:BoxPushing',
)

### Macro-action version
register(
    id='BP-MA-v0',
    entry_point='acac.env.box_pushing_env.box_pushing_MA:BoxPushing_harder',
)

# For Overcooked Env
### Micro-action version
register(
    id='Overcooked-v1',
    entry_point='acac.env.overcooked_env.overcooked_V1:Overcooked_V1',
)

### Macro-action version
register(
    id='Overcooked-MA-v1',
    entry_point='acac.env.overcooked_env.overcooked_MA_V1:Overcooked_MA_V1',
)

# For Overcooked-Large Env
### Macro-action version
register(
    id='Overcooked-MA-v0',
    entry_point='acac.env.overcooked_env.overcooked_MA_V0:Overcooked_MA_V0',
)
