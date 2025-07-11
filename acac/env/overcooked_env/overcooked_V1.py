import gym
import numpy as np
from .render.game import Game
from gym import spaces
from .items import Tomato, Lettuce, Onion, Plate, Knife, Delivery, Agent, Food
import copy
from .map_utils import make_map

DIRECTION = [(0,1), (1,0), (0,-1), (-1,0)]
ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion"]
ITEMIDX= {"space": 0, "counter": 1, "agent": 2, "tomato": 3, "lettuce": 4, "plate": 5, "knife": 6, "delivery": 7, "onion": 8}
AGENTCOLOR = ["blue", "magenta", "green", "yellow"]
TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]

class Overcooked_V1(gym.Env):

    """
    Overcooked Domain Description
    ------------------------------
    Agent with primitive actions ["right", "down", "left", "up"]
    TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]
    
    1) Agent is allowed to pick up/put down food/plate on the counter;
    2) Agent is allowed to chop food into pieces if the food is on the cutting board counter;
    3) Agent is allowed to deliver food to the delivery counter;
    4) Only unchopped food is allowed to be chopped;
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 5
        }

    def __init__(self, grid_dim, task, rewardList, map_type = "A", n_agent = 2, obs_radius = 2, mode = "vector", debug = False, rand_start=False):

        """
        Parameters
        ----------
        gird_dim : tuple(int, int)
            The size of the grid world([7, 7]/[9, 9]).
        task : int
            The index of the target recipe.
        rewardList : dictionary
            The list of the reward.
            e.g rewardList = {"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1}
        map_type : str 
            The type of the map(A/B/C).
        n_agent: int
            The number of the agents.
        obs_radius: int
            The radius of the agents.
        mode: string
            The type of the observation(vector/image).
        debug : bool
            Whehter print the debug information.
        """

        self.xlen, self.ylen = grid_dim
        if debug:
            self.game = Game(self)

        self.task = task
        self.rewardList = rewardList
        self.mapType = map_type
        self.debug = debug
        self.n_agent = n_agent
        self.mode = mode
        self.obs_radius = obs_radius
        self.rand_start = rand_start
        print(f'rand_start: {self.rand_start}')

        assert self.xlen == self.ylen, f'self.xlen {self.xlen} should be matched to self.ylen {self.ylen}'
        self.initMap, self.pomap = make_map(map_type=self.mapType, n_agent=n_agent, size=self.xlen)
        self.map = copy.deepcopy(self.initMap)

        self.oneHotTask = []
        for t in TASKLIST:
            if t == self.task:
                self.oneHotTask.append(1)
            else:
                self.oneHotTask.append(0)

        self._createItems()
        self.n_agent = len(self.agent)

        #action: move(up, down, left, right), stay
        self.action_space = spaces.Discrete(5)

        #Observation: agent(pos[x,y]) dim = 2
        #    knife(pos[x,y]) dim = 2
        #    delivery (pos[x,y]) dim = 2
        #    plate(pos[x,y]) dim = 2
        #    food(pos[x,y]/status) dim = 3

        self._initObs()
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self._get_obs()[0]),), dtype=np.float32)


    def _createItems(self):
        self.agent = []
        self.knife = []
        self.delivery = []
        self.tomato = []
        self.lettuce = []
        self.onion = []
        self.plate = []
        self.itemList = []
        agent_idx = 0
        for y in range(self.ylen):
            for x in range(self.xlen):
                if self.map[x][y] == ITEMIDX["agent"]:
                    self.agent.append(Agent(x, y, color = AGENTCOLOR[agent_idx]))
                    agent_idx += 1
                elif self.map[x][y] == ITEMIDX["knife"]:
                    self.knife.append(Knife(x, y))
                elif self.map[x][y] == ITEMIDX["delivery"]:
                    self.delivery.append(Delivery(x, y))                    
                elif self.map[x][y] == ITEMIDX["tomato"]:
                    self.tomato.append(Tomato(x, y))
                elif self.map[x][y] == ITEMIDX["lettuce"]:
                    self.lettuce.append(Lettuce(x, y))
                elif self.map[x][y] == ITEMIDX["onion"]:
                    self.onion.append(Onion(x, y))
                elif self.map[x][y] == ITEMIDX["plate"]:
                    self.plate.append(Plate(x, y))
        
        self.itemDic = {"tomato": self.tomato, "lettuce": self.lettuce, "onion": self.onion, "plate": self.plate, "knife": self.knife, "delivery": self.delivery, "agent": self.agent}
        for key in self.itemDic:
            self.itemList += self.itemDic[key]


    def _initObs(self):
        obs = []
        for item in self.itemList:
            obs.append(item.x / self.xlen)
            obs.append(item.y / self.ylen)
            if isinstance(item, Food):
                obs.append(item.cur_chopped_times / item.required_chopped_times)
        obs += self.oneHotTask 

        for agent in self.agent:
            agent.obs = obs
        return [np.array(obs)] * self.n_agent


    def _get_vector_state(self):
        state = []
        for item in self.itemList:
            x = item.x / self.xlen
            y = item.y / self.ylen
            state.append(x)
            state.append(y)
            if isinstance(item, Food):
                state.append(item.cur_chopped_times / item.required_chopped_times)

        state += self.oneHotTask
        return [np.array(state)] * self.n_agent
    
    def get_vector_state(self):
        return self._get_vector_state()[0][:-len(self.oneHotTask)]
    
    def set_vector_state(self, state):
        idx = 0
        map_size = len(self.map)
        for item in self.itemList:
            item.x = round(state[idx] * self.xlen)
            item.y = round(state[idx + 1] * self.ylen)
            if isinstance(item, Food):
                item.cur_chopped_times = round(state[idx + 2] * item.required_chopped_times)
                idx += 3
            else:
                idx += 2
        
        layered_map = [[[None for _ in range(len(self.itemDic.keys()))] for _ in range(map_size)] for _ in range(map_size)]
        for i, (item_name, item_list) in enumerate(self.itemDic.items()):
            for j, item in enumerate(item_list):
                layered_map[item.x][item.y][i] = item
                if hasattr(item, "holding"):
                    item.holding = None
                if hasattr(item, "containing"):
                    item.containing = None
        
        self.map = copy.deepcopy(self.initMap)

        # Clear all movable things on the map
        for i in range(map_size):
            for j in range(map_size):
                if self.map[i][j] == ITEMIDX["agent"]:
                    self.map[i][j] = ITEMIDX["space"]
                if self.map[i][j] in [ITEMIDX["tomato"], ITEMIDX["lettuce"], ITEMIDX["onion"], ITEMIDX["plate"]]:
                    self.map[i][j] = ITEMIDX["counter"]
        
        # Set all items on the map and 
        for x in range(map_size):
            for y in range(map_size):
                exist_obj_list = [obj for obj in layered_map[x][y] if obj is not None]
                if len(exist_obj_list) == 0:
                    continue
                elif len(exist_obj_list) == 1:
                    self.map[x][y] = ITEMIDX[exist_obj_list[0].rawName]
                else:
                    # print(f"exist_obj_list at {x}, {y}: ", [obj.rawName for obj in exist_obj_list])
                    food_obj_list = [obj for obj in exist_obj_list if obj.rawName in ["tomato", "lettuce", "onion"]]
                    plate_obj_list = [obj for obj in exist_obj_list if obj.rawName == "plate"]
                    knife_obj_list = [obj for obj in exist_obj_list if obj.rawName == "knife"]
                    delivery_obj_list = [obj for obj in exist_obj_list if obj.rawName == "delivery"]
                    agent_obj_list = [obj for obj in exist_obj_list if obj.rawName == "agent"]
                
                    if len(food_obj_list) > 0 and len(plate_obj_list) > 0:
                        plate_obj_list[0].containing = food_obj_list
                        self.map[x][y] = ITEMIDX["plate"]
                    elif len(food_obj_list) > 0 and len(knife_obj_list) > 0:
                        knife_obj_list[0].holding = food_obj_list[0]
                        self.map[x][y] = ITEMIDX["knife"]
        
                    if len(plate_obj_list) > 0 and len(agent_obj_list) > 0:
                        agent_obj_list[0].holding = plate_obj_list[0]
                        self.map[x][y] = ITEMIDX["agent"]
                    elif len(food_obj_list) > 0 and len(agent_obj_list) > 0:
                        agent_obj_list[0].holding = food_obj_list[0]
                        self.map[x][y] = ITEMIDX["agent"]
                    
                    if len(plate_obj_list) > 0 and len(delivery_obj_list) > 0:
                        delivery_obj_list[0].holding = plate_obj_list[0]
                        self.map[x][y] = ITEMIDX["delivery"]
        

    def _get_image_state(self):
        return [self.game.get_image_obs()] * self.n_agent

    def _get_obs(self):
        """
        Returns
        -------
        obs : list
            observation for each agent.
        """

        vec_obs = self._get_vector_obs()
        if self.obs_radius > 0:
            if self.mode == "vector":
                return vec_obs
            elif self.mode == "image":
                return self._get_image_obs()
        else:
            if self.mode == "vector":
                return self._get_vector_state()
            elif self.mode == "image":
                return self._get_image_state()

    def _get_vector_obs(self):

        """
        Returns
        -------
        vector_obs : list
            vector observation for each agent.
        """

        po_obs = []

        for agent in self.agent:
            obs = []
            idx = 0
            agent.pomap = copy.deepcopy(self.pomap)

            for item in self.itemList:
                if item.x >= agent.x - self.obs_radius and item.x <= agent.x + self.obs_radius and item.y >= agent.y - self.obs_radius and item.y <= agent.y + self.obs_radius \
                    or self.obs_radius == 0:
                    x = item.x / self.xlen
                    y = item.y / self.ylen
                    obs.append(x)
                    obs.append(y)
                    idx += 2
                    if isinstance(item, Food):
                        obs.append(item.cur_chopped_times / item.required_chopped_times)
                        idx += 1
                else:
                    x = agent.obs[idx] * self.xlen
                    y = agent.obs[idx + 1] * self.ylen
                    if x >= agent.x - self.obs_radius and x <= agent.x + self.obs_radius and y >= agent.y - self.obs_radius and y <= agent.y + self.obs_radius:
                        x = item.initial_x
                        y = item.initial_y
                    x = x / self.xlen
                    y = y / self.ylen

                    obs.append(x)
                    obs.append(y)
                    idx += 2
                    if isinstance(item, Food):
                        obs.append(agent.obs[idx] / item.required_chopped_times)
                        idx += 1

                agent.pomap[int(x * self.xlen)][int(y * self.ylen)] = ITEMIDX[item.rawName]
            agent.pomap[agent.x][agent.y] = ITEMIDX["agent"]
            obs += self.oneHotTask 
            agent.obs = obs
            po_obs.append(np.array(obs))
        return po_obs

    def _get_image_obs(self):

        """
        Returns
        -------
        image_obs : list
            image observation for each agent.
        """

        po_obs = []
        frame = self.game.get_image_obs()
        old_image_width, old_image_height, channels = frame.shape
        new_image_width = int((old_image_width / self.xlen) * (self.xlen + 2 * (self.obs_radius - 1)))
        new_image_height =  int((old_image_height / self.ylen) * (self.ylen + 2 * (self.obs_radius - 1)))
        color = (0,0,0)
        obs = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2

        obs[x_center:x_center+old_image_width, y_center:y_center+old_image_height] = frame

        for idx, agent in enumerate(self.agent):
            agent_obs = self._get_PO_obs(obs, agent.x, agent.y, old_image_width, old_image_height)
            po_obs.append(agent_obs)
        return po_obs

    def _get_PO_obs(self, obs, x, y, ori_width, ori_height):
        x1 = (x - 1) * int(ori_width / self.xlen)
        x2 = (x + self.obs_radius * 2) * int(ori_width / self.xlen)
        y1 = (y - 1) * int(ori_height / self.ylen)
        y2 = (y + self.obs_radius * 2) * int(ori_height / self.ylen)
        return obs[x1:x2, y1:y2]

    def _findItem(self, x, y, itemName):
        for item in self.itemDic[itemName]:
            if item.x == x and item.y == y:
                return item
        return None

    def _make_new_rand_map(self,map) :
        map_size = len(map)
        map=np.array(map)
        objects, counts = np.unique(np.array(map),return_counts=True) 
        map[map==2] = 0
        map[map!=0]=1
               
        for i in range(0,len(objects)):
            if objects[i] == 0 or objects[i] == 1 :
                continue
            elif objects[i] == 2 : #agent
                num = counts[i]
                loc_available = False
                while not loc_available:
                    loc_candidate = []
                    for _ in range(0,num):
                        while True :
                            pos = np.random.randint(1,map_size-1,2)
                            if map[pos[0]][pos[1]] == 0 and pos.tolist() not in loc_candidate:
                                loc_candidate.append(pos.tolist())
                                break
                    l_objcounts = np.sum([pos[1] < int(map_size // 2) for pos in loc_candidate])
                    r_objcounts = np.sum([pos[1] >= int(map_size // 2) for pos in loc_candidate])

                    if self.mapType == 'B':
                        if l_objcounts >= 1 and r_objcounts >= 1:
                            loc_available = True
                    else:
                        loc_available = True
                for pos in loc_candidate: 
                    map[pos[0]][pos[1]] = objects[i]
            else : 
                num = counts[i]
                possible_pos = [[0,i] for i in np.arange(1,map_size-1)] + \
                    [[i,0] for i in np.arange(1,map_size-1)] + \
                    [[map_size-1,i] for i in np.arange(1,map_size-1)] + \
                    [[i,map_size-1] for i in np.arange(1,map_size-1)]
                if self.mapType == 'B':
                    possible_pos.remove([0,int(map_size//2)])
                    possible_pos.remove([map_size-1,int(map_size//2)])
                elif self.mapType == 'C':
                    possible_pos.remove([0,int(map_size//2)])
                    
                possible_pos = np.unique(np.array(possible_pos), axis=0)
                
                for _ in range(0,num) :
                    while True :
                        idx = np.random.randint(0,possible_pos.shape[0])
                        pos = possible_pos[idx]
                        if map[pos[0]][pos[1]] == 1 :
                            map[pos[0]][pos[1]] = objects[i]
                            break
        return map
    
    def _make_new_left_right_fixed_map(self,map) :
        map_size = len(map)
        map=np.array(map)
        l_objcounts = np.unique(np.array(map[:, :int(map_size // 2)]),return_counts=True) 
        r_objcounts = np.unique(np.array(map[:, int(map_size // 2):]),return_counts=True) 
        map[map==2] = 0
        map[map!=0]=1
            
        for lr_idx, (objects, counts) in enumerate([l_objcounts, r_objcounts]):
            for i in range(0,len(objects)):
                if objects[i] == 0 or objects[i] == 1 :
                    continue
                elif objects[i] == 2 : #agent
                    num = counts[i]
                    for _ in range(0,num) :
                        while True :
                            if lr_idx == 0:
                                xpos = np.random.randint(1, map_size-1)
                                ypos = np.random.randint(1,int(map_size//2))
                            else:
                                xpos = np.random.randint(1, map_size-1)
                                ypos = np.random.randint(int(map_size//2), map_size-1)
                                
                            if map[xpos][ypos] == 0 :
                                map[xpos][ypos] = objects[i]
                                break
                else : 
                    num = counts[i]
                    possible_pos = [[0,i] for i in np.arange(1,map_size-1)] + \
                        [[i,0] for i in np.arange(1,map_size-1)] + \
                        [[map_size-1,i] for i in np.arange(1,map_size-1)] + \
                        [[i,map_size-1] for i in np.arange(1,map_size-1)]
                    if self.mapType == 'B':
                        possible_pos.remove([0,int(map_size//2)])
                        possible_pos.remove([map_size-1,int(map_size//2)])
                    elif self.mapType == 'C':
                        possible_pos.remove([0,int(map_size//2)])
                    
                    possible_pos = np.unique(np.array(possible_pos), axis=0)
                    if lr_idx == 0:
                        possible_pos = np.array([pos for pos in possible_pos if pos[1] < int(map_size//2)])
                    else:
                        possible_pos = np.array([pos for pos in possible_pos if pos[1] >= int(map_size//2)])
                
                    for _ in range(0,num) :
                        while True :
                            idx = np.random.randint(0,possible_pos.shape[0])
                            pos = possible_pos[idx]
                            if map[pos[0]][pos[1]] == 1 :
                                map[pos[0]][pos[1]] = objects[i]
                                break
        return map
    
    
    @property
    def state_size(self):
        return self.get_state().shape[0]

    @property
    def obs_size(self):
        return [self.observation_space.shape[0]] * self.n_agent

    @property
    def n_action(self):
        return [a.n for a in self.action_spaces]

    @property
    def action_spaces(self):
        return [self.action_space] * self.n_agent

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agent)]

    def get_avail_agent_actions(self, nth):
        return [1] * self.action_spaces[nth].n

    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)
    
    def reset(self):

        """
        Returns
        -------
        obs : list
            observation for each agent.
        """
        
        if self.rand_start :
            self.map = self._make_new_rand_map(copy.deepcopy(self.initMap))
        else:
            self.map = copy.deepcopy(self.initMap)
        

        self._createItems()
        self._initObs()
        if self.debug:
            self.game.on_cleanup()

        return self._get_obs()
    
    def step(self, action):

        """
        Parameters
        ----------
        action: list
            action for each agent

        Returns
        -------
        obs : list
            observation for each agent.
        rewards : list
        terminate : list
        info : dictionary
        """

        reward = self.rewardList["step penalty"]
        done = False
        info = {}
        info['cur_mac'] = action
        info['mac_done'] = [True] * self.n_agent
        info['collision'] = []

        all_action_done = False

        for agent in self.agent:
            agent.moved = False

        if self.debug:
            print("in overcooked primitive actions:", action)

        while not all_action_done:
            for idx, agent in enumerate(self.agent):
                agent_action = action[idx]
                if agent.moved:
                    continue
                agent.moved = True

                if agent_action < 4:
                    target_x = agent.x + DIRECTION[agent_action][0]
                    target_y = agent.y + DIRECTION[agent_action][1]
                    target_name = ITEMNAME[self.map[target_x][target_y]]

                    if target_name == "agent":
                        target_agent = self._findItem(target_x, target_y, target_name)
                        if not target_agent.moved:
                            agent.moved = False
                            target_agent_action = action[AGENTCOLOR.index(target_agent.color)]
                            if target_agent_action < 4:
                                new_target_agent_x = target_agent.x + DIRECTION[target_agent_action][0]
                                new_target_agent_y = target_agent.y + DIRECTION[target_agent_action][1]
                                if new_target_agent_x == agent.x and new_target_agent_y == agent.y:
                                    target_agent.move(new_target_agent_x, new_target_agent_y)
                                    agent.move(target_x, target_y)
                                    agent.moved = True
                                    target_agent.moved = True
                    elif  target_name == "space":
                        self.map[agent.x][agent.y] = ITEMIDX["space"]
                        agent.move(target_x, target_y)
                        self.map[target_x][target_y] = ITEMIDX["agent"]
                    #pickup and chop
                    elif not agent.holding:
                        if target_name == "tomato" or target_name == "lettuce" or target_name == "plate" or target_name == "onion":
                            item = self._findItem(target_x, target_y, target_name)
                            agent.pickup(item)
                            self.map[target_x][target_y] = ITEMIDX["counter"]
                        elif target_name == "knife":
                            knife = self._findItem(target_x, target_y, target_name)
                            if isinstance(knife.holding, Plate):
                                item = knife.holding
                                knife.release()
                                agent.pickup(item)
                            elif isinstance(knife.holding, Food):
                                if knife.holding.chopped:
                                    item = knife.holding
                                    knife.release()
                                    agent.pickup(item)
                                else:
                                    knife.holding.chop()
                                    if knife.holding.chopped:
                                        if knife.holding.rawName in self.task:
                                            reward += self.rewardList["subtask finished"]
                    #put down
                    elif agent.holding:
                        if target_name == "counter":
                            if agent.holding.rawName in ["tomato", "lettuce", "onion", "plate"]:
                                self.map[target_x][target_y] = ITEMIDX[agent.holding.rawName]
                            agent.putdown(target_x, target_y)
                        elif target_name == "plate":
                            if isinstance(agent.holding, Food):
                                if agent.holding.chopped:
                                    plate = self._findItem(target_x, target_y, target_name)
                                    item = agent.holding
                                    agent.putdown(target_x, target_y)
                                    plate.contain(item)
                                    
                        elif target_name == "knife":
                            knife = self._findItem(target_x, target_y, target_name)
                            if not knife.holding:
                                item = agent.holding
                                agent.putdown(target_x, target_y)
                                knife.hold(item)
                            elif isinstance(knife.holding, Food) and isinstance(agent.holding, Plate):
                                item = knife.holding
                                if item.chopped:
                                    knife.release()
                                    agent.holding.contain(item)
                            elif isinstance(knife.holding, Plate) and isinstance(agent.holding, Food):
                                plate_item = knife.holding
                                food_item = agent.holding
                                if food_item.chopped:
                                    knife.release()
                                    agent.pickup(plate_item)
                                    agent.holding.contain(food_item)
                        elif target_name == "delivery":
                            if isinstance(agent.holding, Plate):
                                if agent.holding.containing:
                                    dishName = ""
                                    foodList = [Lettuce, Onion, Tomato]
                                    foodInPlate = [-1] * len(foodList)
                                    
                                    for f in range(len(agent.holding.containing)):
                                        for i in range(len(foodList)):
                                            if isinstance(agent.holding.containing[f], foodList[i]):
                                                foodInPlate[i] = f
                                    for i in range(len(foodList)):
                                        if foodInPlate[i] > -1:
                                            dishName += agent.holding.containing[foodInPlate[i]].rawName + "-"
                                    dishName = dishName[:-1] + " salad"
                                    if dishName == self.task:
                                        item = agent.holding
                                        agent.putdown(target_x, target_y)
                                        self.delivery[0].hold(item)
                                        reward += self.rewardList["correct delivery"]
                                        done = True
                                    else:
                                        reward += self.rewardList["wrong delivery"]
                                        item = agent.holding
                                        agent.putdown(target_x, target_y)
                                        food = item.containing
                                        item.release()
                                        item.refresh()
                                        self.map[item.x][item.y] = ITEMIDX[item.name]
                                        for f in food:
                                            f.refresh()
                                            self.map[f.x][f.y] = ITEMIDX[f.rawName]
                                else:
                                    reward += self.rewardList["wrong delivery"]
                                    plate = agent.holding
                                    agent.putdown(target_x, target_y)
                                    plate.refresh()
                                    self.map[plate.x][plate.y] = ITEMIDX[plate.name]
                            else:
                                reward += self.rewardList["wrong delivery"]
                                food = agent.holding
                                agent.putdown(target_x, target_y)
                                food.refresh()
                                self.map[food.x][food.y] = ITEMIDX[food.rawName]

                        elif target_name in ["tomato", "lettuce", "onion"]:
                            item = self._findItem(target_x, target_y, target_name)
                            if item.chopped and isinstance(agent.holding, Plate):
                                agent.holding.contain(item)
                                self.map[target_x][target_y] = ITEMIDX["counter"]

            all_action_done = True
            for agent in self.agent:
                if agent.moved == False:
                    all_action_done = False
        
        return self._get_obs(), [reward] * self.n_agent, done, info

    def render(self, mode='human'):
        return self.game.on_render()

    





