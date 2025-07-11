import copy

ITEMIDX= {"space": 0, "counter": 1, "agent": 2, "tomato": 3, "lettuce": 4, "plate": 5, "knife": 6, "delivery": 7, "onion": 8}
POSSIBLE_MAP_TYPE = ['A', 'B', 'C']
POSSIBLE_N_AGENT = [2, 3]


def make_map(map_type: str, n_agent: int, size: int):
    assert map_type in POSSIBLE_MAP_TYPE, f'map_type should be one of {POSSIBLE_MAP_TYPE}'
    assert n_agent in POSSIBLE_N_AGENT, f'n_agent should be one of {POSSIBLE_N_AGENT}'

    # Make default map
    map = [[ITEMIDX['space']] * size for _ in range(size)]
    map[0][0] = ITEMIDX['counter']
    map[0][size-1] = ITEMIDX['counter']
    map[size-1][0] = ITEMIDX['counter']
    map[size-1][size-1] = ITEMIDX['counter']

    for i in range(1, size-1):
        map[i][0] = ITEMIDX['counter']
        map[0][i] = ITEMIDX['counter']
        map[i][size-1] = ITEMIDX['counter']
        map[size-1][i] = ITEMIDX['counter']

    # Change map corresponds to map_type
    if map_type == 'A':
        pass
    elif map_type == 'B':
        for i in range(1, size-1):
            map[i][int(size//2)] = ITEMIDX['counter']
    elif map_type == 'C':
        for i in range(1, size-2):
            map[i][int(size//2)] = ITEMIDX['counter']

    pomap = copy.deepcopy(map)
    
    # Locate other objects
    map[0][size-2] = ITEMIDX['tomato']
    map[1][0] = ITEMIDX['knife']
    map[1][size-1] = ITEMIDX['lettuce']
    map[2][0] = ITEMIDX['knife']
    map[2][size-1] = ITEMIDX['onion']
    map[3][0] = ITEMIDX['delivery']
    map[size-2][size-1] = ITEMIDX['plate']
    map[size-1][size-2] = ITEMIDX['plate']

    # Locate agents
    if n_agent == 2:
        map[1][2] = ITEMIDX['agent']
        map[1][size-3] = ITEMIDX['agent']
    elif n_agent == 3:
        map[1][2] = ITEMIDX['agent']
        map[1][size-3] = ITEMIDX['agent']
        map[size-2][2] = ITEMIDX['agent']
    
    return map, pomap



def make_custom_map(map_type: str, n_agent: int, size: int,item_idx,obj, n_plate, n_knife):
    assert map_type in POSSIBLE_MAP_TYPE, f'map_type should be one of {POSSIBLE_MAP_TYPE}'
    
    # Make default map
    map = [[item_idx['space']] * size for _ in range(size)]
    map[0][0] = item_idx['counter']
    map[0][size-1] = item_idx['counter']
    map[size-1][0] = item_idx['counter']
    map[size-1][size-1] = item_idx['counter']

    for i in range(1, size-1):
        map[i][0] = item_idx['counter']
        map[0][i] = item_idx['counter']
        map[i][size-1] = item_idx['counter']
        map[size-1][i] = item_idx['counter']

    # Change map corresponds to map_type
    if map_type == 'A':
        pass
    elif map_type == 'B':
        for i in range(1, size-1):
            map[i][int(size//2)] = item_idx['counter']
    elif map_type == 'C':
        for i in range(1, size-2):
            map[i][int(size//2)] = item_idx['counter']

    pomap = copy.deepcopy(map)
    
    #locate objects
    init_r, current_r = 0,0
    init_c, current_c = size-1,size-1
    for i, o in enumerate(obj) :
        if i%2 == 0 :
            current_r = current_r+1
            assert current_r < size, 'too many objects'
            map[current_r][init_c] = item_idx[o]
            
        else :
            current_c = current_c -1
            assert current_c > -1 , 'too many objects'
            map[init_r][current_c] = item_idx[o]
            
    
    #locate delievery 
    map[size//2][0] = item_idx['delivery']
    
    #locate knife
    knife_pos = 1
    for i in range(0,n_knife) :
        assert knife_pos < size, 'too many knifes'
        if map[knife_pos][0] == item_idx['counter'] :
            map[knife_pos][0] = item_idx['knife']
            knife_pos+=1
        else :
            knife_pos+=1
            map[knife_pos][0] = item_idx['knife']
            knife_pos+=1
    
    #locate plates
    init_r, current_r = size-1,size-1
    init_c, current_c = size-1,size-1
    for i in range(0,n_plate) :
        if i%2 == 0 :
            current_c = current_c -1
            assert current_c > -1 , 'too many platess'
            assert map[init_r][current_c] == 1, 'too many plates'
            map[init_r][current_c] = item_idx['plate']
        else :
            current_r = current_r -1
            assert current_r > -1, 'too many plates'
            assert map[current_r][init_c] == 1, 'too many plates'
            map[current_r][init_c] = item_idx['plate']
    
    # Locate agents
    
    current_0 = 0
    current_1 = 0
    current_2 = size-1
    current_3 = size-1
    
    for i in range(0,n_agent):
        if i%4 == 0:
            current_0 = current_0 +1 
            assert map[current_0][int(size//2)-1] == 0 , 'too many agents'
            map[current_0][int(size//2)-1] = item_idx['agent']
        elif i%4 == 1 :
            current_1 = current_1 +1
            assert map[current_1][int(size//2)+1] == 0 , 'too many agents'
            map[current_1][int(size//2)+1] = item_idx['agent']
        elif i%4 == 2 :
            current_2 = current_2 -1
            assert map[current_2][int(size//2)-1]== 0 , 'too many agents'
            map[current_2][int(size//2)-1] = item_idx['agent']
        else :
            current_3 = current_3 -1
            assert map[current_3][int(size//2)+1] == 0 , 'too many agents'
            map[current_3][int(size//2)+1] = item_idx['agent']
    
    return map, pomap


def print_map(map):
    for row in map:
        print(row)


def make_counter_seq(size, map_type):
    assert map_type in POSSIBLE_MAP_TYPE, f'map_type should be one of {POSSIBLE_MAP_TYPE}'

    start = int(size // 2)
    seq = [start]
    
    if map_type == 'A':
        return []
    
    if size % 2 == 0:
        for i in range(1, int((size-1) // 2)):
            seq.extend([start -i, start+i])
        if map_type == 'B':
            seq.append(1)
    else:    
        for i in range(1, int(size // 2)):
            seq.extend([start -i, start+i])
        if map_type == 'C':
            seq.remove(size-2)
    return seq


if __name__ == '__main__':
    size = 7
    n_agent = 3
    for map_type in POSSIBLE_MAP_TYPE:
        print(f'map {map_type}:')
        map, pomap = make_map(size=size, n_agent=n_agent, map_type=map_type)
        print_map(map)
        print('-')
        print_map(pomap)
        print('==============')





