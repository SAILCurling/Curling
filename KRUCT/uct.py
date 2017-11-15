from gpwCurlingSimulator_window import Simulation, CreateShot, _GameState, _ShotVec, _ShotPos
from Config import Config
from copy import copy
import numpy as np
from features import MakeFeatures
from utils import *

import time

from multiprocessing import Process, Queue, Value


MOVE = [(-1,  1), (0,  1), (1,  1),
        (-1,  0), (0,  0), (1,  0),
        (-1, -1), (0, -1), (1, -1)]

TRANS_PROB_TAKEOUT = [0.025, 0.400, 0.025,
                      0.050, 0.000, 0.050,
                      0.025, 0.400, 0.025]

TRANS_PROB_DRAW = [0.125, 0.125, 0.125,
                   0.125, 0.000, 0.125,
                   0.125, 0.125, 0.125]

TRANS_PROB_GUARD = [0.025, 0.050, 0.025,
                    0.400, 0.00, 0.400,
                    0.025, 0.050, 0.025]

class Node(object):
    def __init__(self, parent,color):
        self.initialized_actions_hw = {0:[], 1:[]}
        
        self.parent = parent
        self.children = {0:{}, 1:{}}  # 0 is for clock-wise spin, 1 is for counter clock-wise spin
        self.n_visits = 0.
        self.Q = 0.
        self.color = color 
        
        self.rollout_state = None

    def addChild(self, action_hw, color, spin):
        new_node = Node(self, color)
        # init actions
        self.children[spin][tuple(action_hw)] = new_node
        return new_node

    def update(self, leaf_value):
        self.n_visits += 1
        self.Q = ((self.n_visits - 1) * self.Q + leaf_value) / self.n_visits
           

class UCT(object):
    def __init__(self,
                 num_initActions,
                 x_range,
                 y_range,
                 playout_depth=5,
                 n_playout=100,
                 uct_const=1,
                 policy_net=None,
                 limit_time=None):
        
        assert playout_depth >= 1 and playout_depth <= 16, 'playout_depth should be (1 ~ 16)'
        assert num_initActions >=4 , 'num_initActions should be greater than or equal to 4'
        
        self.policy_net = policy_net
        
        self.features_generator = MakeFeatures()
        
        self.num_initActions = num_initActions

        self.playout_depth = playout_depth
        self.n_playout = n_playout
        self.uct_const = uct_const
        
        self.x_range = x_range
        self.y_range = y_range

        self.limit_time = limit_time
    
    def playout(self, state):
        """
            state  -- a copy of the state.
        
        """
        node = self.root
        isTerminal = False
        depth = 0

        while not isTerminal and depth < self.playout_depth:
            #A = len(node.children) # num_children
            A = len(node.children[0]) + len(node.children[1]) 
            if A < self.num_initActions:
            #if len(node.children[0]) < self.num_initActions:
                node, init_action_hw, init_spin = self.initChildren(node, state, depth)
                _, ShotVec = CreateShot(_ShotPos(HtoX(init_action_hw[0]), WtoY(init_action_hw[1]), init_spin))
                success, ResShot = Simulation(state, ShotVec, Config.RAND, -1)
                isTerminal = (state.ShotNum==0)
        
                depth += 1
                break
                
            n_a = [c.n_visits for c in node.children[0].values()] + [c.n_visits for c in node.children[1].values()]
            # progressive widening
            # if chilren node has been visited much times then expand
            #if np.sqrt(sum(n_a)) >= A:
            if sum(n_a) >=  10 * A:    
                # expand
                node, expanded_action_hw, expanded_spin = self.expand(node)
                _, ShotVec = CreateShot(_ShotPos(HtoX(expanded_action_hw[0]), WtoY(expanded_action_hw[1]), expanded_spin))
                success, ResShot = Simulation(state, ShotVec, Config.RAND, -1)
                isTerminal = (state.ShotNum==0) # one end game
                
                depth += 1
                break

            # select
            node, selected_action_hw, selected_spin = self.ucb_select(node)
            _, ShotVec = CreateShot(_ShotPos(HtoX(selected_action_hw[0]), WtoY(selected_action_hw[1]), selected_spin))
            success, ResShot = Simulation(state, ShotVec, Config.RAND, -1)
            isTerminal = (state.ShotNum==0) # one end game
            
            depth += 1

            if isTerminal:
                break

        if not isTerminal and depth < self.playout_depth:
            # save the rollout_state for speed.
            #if node.rollout_state is None:
            state = self.rollOut(node, state, depth)
            #node.rollout_state = state
            #else:
            #    state = node.rollout_state
      
        self.update(node, state)

    def get_move(self, state):
        
        #if state.ShotNum 
        if state.SecondTeamMove:
            root_color = 'Y'
        else:
            root_color = 'R'
  
        self.root = Node(None, root_color)
        for n in range(self.n_playout):
            vstate = copy(state)
            self.playout(vstate)
        
        _, best_action, best_spin = self.lcb_select(self.root)
        
        return best_action, best_spin

    # upper confidence bound
    def ucb_select(self, node):
        PREV_UCB_max = -9999

        tot_n_A = np.array([i.n_visits for i in node.children[0].values()] + [i.n_visits for i in node.children[1].values()]).sum()
        
        for spin in [0,1]:
            A = np.array([i for i in node.children[spin].keys()]) # action sets
            n_A = np.array([i.n_visits for i in node.children[spin].values()]) # n_visits of action set
            v_A = np.array([i.Q for i in node.children[spin].values()]) # empirical value of action set  
            bound = np.array([np.sqrt(np.log(tot_n_A)/ n_a) for n_a in n_A])
            UCB_array = v_A + self.uct_const * bound

            UCB_array_max_idx = UCB_array.argmax()
            UCB_max = UCB_array[UCB_array_max_idx]
            
            if UCB_max > PREV_UCB_max:
                PREV_UCB_max = UCB_max
                selected_action_hw = A[UCB_array_max_idx]
                selected_node = node.children[spin][tuple(selected_action_hw)]
                selected_spin = spin
        
        return selected_node, selected_action_hw, selected_spin
    
    # lower confidence bound
    def lcb_select(self, node):
        PREV_LCB_max = -9999
        for spin in [0,1]:
            A = np.array([i for i in node.children[spin].keys()]) # action sets
            n_A = np.array([i.n_visits for i in node.children[spin].values()]) # n_visits of action set
            v_A = np.array([i.Q for i in node.children[spin].values()]) # empirical value of action set
            bound = np.array([np.sqrt(np.log(n_A.sum())/ n_a) for n_a in n_A])
            LCB_array = v_A - self.uct_const * bound

            LCB_array_max_idx = LCB_array.argmax()
            LCB_max = LCB_array[LCB_array_max_idx]

            if LCB_max > PREV_LCB_max:
                PREV_LCB_max = LCB_max
                selected_action_hw = A[LCB_array_max_idx]
                selected_node = node.children[spin][tuple(selected_action_hw)]
                selected_spin = spin
        return selected_node, selected_action_hw, selected_spin

    def expand(self, node):
        _ , selected_action_hw, selected_spin = self.ucb_select(node)
        
        expanded_action_hw = copy(selected_action_hw)
        while tuple(expanded_action_hw) in node.children[selected_spin].keys():
            if expanded_action_hw[1] < Config.TEE_Y - 1.83:
                trans_prob = TRANS_PROB_TAKEOUT
            elif expanded_action_hw[1] < Config.TEE_Y + 1.83:
                trans_prob = TRANS_PROB_DRAW
            else:
                trans_prob = TRANS_PROB_GUARD
            move_h, move_w = MOVE[np.random.choice(range(len(MOVE)), p = trans_prob)]
            expanded_action_hw = (expanded_action_hw[0]+move_h, expanded_action_hw[1]+move_w)

        if node.color == 'R':
            expanded_node = node.addChild(expanded_action_hw, 'Y', selected_spin)
        else:
            expanded_node = node.addChild(expanded_action_hw, 'R', selected_spin)

        return expanded_node, expanded_action_hw, selected_spin

    def rollOut(self, node, state, depth):
        isTerminal = False
        while not isTerminal and depth < self.playout_depth:
            if self.policy_net is None:
                sample_x, sample_y = np.random.multivariate_normal([Config.TEE_X, Config.TEE_Y], (1.83/2)**2*np.identity(2),1)[0]
                sample_spin = np.random.choice([0,1])
            else:
                features = self.features_generator.extract_features(state.body, state.ShotNum)
                prediction = self.policy_net.predict_p([features])[0]
                sample_action_idx = np.random.choice(np.arange(2 * Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT), p=prediction)
                sample_x, sample_y, sample_spin = actionID_to_xys(sample_action_idx)

            _, ShotVec = CreateShot(_ShotPos(sample_x, sample_y, sample_spin))
            success, ResShot = Simulation(state, ShotVec, Config.RAND, -1)

            isTerminal = (state.ShotNum==0)

            depth+=1

        node.rollout_state = copy(state)

        return state

    def update(self,node, state):
        # Backpropagate update
        if state.ShotNum == 0:
            value = list(state.Score)[state.CurEnd-1]
        else:
            value = list(state.Score)[state.CurEnd]

        while node.parent != None:
            # since score is based on FirstTeam
            if node.color == 'R':
                node.update(-value)
            else:
                node.update(value)
            node = node.parent
            
        # update n_visits of root node 
        node.update(0)       

    def initChildren(self, node, state, depth):
        # when the case, first initialization is started
        # one for guard.
        #if node.initialized_actions == []:
        if node.initialized_actions_hw == {0:[], 1:[]}:  
            if self.policy_net is None:
                init_h, init_w = XtoH(Config.TEE_X), YtoW(Config.TEE_Y)
                while len(node.initialized_actions_hw[0]) < self.num_initActions:
                    if [init_h, init_w] not in node.initialized_actions_hw[0]:
                        node.initialized_actions_hw[0].append([init_h, init_w])
                    move_h, move_w = MOVE[np.random.choice(range(len(MOVE)), p = TRANS_PROB_TAKEOUT)]
                    init_h, init_w = init_h+move_h, init_w+move_w
                    
                init_h, init_w = XtoH(Config.TEE_X), YtoW(Config.TEE_Y)
                while len(node.initialized_actions_hw[1]) < self.num_initActions:
                    if [init_h, init_w] not in node.initialized_actions_hw[1]:
                        node.initialized_actions_hw[1].append([init_h, init_w])
                    move_h, move_w = MOVE[np.random.choice(range(len(MOVE)), p = TRANS_PROB_TAKEOUT)]
                    init_h, init_w = init_h+move_h, init_w+move_w
                
            else:
                features = self.features_generator.extract_features(state.body, state.ShotNum)
                prediction = self.policy_net.predict_p([features])[0]
                
                # for spin 0
                spin_0_top_n_action_hw_idx = np.argpartition(prediction[:Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT], -(int(self.num_initActions/2)-1))[-(int(self.num_initActions/2)-1):][::-1]
                spin_0_top_n_action_hw = [actionID_to_hw(i) for i in spin_0_top_n_action_hw_idx]
                node.initialized_actions_hw[0] = spin_0_top_n_action_hw
                
                guard_x, guard_y = front_guard_shot()
                node.initialized_actions_hw[0].append([XtoH(guard_x), YtoW(guard_y)])
                
                # for spin 1
                spin_1_top_n_action_hw_idx = np.argpartition(prediction[Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT:], -(int(self.num_initActions/2)-1))[-(int(self.num_initActions/2)-1):][::-1]
                spin_1_top_n_action_hw = [actionID_to_hw(i) for i in spin_1_top_n_action_hw_idx]
                node.initialized_actions_hw[1] = spin_1_top_n_action_hw
                
                tee_x, tee_y = front_tee_shot()
                node.initialized_actions_hw[1].append([XtoH(tee_x), YtoW(tee_y)])
                
        if len(node.initialized_actions_hw[0]) == len(node.initialized_actions_hw[1]):
            init_action_hw = node.initialized_actions_hw[0].pop(0)
            init_spin = 0
        else:
            init_action_hw = node.initialized_actions_hw[1].pop(0)
            init_spin = 1
            
    
        #init_action_xys = node.initialized_actions.pop(0)
        #init_action_xy = init_action_xys[:2]
        #init_spin = init_action_xys[-1]
        
        if node.color == 'R':
            init_node = node.addChild(init_action_hw, 'Y', init_spin)
        else:
            init_node = node.addChild(init_action_hw, 'R', init_spin)
        
        return init_node, init_action_hw, init_spin
        

        

