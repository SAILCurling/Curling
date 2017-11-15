from kde import gaussian_kde
from gpwCurlingSimulator_window import Simulation, CreateShot, _GameState, _ShotVec, _ShotPos
from Config import Config
from copy import copy
import numpy as np
from features import MakeFeatures
from utils import *

import time

from multiprocessing import Process, Queue, Value
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self, parent,color):
        self.initialized_actions_xy = {0:[], 1:[]}
        
        self.parent = parent
        self.children = {0:{}, 1:{}}  # 0 is for clock-wise spin, 1 is for counter clock-wise spin
        self.n_visits = 0.
        self.Q = 0.
        self.color = color 
        
        self.rollout_state = None

    def addChild(self, action_xy, color, spin):
        new_node = Node(self, color)
        # init actions
        self.children[spin][tuple(action_xy)] = new_node
        return new_node

    def update(self, leaf_value):
        self.n_visits += 1
        self.Q = ((self.n_visits - 1) * self.Q + leaf_value) / self.n_visits
           

class KR_UCT(object):
    def __init__(self,
                 num_initActions,
                 x_range,
                 y_range,
                 kde,
                 cov_mat,
                 band_width,
                 exec_var,
                 tau,
                 num_samples=100,
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
        
        # kde
        self.kde = kde(cov_mat = cov_mat, band_width = band_width)
        self.cov_mat = cov_mat
        self.band_width = band_width
        
        self.x_range = x_range
        self.y_range = y_range
        self.exec_var = exec_var# execution uncertainty for sampling in expansion step
        self.tau = tau
        self.num_samples = num_samples

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
                node, init_action_xy, init_spin = self.initChildren(node, state, depth)
                _, ShotVec = CreateShot(_ShotPos(init_action_xy[0], init_action_xy[1], init_spin))
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
                node, expanded_action_xy, expanded_spin = self.expand(node)
                _, ShotVec = CreateShot(_ShotPos(expanded_action_xy[0],expanded_action_xy[1], expanded_spin))
                success, ResShot = Simulation(state, ShotVec, Config.RAND, -1)
                isTerminal = (state.ShotNum==0) # one end game
                
                depth += 1
                break

            # select
            node, selected_action_xy, selected_spin = self.ucb_select(node)
            _, ShotVec = CreateShot(_ShotPos(selected_action_xy[0], selected_action_xy[1], selected_spin))
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
        start_time = time.time()
        
        #if state.ShotNum 
        if state.SecondTeamMove:
            root_color = 'Y'
        else:
            root_color = 'R'
       
        self.root = Node(None, root_color)
        for n in range(self.n_playout):
            if self.limit_time is not None and time.time() - start_time > self.limit_time:
                break
            vstate = copy(state)
            self.playout(vstate)
        
        _, best_action, best_spin = self.lcb_select(self.root)
        
        return best_action, best_spin

    
  
        
    # upper confidence bound
    def ucb_select(self, node):
        PREV_UCB_max = -9999
        #selected_node = None
        #selected_action_xy = None
        #selected_spin = None
        for spin in [0,1]:
            A = np.array([i for i in node.children[spin].keys()]) # action sets
            n_A = np.array([i.n_visits for i in node.children[spin].values()]) # n_visits of action set
            v_A = np.array([i.Q for i in node.children[spin].values()]) # empirical value of action set  
            # nominator of E[v|a]
            self.kde.set_dataset(A.T)
            self.kde.set_weights(n_A * v_A)
            nomi_E_v_a = self.kde.evaluate(A.T)

            # W(a) = denominator of E[v|a]
            self.kde.set_dataset(A.T)
            self.kde.set_weights(n_A)
            denomi_E_v_a = self.kde.evaluate(A.T)

            E_v_a = nomi_E_v_a / denomi_E_v_a

            bound = np.array([np.sqrt(np.log(denomi_E_v_a.sum())/ W_a) for W_a in denomi_E_v_a])

            UCB_array = E_v_a + self.uct_const * bound

            UCB_array_max_idx = UCB_array.argmax()
            UCB_max = UCB_array[UCB_array_max_idx]
            
            if UCB_max > PREV_UCB_max:
                PREV_UCB_max = UCB_max
                selected_action_xy = A[UCB_array_max_idx]
                selected_node = node.children[spin][tuple(selected_action_xy)]
                selected_spin = spin
        
        return selected_node, selected_action_xy, selected_spin
    
    # lower confidence bound
    def lcb_select(self, node):
        PREV_LCB_max = -9999
        for spin in [0,1]:
            A = np.array([i for i in node.children[spin].keys()]) # action sets
            n_A = np.array([i.n_visits for i in node.children[spin].values()]) # n_visits of action set
            v_A = np.array([i.Q for i in node.children[spin].values()]) # empirical value of action set

            # nominator of E[v|a]
            self.kde.set_dataset(A.T)
            self.kde.set_weights(n_A * v_A)
            nomi_E_v_a = self.kde.evaluate(A.T)

            # W(a) = denominator of E[v|a]
            self.kde.set_dataset(A.T)
            self.kde.set_weights(n_A)
            denomi_E_v_a = self.kde.evaluate(A.T)

            E_v_a = nomi_E_v_a / denomi_E_v_a

            bound = np.array([np.sqrt(np.log(denomi_E_v_a.sum())/ W_a) for W_a in denomi_E_v_a])

            LCB_array = E_v_a - self.uct_const * bound

            LCB_array_max_idx = LCB_array.argmax()
            LCB_max = LCB_array[LCB_array_max_idx]

            if LCB_max > PREV_LCB_max:
                PREV_LCB_max = LCB_max
                selected_action_xy = A[LCB_array_max_idx]
                selected_node = node.children[spin][tuple(selected_action_xy)]
                selected_spin = spin
        return selected_node, selected_action_xy, selected_spin
        

    
    def expand(self, node):
        _ , selected_action_xy, selected_spin = self.ucb_select(node)
        
        A = [i for i in node.children[selected_spin].keys()] # action sets
        n_A = [i.n_visits for i in node.children[selected_spin].values()] # n_visits of actions sets
        
        sample_action_xy = np.random.multivariate_normal(selected_action_xy, self.exec_var, self.num_samples)
        # keep only action in action range
        sample_action_xy = sample_action_xy[sample_action_xy.T[0] >= self.x_range[0]]
        sample_action_xy = sample_action_xy[sample_action_xy.T[0] <= self.x_range[1]]
        sample_action_xy = sample_action_xy[sample_action_xy.T[1] >= self.y_range[0]]
        sample_action_xy = sample_action_xy[sample_action_xy.T[1] <= self.y_range[1]]
        
        # keep sample which has kernel value bigger than tau
        self.kde.set_dataset(selected_action_xy[np.newaxis,:].T)
        self.kde.set_weights(None)
        kernel_value = self.kde.evaluate(sample_action_xy.T)

        sample_action_xy = sample_action_xy[kernel_value > self.tau]
        
        # W(a) 
        self.kde.set_dataset(selected_action_xy[np.newaxis,:].T)
        self.kde.set_weights(n_A)
        W_a = self.kde.evaluate(sample_action_xy.T)
        
        #W_a_array = np.array([self.compute_W_a(a, A, n_A) for a in sample_action])
        expanded_action_xy = sample_action_xy[W_a.argmin()]
        if node.color == 'R':
            expanded_node = node.addChild(expanded_action_xy, 'Y', selected_spin)
        else:
            expanded_node = node.addChild(expanded_action_xy, 'R', selected_spin)

        return expanded_node, expanded_action_xy, selected_spin
        
        
    
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
        if node.initialized_actions_xy == {0:[], 1:[]}:  
            if self.policy_net is None:
                node.initialized_actions_xy[0] = np.random.multivariate_normal([Config.TEE_X, Config.TEE_Y], (1.83/2)**2*np.identity(2), self.num_initActions).tolist()
                node.initialized_actions_xy[1] = np.random.multivariate_normal([Config.TEE_X, Config.TEE_Y], (1.83/2)**2*np.identity(2), self.num_initActions).tolist()
            else:
                features = self.features_generator.extract_features(state.body, state.ShotNum)
                prediction = self.policy_net.predict_p([features])[0]
                
                # for spin 0
                ## not to initialize draw shots behind the tee line
                spin_0_idx = prediction[:Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT]
                #spin_0_idx = np.reshape(spin_0_idx,(Config.IMAGE_WIDTH,Config.IMAGE_HEIGHT))
                #y = 8
                #y_ = 13
                #spin_0_idx[:,y:y_] = np.min(spin_0_idx)
                #plt.imshow(spin_0_idx)
                #plt.show()
                #spin_0_idx = np.reshape(spin_0_idx,(Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT,))
                spin_0_top_n_action_xy_idx = np.argpartition(spin_0_idx, -(int(self.num_initActions/2)-1))[-(int(self.num_initActions/2)-1):][::-1]
                spin_0_top_n_action_xy = [actionID_to_xy(i) for i in spin_0_top_n_action_xy_idx]
                node.initialized_actions_xy[0] = spin_0_top_n_action_xy
                
                guard_x, guard_y = narrow_guard_shot_generate()
                node.initialized_actions_xy[0].append([guard_x, guard_y])
                
                # for spin 1
                ## not to initialize draw shots behind the tee line
                spin_1_idx = prediction[Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT:]
                #spin_1_idx = np.reshape(spin_1_idx,(Config.IMAGE_WIDTH,Config.IMAGE_HEIGHT))
                #y = 8
                #y_ = 13
                #spin_1_idx[:,y:y_] = np.min(spin_1_idx)
                #plt.imshow(spin_1_idx)
                #plt.show()
                #spin_1_idx = np.reshape(spin_1_idx,(Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT,))
                
                spin_1_top_n_action_xy_idx = np.argpartition(spin_1_idx, -(int(self.num_initActions/2)-1))[-(int(self.num_initActions/2)-1):][::-1]
                spin_1_top_n_action_xy = [actionID_to_xy(i) for i in spin_1_top_n_action_xy_idx]
                node.initialized_actions_xy[1] = spin_1_top_n_action_xy
                
                guard_x, guard_y = narrow_guard_shot_generate()
                node.initialized_actions_xy[1].append([guard_x, guard_y])
                
        if len(node.initialized_actions_xy[0]) == len(node.initialized_actions_xy[1]):
            init_action_xy = node.initialized_actions_xy[0].pop(0)
            init_spin = 0
        else:
            init_action_xy = node.initialized_actions_xy[1].pop(0)
            init_spin = 1
            
    
        #init_action_xys = node.initialized_actions.pop(0)
        #init_action_xy = init_action_xys[:2]
        #init_spin = init_action_xys[-1]
        
        if node.color == 'R':
            init_node = node.addChild(init_action_xy, 'Y', init_spin)
        else:
            init_node = node.addChild(init_action_xy, 'R', init_spin)
        #print(init_action_xy, init_spin)
        return init_node, init_action_xy, init_spin
        

        

