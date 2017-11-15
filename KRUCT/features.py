from gpwCurlingSimulator_window import Simulation, CreateShot, _GameState, _ShotVec, _ShotPos
from utils import *
from Config import Config
import os

def planes(num_planes):
    def deco(f):
        f.planes = num_planes
        return f
    return deco

class MakeFeatures:
    def __init__(self, default_features = Config.DEFAULT_FEATURES):
        self.default_features = default_features
        if not os.path.exists('./table'):
            os.makedirs('./table')
            
        if ("collision_spin_0_feature" in self.default_features) or \
            ("collision_spin_1_feature" in self.default_features):
            self.init_table()
        
        self.feature_dct = {"stone_color_feature" : self.stone_color_feature,
                            "all_ones_feature" : self.all_ones_feature,
                            "playground_ones_feature" : self.playground_ones_feature,
                            #"remaining_stones" : self.remaining_stones,
                            "turn_num" : self.turn_num,
                            "collision_spin_0_feature" : self.collision_spin_0_feature}
        
    def init_table(self):
        if os.path.exists(os.path.join('table', 'collision_table.npy')):
            self.collision_table = np.load(os.path.join('table', 'collision_table.npy'))
            return
        
        
        """
            This table contains collision info asumming spin = 0 (clock wise)
        """
        
        self.collision_table = [[],[]]
        for spin in [0,1]:
            for h in range(Config.IMAGE_HEIGHT):
                print(h)
                for w in range(Config.IMAGE_WIDTH):
                    candidates = []

                    for target_h in range(h - 1, h + 5):
                        for target_w in range(w + 2):
                            if (target_h > Config.IMAGE_HEIGHT-1) or (target_w > Config.IMAGE_WIDTH-1):
                                break
                            
                            
                            random = 0
                            game_state = _GameState()
                            game_state.LastEnd = 1 # 1-end gam
                            game_state.body[0][0] = HtoX(h)
                            game_state.body[0][1] = WtoY(w)
                            game_state.WhiteToMove = not(game_state.WhiteToMove)
                            game_state.ShotNum += 1

                            prev_x = game_state.body[0][0]            

                            _, ShotVec = CreateShot(_ShotPos(HtoX(target_h), WtoY(target_w), spin))

                            success, ResShot = Simulation(game_state, ShotVec, 0., -1)
                            next_x = game_state.body[0][0]
                            #print([list(i) for i in list(game_state.body)])
                            if prev_x != next_x:
                                candidates.append([target_h, target_w])
                    self.collision_table[spin].append(candidates)
        np.save('./table/collision_table', self.collision_table)
        
    @planes(3)
    def stone_color_feature(self, position, turn):
        features = np.zeros((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 3))
        features[:,:,2] = 1.
        for idx in range(16):
            x , y = position[idx][0], position[idx][1]
            if IsInPlayArea(x, y):
                h, w = XtoH(x), YtoW(y)

                if idx % 2 == 0: # team who shot first
                    features[h][w][0] = 1.
                    features[h][w][2] = 0. # not empty
                else:
                    features[h][w][1] = 1.
                    features[h][w][2] = 0. # not empty
        return features
    
    @planes(1)
    def all_ones_feature(self, position, turn):
        return np.ones((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 1))
    
    @planes(1)
    def playground_ones_feature(self, position, turn):
        features = np.ones((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 1))
        features[:,:YtoW(Config.Y_PLAYAREA_MIN),0] = 0
        return features
    
    @planes(16)
    def turn_num(self, position, turn):
        features = np.zeros((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 16))
        features[:,:, turn] = 1
        return features
    
    """
    @planes(8)
    def remaining_stones(self, position, remaining_stones):
        features = np.zeros((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 8))
        features[:,:, remaining_stones - 1] = 1
        return features
    """
    
    @planes(2)
    def collision_spin_0_feature(self, position, turn):
        features = np.zeros((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 2))

        for idx in range(16):
            x , y = position[idx][0], position[idx][1]
            if IsInPlayArea(x, y):
                h, w = XtoH(x), YtoW(y)
                for th, tw in self.collision_table[0][h*Config.IMAGE_WIDTH + w]:
                    if idx % 2 == 0:
                        features[th][tw][0] = 1
                    else:
                        features[th][tw][1] = 1
                    
        return features
        
        """
        features_1 = np.zeros((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 1))
        features_2 = np.zeros((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 1))
        for idx in range(16):
            x , y = position[idx][0], position[idx][1]
            if IsInPlayArea(x, y):
                h, w = XtoH(x), YtoW(y)
                if idx % 2 == 0:
                    features_1[zip(*self.collision_table[0][h*Config.IMAGE_WIDTH + w])] = 1
                else:
                    features_2[zip(*self.collision_table[0][h*Config.IMAGE_WIDTH + w])] = 1
                    
        return np.concatenate([features_1,features_2], axis = 2)
        """
        
    def extract_features(self, position, turn):
        return np.concatenate([self.feature_dct[feature](position, turn) for feature in self.default_features], axis=2)
        
        