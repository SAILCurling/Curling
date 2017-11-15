import numpy as np
from Config import Config
import math

"""
STEP_H_TO_X = Config.PLAYAREA_HEIGHT / (Config.IMAGE_HEIGHT - 1)
STEP_W_TO_Y = Config.PLAYAREA_WIDTH / (Config.IMAGE_WIDTH - 1)

def HtoX(h):
    return Config.X_PLAYAREA_MIN + h * STEP_H_TO_X
def WtoY(w):
    return Config.Y_MIN + w * STEP_W_TO_Y
    #return Config.Y_PLAYAREA_MIN + w * STEP_W_TO_Y

def XtoH(x):
    return int((x - Config.X_PLAYAREA_MIN) / STEP_H_TO_X)

def YtoW(y):
    return int((y - Config.Y_MIN) / STEP_W_TO_Y)
    #return int((y - Config.Y_PLAYAREA_MIN) / STEP_W_TO_Y)
""" 



    
STEP_H_TO_X = Config.PLAYAREA_HEIGHT / (Config.IMAGE_HEIGHT)
STEP_W_TO_Y = Config.PLAYAREA_WIDTH / (Config.IMAGE_WIDTH)

def HtoX(h):
    return Config.X_PLAYAREA_MIN + h * STEP_H_TO_X + 0.5 * STEP_H_TO_X
def WtoY(w):
    return Config.Y_MIN + w * STEP_W_TO_Y + 0.5 * STEP_W_TO_Y
    #return Config.Y_PLAYAREA_MIN + w * STEP_W_TO_Y

def XtoH(x):
    if x == Config.X_PLAYAREA_MAX:
        return Config.IMAGE_HEIGHT - 1
    return int((x - Config.X_PLAYAREA_MIN) / STEP_H_TO_X)

def YtoW(y):
    if y == Config.Y_PLAYAREA_MAX:
        return  Config.IMAGE_WIDTH - 1
    return int((y - Config.Y_MIN) / STEP_W_TO_Y)
    

def IsInPlayArea(x, y):
    return (Config.X_PLAYAREA_MIN + Config.STONE_RADIUS < x) and (Config.X_PLAYAREA_MAX - Config.STONE_RADIUS > x) \
        and (Config.Y_PLAYAREA_MIN + Config.STONE_RADIUS < y) and (Config.Y_PLAYAREA_MAX - Config.STONE_RADIUS > y)

def IsInHouse(x, y):
    rx = x - Config.TEE_X
    ry = y - Config.TEE_Y
    R = math.sqrt(rx**2 + ry**2)
    return R < 1.83 + Config.STONE_RADIUS
    
def get_score(gamestate, turn):
    alpha = 0.5
    if Config.SCORE_TYPE == 0: # win 1 # Lose 0
        score = list(gamestate.Score)[0]
        score /= np.abs(score)
        if (turn+1) % 2 == 0:
            score *= -1   
    elif Config.SCORE_TYPE == 1:
        score = list(gamestate.Score)[0]
        if (turn+1) % 2 == 0:
            score *= -1
    elif Config.SCORE_TYPE == 2:
        score = list(gamestate.Score)[0]
        if (turn+1) % 2 == 0:
            score *= -1
        
        opp_count = 0
        for opp in range((turn+1) % 2, Config.MAX_TURN, 2):
            if IsInHouse(gamestate.body[opp][0],gamestate.body[opp][1]):
                opp_count += 1
        score -= alpha * opp_count
    else:
         assert 0, 'Not implemented yet'
        
    return score
    
def state_to_image(state):
    img = np.zeros((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 1), dtype = float)
    for idx in range(16):
        x , y = state[idx][0], state[idx][1]
        if IsInPlayArea(x, y):
            h = XtoH(x)
            w = YtoW(y)

            if idx % 2 == 0: # team who shot first
                img[h][w][0] = 1
            else:
                img[h][w][0] = -1
    return img


def actionID_to_xy(idx):
    h, w = int((idx)/Config.IMAGE_WIDTH), (idx % Config.IMAGE_WIDTH)
    x, y = HtoX(h), WtoY(w)  

    #y -= Config.Y_PLAYAREA_MIN 
    return x, y

def xy_to_actionID(x, y):
    #y += Config.Y_PLAYAREA_MIN 
    
    h = XtoH(x)
    w = YtoW(y)
    idx = Config.IMAGE_WIDTH * h + w
    return idx

def actionID_to_hw(idx):
    h, w = int((idx)/Config.IMAGE_WIDTH), (idx % Config.IMAGE_WIDTH)
    return h, w
    

def hw_to_actionID(h, w):
    idx = Config.IMAGE_WIDTH * h + w
    return idx
 

def actionID_to_xys(idx):
    if idx < Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT:
        spin = 0
    else:
        idx -= Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT
        spin = 1
    
    h, w = int((idx)/Config.IMAGE_WIDTH), (idx % Config.IMAGE_WIDTH)
    x, y = HtoX(h), WtoY(w)  

    #y -= Config.Y_PLAYAREA_MIN 
    return x, y, spin

def xys_to_actionID(x, y, spin):
    #y += Config.Y_PLAYAREA_MIN 
    
    h = XtoH(x)
    w = YtoW(y)
    idx = Config.IMAGE_WIDTH * h + w
    if spin == 1:
        idx += Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT
    return idx

def actionID_to_hws(idx):
    if idx < Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT:
        spin = 0
    else:
        idx -= Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT
        spin = 1

    h, w = int((idx)/Config.IMAGE_WIDTH), (idx % Config.IMAGE_WIDTH)
    return h, w, spin
    

def hws_to_actionID(h, w, spin):
    idx = Config.IMAGE_WIDTH * h + w
    if spin == 1:
        idx += Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT
    return idx
def guard_shot_generate():
    guard_x = np.random.uniform(Config.TEE_X - 1.83, Config.TEE_X + 1.83)
    guard_y = np.random.uniform(Config.TEE_Y + 1.83, Config.TEE_Y + 3*1.83)
    return guard_x, guard_y

def narrow_guard_shot_generate():
    guard_x = np.random.uniform(Config.TEE_X - 1.83/2, Config.TEE_X + 1.83/2)
    guard_y = np.random.uniform(Config.TEE_Y + 1.83, Config.TEE_Y + 2*1.83)
    return guard_x, guard_y

def front_guard_shot():
    return Config.TEE_X, Config.TEE_Y + 1.5 * 1.83

def front_tee_shot():
    return Config.TEE_X, Config.TEE_Y + 0.145