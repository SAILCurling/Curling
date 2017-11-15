from ctypes import *
from ctypes import Structure, POINTER
import numpy as np
import time
CurlingSimulator_API = cdll.LoadLibrary("./dll/CurlingSimulator_vec2pos.dll")
#CurlingSimulator_API = cdll.LoadLibrary("./dll/fast_CurlingSimulator.dll")

class _GameState(Structure):
    _fields_ = [("ShotNum", c_int), # Number of current shots, ShotNum Is n, the next shot is the n + 1 th shot
               ("CurEnd", c_int), # Current end number (0~15)
               ("LastEnd", c_int), # Final end number
#               ("Score", POINTER(c_int)), 
               ("Score", c_int * 10), # Score from 1st to 10th end / 
               ("SecondTeamMove", c_bool), # Information on turn 
               ("body", (c_float * 2) * 16)]# 
    
# Shot information (coordinates)
class _ShotPos(Structure):
    _fields_ = [("x", c_float),
               ("y", c_float),
               ("spin", c_bool)]
    
# Shot information (intensity vector)
class _ShotVec(Structure):
    _fields_ = [("x", c_float),
               ("y", c_float),
               ("spin", c_bool)]


"""Simulation function 

    GAMESTATE - Phase information before simulation
    Shot - Shot vector for simulation
    Rand - the size of the random number
    LoopCount - how many frames to simulate (specify -1 to do simulation to the end)
    
    Return 
        - success - 0 is returned if the Simulation function fails. Other values will be returned if it succeeds
        - ResShot - Shot vector actually used in simulation (value of shot vector with random number added)
    

    

"""
def Simulation(GAMESTATE, Shot, Rand, ResShot, LoopCount = -1):
    ResShot = _ShotVec()
    success = CurlingSimulator_API.Simulation(byref(GAMESTATE),
                                              Shot,
                                              c_float(Rand),
                                              byref(ResShot),
                                              c_int(LoopCount))
    return success, ResShot

"""Simulation function (extended version)

GAMESTATE - Phase information before simulation
    Shot - Shot vector for simulation
    RandX - size of horizontal random number
    RandY - Size of vertical random number
    trj - an array that receives the simulation result (trajectory)
        The position coordinates (32 sets of X and Y together) of the stone per frame are returned as a one-dimensional array
    ResLociSize - the maximum size of the array receiving the simulation result (trajectory) (If the simulation result exceeds this size, the result is stored up to this size)
    
    Returned 
        - success - 0 is returned if the Simulation function fails. Other values will be returned if it succeeds
        - ResShot - Shot vector actually used in simulation (value of shot vector with random number added)
        - trj - trajectories of all stones.
"""
def SimulationEx(GAMESTATE, Shot, RandX, RandY, ResShot, trj = (((c_float * 2) * 16) * 4000)(), ResLociSize = 1.0E6):
#     trj = trj.ctypes.data_as(POINTER(c_float))
    ResShot = _ShotVec()
    success = CurlingSimulator_API.SimulationEx(byref(GAMESTATE),
                                              Shot,
                                              c_float(RandX),
                                              c_float(RandY),
                                              byref(ResShot),
                                              byref(trj),
#                                               pLoci,
                                              c_int(int(ResLociSize)))
    return success,  ResShot, np.array(trj)

"""Shot generation function (draw shot)

   SHOTPOS ShotPos - Specify the coordinates. A shot that stops at the coordinates specified here will be generated
   
   Return
       success - 0 will be returned if the CreateShot function fails. Other values will be returned if it succeeds
       SHOTVEC * lpResShotVec - Specify the address to receive the generated shot
"""
def CreateShot(ShotPos):
    ResShotVec = _ShotVec()
    success = CurlingSimulator_API.CreateShot(ShotPos, byref(ResShotVec))
    return success, ResShotVec
    
    

               



