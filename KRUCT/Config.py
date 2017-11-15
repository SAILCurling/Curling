

class Config:

    #########################################################################
    
    
    
    
    # Name of the game, with version (e.g. PongDeterministic-v0)
#    ATARI_GAME = 'PongDeterministic-v0'
#    ATARI_GAME = 'Breakout-v0'
    
    
  
    
    """FAST LEARNING"""
    USE_ACTION_PROPAGATE = True
    ACTION_PROPAGATE_RANGE = 1
    
    SCORE_TYPE = 1
    
    
    
    DEFAULT_FEATURES = [
        "stone_color_feature",
        "all_ones_feature",
        "playground_ones_feature",
        "turn_num"
   #     "remaining_stones",
   #     "collision_spin_0_feature"
    ]
    
    
    
    MAX_TURN = 2# MUST BE EVEN NUMBERS
    RAND = 0
    """currently fix the spin as 0 """
    SPIN = 0 
    # with probability 0.1, try takeout shot
    USE_GUARD_SHOT_GEN = True
    # CNN
    NUM_CONV_LAYERS = 3
    NUM_FILTER = 32
    FILTER_SIZE = 3
    
    
    

    
    # Enable to see the trained agent in action
    PLAY_MODE = False
    # Enable to train
    TRAIN_MODELS = True
    # Load old models. Throws if the model doesn't exist
    LOAD_CHECKPOINT = True
    # If 0, the latest checkpoint is loaded
    LOAD_EPISODE = 0 

    # Device
    DEVICE = 'gpu:1'

    #########################################################################
    # Algorithm parameters

    # Discount factor
    DISCOUNT = 0.99
    
    # Tmax
    #TIME_MAX = 5
    
    # Reward Clipping
    #REWARD_MIN = -1
    #REWARD_MAX = 1

    # Max size of the queue
    MAX_QUEUE_SIZE = 100
    PREDICTION_BATCH_SIZE = 128

    # Input of the DNN
    #IMAGE_WIDTH = 80
    IMAGE_WIDTH = 30
    #IMAGE_HEIGHT = 30
    IMAGE_HEIGHT = 30
    #NUM_INPUT_PLAINS = 1

    # Total number of episodes and annealing frequency
    EPISODES = 10000000
    ANNEALING_EPISODE_COUNT = 400000

    # Entropy regualrization hyper-parameter
    BETA_START = 0.01
    BETA_END = BETA_START

    # Learning rate
    #LEARNING_RATE_START = 0.0003
    LEARNING_RATE_START = 0.03
    LEARNING_RATE_END = LEARNING_RATE_START

    # RMSProp parameters
    RMSPROP_DECAY = 0.99
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.1

    # Dual RMSProp - we found that using a single RMSProp for the two cost function works better and faster
    DUAL_RMSPROP = False
    
    # Gradient clipping
    USE_GRAD_CLIP = False
    GRAD_CLIP_NORM = 40.0 
    # Epsilon (regularize policy lag in GA3C)
    LOG_EPSILON = 1e-6
    # Training min batch size - increasing the batch size increases the stability of the algorithm, but make learning slower
    TRAINING_MIN_BATCH_SIZE = 64
    
    #########################################################################
    # Log and save

    # Enable TensorBoard
    TENSORBOARD = True
    # Update TensorBoard every X training steps 
    TENSORBOARD_UPDATE_FREQUENCY = 1000

    # Enable to save models every SAVE_FREQUENCY episodes
    SAVE_MODELS = True
    # Save every SAVE_FREQUENCY episodes
    SAVE_FREQUENCY = 100000
    
    # Print stats every PRINT_STATS_FREQUENCY episodes
    PRINT_STATS_FREQUENCY = 500
    # The window to average stats
    STAT_ROLLING_MEAN_WINDOW = 100

    # Results filename
    #RESULTS_FILENAME = 'results.txt'
    # Network checkpoint name
    NETWORK_NAME = 'network'

    #########################################################################
    # More experimental parameters here
    
    # Minimum policy
    MIN_POLICY = 0.0
    # Use log_softmax() instead of log(softmax())
    USE_LOG_SOFTMAX = False
    
    
    
    """GPW Curling Game configuration"""
    X_PLAYAREA_MIN = 0
    X_PLAYAREA_MAX = 4.75
    Y_PLAYAREA_MIN = 3.05
    Y_PLAYAREA_MAX = 3.05 + 8.23
    Y_MIN = 0 # To limit the max power of the shot
    
    PLAYAREA_HEIGHT = X_PLAYAREA_MAX - X_PLAYAREA_MIN
    #PLAYAREA_WIDTH = Y_PLAYAREA_MAX - Y_PLAYAREA_MIN
    PLAYAREA_WIDTH = Y_PLAYAREA_MAX - Y_MIN
    
    TEE_X = 2.375
    TEE_Y = 4.88
    
    STONE_RADIUS = 0.145
    
    ## added
    USE_BATCH_NORM = True
    NUM_CONV_LAYERS=9
    DEVICE = 'gpu:0'
    PLAY_MODE = True    

