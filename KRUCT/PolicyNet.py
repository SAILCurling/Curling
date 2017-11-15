import os
import re
import numpy as np
import tensorflow as tf
from features import MakeFeatures
from Config import Config

from tensorflow.python.training import moving_averages


import functools, operator, math

def product(numbers):
    return functools.reduce(operator.mul, numbers)

#convenience functions for initializing weights and biases
def _weight_variable(shape, name):
    # If shape is [5, 5, 20, 32], then each of the 32 output planes
    # has 5 * 5 * 20 inputs.
    number_inputs_added = product(shape[:-1])
    stddev = 1 / math.sqrt(number_inputs_added)
    # http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

def _conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")


class PolicyNet:
    def __init__(self, device, model_name, train_models):
        self.train_models = train_models
        
        self.device = device
        self.model_name = model_name
        self.num_actions = Config.IMAGE_WIDTH * Config.IMAGE_HEIGHT * 2 

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = sum(getattr(MakeFeatures, f).planes for f in Config.DEFAULT_FEATURES)
        #self.img_channels = Config.NUM_INPUT_PLAINS
        
        self.num_conv_layers = Config.NUM_CONV_LAYERS
        self.num_filter = Config.NUM_FILTER
        self.filter_size = Config.FILTER_SIZE

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        
        self.use_batch_norm = Config.USE_BATCH_NORM

        self.model_dir = self.get_model_dir(["model_name", "use_batch_norm", "learning_rate", "num_conv_layers",
                                            "num_filter","filter_size"])
        
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                
                self.extra_train_ops = []
                
                #self._create_graph()
                self._create_placeholder()
                self._create_policy_network()
                #vars = tf.global_variables()
                #self.saver_p = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
                self._create_train_op()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
                
    def batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
              'beta', params_shape, tf.float32,
              initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
              'gamma', params_shape, tf.float32,
              initializer=tf.constant_initializer(1.0, tf.float32))

            if self.train_models == True:
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self.extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self.extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
                #tf.summary.histogram(mean.op.name, mean)
                #tf.summary.histogram(variance.op.name, variance)
            # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(
                  x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y
        
    def _create_placeholder(self):
        self.x = tf.placeholder(
            tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
        
        self.action_index = tf.placeholder(tf.float32, [None, self.num_actions])

        self.global_step = tf.Variable(0, trainable=False, name='step')
        
    def _create_policy_network(self):
        W_conv_init55 = _weight_variable([5, 5, self.img_channels, self.num_filter], name="W_conv_init55")
        W_conv_init11 = _weight_variable([1, 1, self.img_channels, self.num_filter], name="W_conv_init11")
        h_conv_init = tf.nn.relu(_conv2d(self.x, W_conv_init55) + _conv2d(self.x, W_conv_init11), name="h_conv_init")
        
        # followed by a series of resnet 3x3 conv layers
        W_conv_intermediate = []
        h_conv_intermediate = []
        _current_h_conv = h_conv_init
        for i in range(self.num_conv_layers):
            #with tf.name_scope("layer"+str(i)):
            with tf.variable_scope("layer"+str(i)):
                _resnet_weights1 = _weight_variable([3, 3, self.num_filter, self.num_filter], name="W_conv_resnet1")
                _resnet_weights2 = _weight_variable([3, 3, self.num_filter, self.num_filter], name="W_conv_resnet2")
                if self.use_batch_norm:
                    _int_conv = tf.nn.relu(
                        self.batch_norm("bn_conv_intermediate",_conv2d(_current_h_conv, _resnet_weights1)),
                        name="h_conv_intermediate")
                    _output_conv = tf.nn.relu(
                        self.batch_norm("bn_conv",_current_h_conv + _conv2d(_int_conv, _resnet_weights2)),
                        name="h_conv")
                else:
                    _int_conv = tf.nn.relu(
                        _conv2d(_current_h_conv, _resnet_weights1),
                        name="h_conv_intermediate")
                    _output_conv = tf.nn.relu(
                        _current_h_conv + _conv2d(_int_conv, _resnet_weights2),
                        name="h_conv")
                W_conv_intermediate.extend([_resnet_weights1, _resnet_weights2])
                h_conv_intermediate.append(_output_conv)
                _current_h_conv = _output_conv

        """
        W_conv_final = _weight_variable([1, 1, self.num_filter, 1], name="W_conv_final")
        b_conv_final = tf.Variable(tf.constant(0, shape=[self.num_actions], dtype=tf.float32), name="b_conv_final")
        h_conv_final = _conv2d(h_conv_intermediate[-1], W_conv_final)
        
        self.softmax_p = tf.nn.softmax(tf.reshape(h_conv_final, [-1, self.num_actions]) + b_conv_final)
        self.logits_p = tf.reshape(h_conv_final, [-1, self.num_actions]) + b_conv_final
        """
        # for both spin
        # final use two filter. one for spin 0, the other for spin 1
        W_conv_final = _weight_variable([1, 1, self.num_filter, 2], name="W_conv_final")
        b_conv_final = tf.Variable(tf.constant(0, shape=[self.num_actions], dtype=tf.float32), name="b_conv_final")
        h_conv_final = _conv2d(h_conv_intermediate[-1], W_conv_final)
        
        self.softmax_p = tf.nn.softmax(tf.reshape(h_conv_final, [-1, self.num_actions]) + b_conv_final)
        self.logits_p = tf.reshape(h_conv_final, [-1, self.num_actions]) + b_conv_final
    
        
    def _create_train_op(self):
        self.opt = tf.train.RMSPropOptimizer(
            learning_rate=self.var_learning_rate,
            decay=Config.RMSPROP_DECAY,
            momentum=Config.RMSPROP_MOMENTUM,
            epsilon=Config.RMSPROP_EPSILON)
        
        
        self.log_likelihood_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_p, labels=self.action_index))
        train_op = self.opt.minimize(self.log_likelihood_cost, global_step=self.global_step)
        train_ops = [train_op] + self.extra_train_ops
        self.train_op = tf.group(*train_ops)

        # for the case of reinforcement learning
        self.selected_action_prob = tf.reduce_sum(self.softmax_p * self.action_index, axis=1)
        self.rl_cost = - tf.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) * self.y_r    
        self.rl_cost = tf.reduce_sum(self.rl_cost, axis=0)

        self.opt_grad = self.opt.compute_gradients(self.rl_cost)
        self.opt_grad_clipped = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM),v) for g,v in self.opt_grad]
        train_rl_op = self.opt.apply_gradients(self.opt_grad_clipped)

        #train_rl_op = self.opt.minimize(self.rl_cost, global_step=self.global_step)
        train_rl_ops = [train_rl_op] + self.extra_train_ops
        self.train_rl_op = tf.group(*train_rl_ops)
    
   

    def _create_tensor_board(self):
        """
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("GA3C_curling/Pcost_advantage", self.cost_p_1_agg))
        summaries.append(tf.summary.scalar("GA3C_curling/Pcost_entropy", self.cost_p_2_agg))
        summaries.append(tf.summary.scalar("GA3C_curling/Pcost", self.cost_p))
        summaries.append(tf.summary.scalar("GA3C_curling/Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("GA3C_curling/LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("GA3C_curling/Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("GA3C_curling/weights_%s" % var.name.replace(':','_'), var))

        #summaries.append(tf.summary.histogram("GA3C_curling/activation_n1", self.n1))
        #summaries.append(tf.summary.histogram("GA3C_curling/activation_n2", self.n2))
        #summaries.append(tf.summary.histogram("GA3C_curling/activation_d2", self.d1))
        #for weight_var in self.conv_layers:
        #    summaries.append(tf.summary.histogram("GA3C_curling/activation_%s" % weight_var.name.replace(':','_'), weight_var))      
        summaries.append(tf.summary.histogram("GA3C_curling/activation_v", self.logits_v))
        summaries.append(tf.summary.histogram("GA3C_curling/activation_p", self.softmax_p))

        self.summary_op = tf.summary.merge(summaries)
        """
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_dir, self.sess.graph)
        
    
    
    def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            #b_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.zeros_initializer()
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    """
    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            #b_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.zeros_initializer()
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)
            #output = tf.nn.conv2d(input, w, strides=strides, padding='SAME')
            output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output
    """

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict_single(self, x):
        return self.predict_p(x[None, :])[0]

    def predict_v(self, x):
        prediction = self.sess.run(self.logits_v, feed_dict={self.x: x})
        return prediction

    def predict_p(self, x):
        prediction = self.sess.run(self.softmax_p, feed_dict={self.x: x})
        return prediction
    
    def predict_p_and_v(self, x):
        return self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.x: x})
    
    def train(self, x, a):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.action_index: a})
        self.sess.run(self.train_op, feed_dict=feed_dict)
        
    # reinforcement learning
    def train_rl(self, x, y_r, a):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        self.sess.run(self.train_rl_op, feed_dict=feed_dict)
        
        
    def get_log_likelihood_cost(self, x, a):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.action_index: a})
        cost = self.sess.run(self.log_likelihood_cost, feed_dict=feed_dict)
        return cost
    
    def debug(self, x):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x})
        return self.sess.run(self.logits_p, feed_dict=feed_dict)

    def log(self, x, y_r, a):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    #def _checkpoint_filename(self, episode):
    #    return 'checkpoints/%s_%08d' % (self.model_name, episode)
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(filename.split('/')[-1].split('-')[-1])
        #return int(re.split('/|_|\.', filename)[2])

    def get_model_dir(self, attr):
        model_dir = ''
        for attr in attr:
            if hasattr(self, attr):
                if attr == "model_name":
                    model_dir += "/%s" % (getattr(self, attr))
                else:
                    model_dir += "/%s=%s" % (attr, getattr(self, attr))
        return model_dir
    
    def save(self, episode):
        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__
        
        #save_model_dir = '%s/%s' % ('checkpoints',self.model_dir)
        save_model_dir = 'checkpoints' + self.model_dir
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        self.saver.save(self.sess, os.path.join(save_model_dir, model_name), global_step = episode)
        #self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self, load_model_dir = None):
        print(" [*] Loading checkpoints...")
        if not(load_model_dir):
            #load_model_dir = '%s/%s' % ('checkpoints',self.model_dir)
            load_model_dir = 'checkpoints' + self.model_dir
        ckpt = tf.train.get_checkpoint_state(load_model_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(load_model_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return self._get_episode_from_filename(fname)
        else:
            print(" [!] Load FAILED: %s" % load_model_dir)
            return False

      
        
        #filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        #if Config.LOAD_EPISODE > 0:
        #    filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        #self.saver.restore(self.sess, filename)
        #return self._get_episode_from_filename(filename)
       
    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))
