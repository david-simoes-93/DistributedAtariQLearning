import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from Maze2.Helper import normalized_columns_initializer


# 1-step Q-network
class QNetwork1Step:
    def make_network(self, s_size, a_size, trainable=True):
        network = type('', (), {})()

        network.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)

        # hidden = slim.fully_connected(network.inputs, 150,
        #                              weights_initializer=tf.contrib.layers.xavier_initializer(),
        #                              activation_fn=tf.nn.elu, trainable=trainable)
        # hidden2 = slim.fully_connected(hidden, 150, weights_initializer=tf.contrib.layers.xavier_initializer(),
        #                               activation_fn=tf.nn.elu, trainable=trainable)

        hidden2 = slim.fully_connected(network.inputs, 40, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                       activation_fn=tf.nn.relu, trainable=trainable)

        #hidden2 = network.inputs

        network.state_init = None
        network.value = slim.fully_connected(hidden2, a_size, activation_fn=None,
                                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                                             biases_initializer=None, trainable=trainable)

        network.best_q = tf.reduce_max(network.value, 1)
        network.best_action_index = tf.arg_max(network.value, 1)
        network.policy = tf.one_hot(network.best_action_index, a_size)

        return network

    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            print("Scope", scope)
            with tf.variable_scope("regular"):
                self.network = self.make_network(s_size, a_size)
            with tf.variable_scope("target"):
                self.target_network = self.make_network(s_size, a_size)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)  # Index of actions taken
                self.actions_onehot = tf.one_hot(self.actions, a_size,
                                                 dtype=tf.float32)  # 1-hot tensor of actions taken

                # losses!
                self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
                self.q_acted = tf.reduce_sum(self.network.value * self.actions_onehot, reduction_indices=1,
                                             name='q_acted')
                self.loss = tf.reduce_mean(tf.square(self.target_q_t - self.q_acted), name='loss')

                # Get gradients from local network using local losses
                self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, self.local_vars)
                self.var_norms = tf.global_norm(self.local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global/regular')
                target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "global/target")

                # set global target network to be the same as local network
                weights = slim.get_trainable_variables(scope=scope + "/regular")
                self.assign_op = {}
                for w, t_w in zip(weights, target_weights):
                    self.assign_op[w.name] = t_w.assign(w)

                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
