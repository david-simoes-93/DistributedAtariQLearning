import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from Maze2.Helper import normalized_columns_initializer


# Actor-Critic Network
class AC_Network:
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            print("Scope", scope)

            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            #hidden = slim.fully_connected(self.inputs, 150,
            #                              weights_initializer=tf.contrib.layers.xavier_initializer(),
            #                              activation_fn=tf.nn.elu)
            #hidden2 = slim.fully_connected(hidden, 150, weights_initializer=tf.contrib.layers.xavier_initializer(),
            #                               activation_fn=tf.nn.elu)

            hidden2 = slim.fully_connected(self.inputs, 40,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          activation_fn=tf.nn.relu)

            self.policy = slim.fully_connected(hidden2, a_size, activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(hidden2, 1, activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)                 # Index of actions taken
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)    # 1-hot tensor of actions taken
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)              # Target Value
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)            # temporary difference (R-V)

                self.log_policy = tf.log(tf.clip_by_value(self.policy, 1e-20, 1.0))         # avoid NaN with clipping when value in policy becomes zero
                self.responsible_outputs = tf.reduce_sum(self.log_policy * self.actions_onehot, [1]) # Get policy*actions influence
                self.r_minus_v = self.target_v - tf.reshape(self.value, [-1])               # difference between target value and actual value

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.r_minus_v))            # same as tf.nn.l2_loss(r_minus_v)
                self.entropy = - tf.reduce_sum(self.policy * self.log_policy)               # policy entropy
                self.policy_loss = -tf.reduce_sum(self.responsible_outputs * self.advantages)   # policy loss

                # Learning rate for Critic is half of Actor's, so value_loss/2 + policy loss
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, self.local_vars)
                self.var_norms = tf.global_norm(self.local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
