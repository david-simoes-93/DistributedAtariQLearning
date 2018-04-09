import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from Atari2.Helper import normalized_columns_initializer


# 1-step Q-network
class QNetwork1Step:
    def make_network(self, s_size, a_size, trainable=True):
        network = type('', (), {})()

        # Input and visual encoding layers
        network.inputs = tf.placeholder(shape=[None,]+s_size, dtype=tf.float32)

        network.imageIn = network.inputs #tf.reshape(network.inputs, shape=[-1,]+s_size) # probably unnecessary
        #print(tf.shape(network.imageIn))

        """conv1 = tf.layers.conv2d(
            inputs=network.imageIn,
            filters=16,
            kernel_size=[8, 8],
            padding="same",
            strides=[4, 4],
            activation=tf.nn.elu,
            data_format='channels_first')"""

        network.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=network.imageIn, num_outputs=16,
                                 kernel_size=[8, 8], stride=[4, 4], padding='SAME', data_format='NHWC')
        network.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=network.conv1, num_outputs=32,
                                 kernel_size=[4, 4], stride=[2, 2], padding='SAME', data_format='NHWC')
        hidden = slim.fully_connected(slim.flatten(network.conv2), 256, activation_fn=tf.nn.elu)

        if False:
            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

            # initial states
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            network.state_init = [c_init, h_init]

            # lstm input
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            network.state_in = (c_in, h_in)

            # lstm shape and output
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(network.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            network.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])
        else:
            network.state_init = [0, 0]
            #network.state_in = (tf.placeholder(tf.float32, [1]), tf.placeholder(tf.float32, [1]))
            network.state_out = hidden  # any tensor
            rnn_out = slim.fully_connected(hidden, 256, activation_fn=tf.nn.elu)

        network.value = slim.fully_connected(rnn_out, a_size, activation_fn=None,
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
