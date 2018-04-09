from time import time
import tensorflow as tf
from Atari2.QNetwork1step import QNetwork1Step
from Atari2.Helper import update_target_graph
import numpy as np
import matplotlib.pyplot as mpl


# Worker class
class Worker:
    def __init__(self, game, name, s_size, a_size, trainer, model_path, global_step):
        self.name = "worker_" + str(name)
        self.is_chief = self.name == 'worker_0'
        print(self.name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_step = global_step

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        with tf.variable_scope(self.name):
            self.increment_global_step = self.global_step.assign_add(1)
            self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = QNetwork1Step(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        self.batch_rnn_state = None

        # Env set-up
        self.a_size = a_size
        self.env = game
        self.s_size = s_size

        self.batch_size = 30
        self.target_update_steps = 5000
        self.summary_save_episodes = 5
        self.model_save_episodes = 250

    def train(self, rollout, sess, gamma, ac_network):
        rollout = np.array(rollout)
        observations = np.reshape(np.vstack(rollout[:, 0]), newshape=[-1,]+self.s_size)
        actions = rollout[:, 1]  # action taken at timestep t
        rewards = rollout[:, 2]  # reward t
        # next_observations = np.vstack(rollout[:, 3])   # state t+1
        terminals = rollout[:, 4]  # whether timestep t was terminal
        next_max_q = rollout[:, 5]  # the best Q value of state t+1 as calculated by target network

        # we get rewards, terminals, prev_screen, next screen, and target network
        n_step, q_batch = True, False
        discounted_rewards = (1. - terminals) * gamma * next_max_q + rewards
        if n_step:
            for i in reversed(range(len(discounted_rewards) - 1)):
                discounted_rewards[i] = gamma * discounted_rewards[i + 1] + rewards[i]
        elif q_batch:
            for i in reversed(range(len(discounted_rewards) - 1)):
                discounted_rewards[i] = gamma * max(discounted_rewards[i + 1], next_max_q[i]) + rewards[i]

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {ac_network.target_q_t: discounted_rewards,
                     ac_network.network.inputs: observations,
                     ac_network.actions: actions,
                     #ac_network.network.state_in[0]: self.batch_rnn_state[0],
                     #ac_network.network.state_in[1]: self.batch_rnn_state[1]
                     }
        v_l, q_value, q_acted, g_n, v_n, self.batch_rnn_state, _ = sess.run(
            [ac_network.loss, ac_network.network.value, ac_network.q_acted,
             ac_network.grad_norms, ac_network.var_norms, ac_network.network.state_out,
             ac_network.apply_grads],
            feed_dict=feed_dict)

        return v_l / len(rollout), 0, 0, g_n, v_n

    # Take an action using probabilities from policy network output.
    def take_action_from_network(self, sess, network, previous_screen, rnn_state):
        action, value, rnn_state = sess.run([network.best_action_index, network.value, network.state_out],
                                            feed_dict={network.inputs: [previous_screen],
                                                       #network.state_in[0]: rnn_state[0],
                                                       #network.state_in[1]: rnn_state[1]
                                                       })
        return action[0], np.max(value), rnn_state

    def work(self, max_steps, exploration_anealing_steps, max_episode_length, gamma, sess, coord=None, saver=None):
        global_step_count = sess.run(self.global_step)
        local_step_count = 0
        episode_count = 0
        print("Starting worker " + str(self.number))

        prev_clock = time()
        if coord is None:
            coord = sess

        sess.run(self.local_AC.assign_op)
        print(self.number, "Resetting target network")

        while not coord.should_stop() and global_step_count < max_steps:
            # copy global to local networks
            sess.run(self.update_local_ops)

            episode_buffer = []
            episode_values = []
            episode_reward = 0
            episode_step_count = 0
            action_indexes = list(range(self.a_size))

            previous_screen = self.env.reset()

            end_epsilon_probs = [0.01, 0.1, 0.5]
            rnn_state = self.local_AC.network.state_init
            self.batch_rnn_state = rnn_state
            while episode_step_count < max_episode_length:
                self.env.render()
                exploration_rate = max(1 - (1 - end_epsilon_probs[self.number % len(end_epsilon_probs)]) *
                                       (global_step_count / exploration_anealing_steps),
                                       end_epsilon_probs[self.number % len(end_epsilon_probs)])

                # explore / exploit
                if np.random.random() < exploration_rate:
                    action, value, rnn_state = \
                        self.take_action_from_network(sess, self.local_AC.network, previous_screen, rnn_state)
                    episode_values.append(value)
                    # override action
                    action = np.random.choice(action_indexes)
                else:
                    action, value, rnn_state = \
                        self.take_action_from_network(sess, self.local_AC.network, previous_screen, rnn_state)
                    episode_values.append(value)

                # Watch environment
                current_screen, reward, terminal, _ = self.env.step(action)
                current_screen = list(current_screen)
                episode_reward += reward

                # get target network values
                next_max_q = sess.run(self.local_AC.target_network.best_q,
                                      feed_dict={self.local_AC.target_network.inputs: [current_screen],
                                                 #self.local_AC.target_network.state_in[0]: rnn_state[0],
                                                 #self.local_AC.target_network.state_in[1]: rnn_state[1]
                                                 })

                # Store environment
                episode_buffer.append([previous_screen, action, reward, current_screen, terminal, next_max_q])
                previous_screen = current_screen

                # If the episode hasn't ended, but the experience buffer is full, then we make an update step using
                # that experience rollout.
                if not terminal and len(episode_buffer) == self.batch_size and \
                                episode_step_count < max_episode_length - 1:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, self.local_AC)

                    episode_buffer = []
                    sess.run(self.update_local_ops)

                # Measure time and increase episode step count
                local_step_count += 1
                global_step_count = sess.run(self.increment_global_step)
                episode_step_count += 1
                if local_step_count % 2000 == 0:
                    new_clock = time()
                    print(2000.0 / (new_clock - prev_clock), "it/s,   ")
                    prev_clock = new_clock

                # Copy local online -> global target networks
                if global_step_count % self.target_update_steps == 0:
                    sess.run(self.local_AC.assign_op)

                # If terminal, then break from episode
                if terminal:
                    break

            # print("0ver ",episode_count)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_step_count)
            self.episode_mean_values.append(np.mean(episode_values))

            # Update the network using the experience buffer at the end of the episode.
            v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, self.local_AC)

            mean_length = np.mean(self.episode_lengths[-5:])
            mean_reward = np.mean(self.episode_rewards[-5:])
            mean_value = np.mean(self.episode_mean_values[-5:])

            # Periodically save model parameters, and summary statistics.
            if episode_count % self.summary_save_episodes == 0 and episode_count != 0:
                # Save current model
                if self.is_chief and saver is not None and episode_count % self.model_save_episodes == 0:
                    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                    print("Saved Model")

                # Save statistics for TensorBoard
                summary = tf.Summary()
                summary.value.add(tag='Perf/Length', simple_value=float(mean_length))  # avg episode length
                summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))  # avg reward
                summary.value.add(tag='Perf/Value', simple_value=float(mean_value))  # avg episode value_predator
                summary.value.add(tag='Losses/Value Loss', simple_value=float(np.mean(v_l)))  # value_loss
                summary.value.add(tag='Losses/Policy Loss', simple_value=float(np.mean(p_l)))  # policy_loss
                summary.value.add(tag='Losses/Entropy', simple_value=float(np.mean(e_l)))  # entropy
                summary.value.add(tag='Losses/Grad Norm', simple_value=float(np.mean(g_n)))  # grad_norms
                summary.value.add(tag='Losses/Var Norm', simple_value=float(np.mean(v_n)))  # var_norms

                self.summary_writer.add_summary(summary, global_step_count)
                self.summary_writer.flush()

            if self.is_chief:
                print("Global step @", global_step_count, " epsilon =", exploration_rate, " mean reward =", mean_reward)
            # Update episode count
            episode_count += 1
