from time import time
import tensorflow as tf
from Maze2.ActorCriticNetwork import AC_Network
from Maze2.Helper import update_target_graph, discount, get_empty_loss_arrays
import numpy as np
from time import sleep
import math


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
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        # Env Pursuit set-up
        self.env = game
        self.s_size = s_size

        self.batch_size = 30
        self.summary_save_episodes = 5
        self.model_save_episodes = 250

    def train(self, rollout, sess, gamma, bootstrap_value, ac_network):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        #next_observations = rollout[:, 3]
        #terminals = rollout[:, 4]  # whether timestep t was terminal
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(rewards_plus, gamma)[:-1]
        value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {ac_network.target_v: discounted_rewards,
                     ac_network.inputs: np.vstack(observations),
                     ac_network.actions: actions,
                     ac_network.advantages: advantages}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([ac_network.value_loss,
                                               ac_network.policy_loss,
                                               ac_network.entropy,
                                               ac_network.grad_norms,
                                               ac_network.var_norms,
                                               ac_network.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    # Take an action using probabilities from policy network output.
    def take_action_from_network(self, sess, network, previous_screen, action_indexes):
        action_distribution, value = sess.run([network.policy, network.value],
                                              feed_dict={network.inputs: [previous_screen]})

        action = np.random.choice(action_indexes, p=action_distribution[0])
        return action, value

    def work(self, max_steps, max_episode_length, gamma, sess, coord=None, saver=None):
        global_step_count = sess.run(self.global_step)
        local_step_count = 0
        episode_count = 0
        print("Starting worker " + str(self.number))

        prev_clock = time()
        if coord is None:
            coord = sess

        while not coord.should_stop() and global_step_count < max_steps:
            # copy global to local networks
            sess.run(self.update_local_ops)

            episode_buffer = []
            episode_values = []
            episode_reward = 0
            episode_step_count = 0
            action_indexes = list(range(4))

            previous_screen = self.env.reset()

            while episode_step_count < max_episode_length:
                # explore / exploit
                action, value = \
                    self.take_action_from_network(sess, self.local_AC, previous_screen, action_indexes)

                # Watch environment
                current_screen, reward, terminal, _ = self.env.step(action)
                episode_reward += reward

                # Store environment
                episode_buffer.append([previous_screen, action, reward, current_screen, terminal, value[0]])
                episode_values.append(value[0])
                previous_screen = current_screen

                # If the episode hasn't ended, but the experience buffer is full,
                # then we make an update step using that experience rollout.
                if not terminal and len(episode_buffer[0]) == self.batch_size and \
                                episode_step_count < max_episode_length - 1:
                    # Since we don't know what the true final return is,
                    # we "bootstrap" from our current value_predator estimation.
                    v1 = sess.run(self.local_AC.value, feed_dict={self.local_AC.inputs: previous_screen})
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1[0], self.local_AC)

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

                # If game over, then break from episode
                if terminal:
                    break

            # print("0ver ",episode_count)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_step_count)
            self.episode_mean_values.append(np.mean(episode_values))

            # Update the network using the experience buffer at the end of the episode.
            v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0, self.local_AC)

            # Save statistics for TensorBoard
            mean_length = np.mean(self.episode_lengths[-5:])
            mean_reward = np.mean(self.episode_rewards[-5:])
            mean_value = np.mean(self.episode_mean_values[-5:])

            # Periodically save model parameters, and summary statistics.
            if episode_count % self.summary_save_episodes == 0 and episode_count != 0:
                # Save current model
                if self.is_chief and saver is not None and episode_count % self.model_save_episodes == 0:
                    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                    print("Saved Model")

                summary = tf.Summary()

                summary.value.add(tag='Perf/Length', simple_value=float(mean_length))               # avg episode length
                summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))               # avg reward
                summary.value.add(tag='Perf/Value', simple_value=float(mean_value))       # avg episode value_predator
                summary.value.add(tag='Losses/Value Loss', simple_value=float(np.mean(v_l)))       # value_loss
                summary.value.add(tag='Losses/Policy Loss', simple_value=float(np.mean(p_l)))      # policy_loss
                summary.value.add(tag='Losses/Entropy', simple_value=float(np.mean(e_l)))          # entropy
                summary.value.add(tag='Losses/Grad Norm', simple_value=float(np.mean(g_n)))        # grad_norms
                summary.value.add(tag='Losses/Var Norm', simple_value=float(np.mean(v_n)))         # var_norms

                self.summary_writer.add_summary(summary, global_step_count)
                self.summary_writer.flush()

            if self.is_chief:
                print("Global step @", global_step_count, " mean reward =", mean_reward)
            # Update episode count
            episode_count += 1
