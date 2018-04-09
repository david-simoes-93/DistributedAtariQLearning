from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from Atari3.qmodel import QNetwork
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def process_rollout(rollout, gamma, lambda_=1.0):
    """
given a rollout, compute its returns and the advantage
"""
    batch_si = np.asarray(rollout.states)  # observations
    batch_a = np.asarray(rollout.actions)  # actions
    rewards = np.asarray(rollout.rewards)  # rewards
    vpred_t = np.asarray(rollout.values + [rollout.r])  # next_max_q

    # discounted rewards
    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    nstep = True
    if nstep:
        batch_r = discount(rewards_plus_v, gamma)[:-1]
    else:
        batch_r = np.zeros(len(rewards_plus_v))
        batch_r[-1] = rollout.r
        for i in reversed(range(len(rewards_plus_v) - 1)):
            batch_r[i] = gamma * batch_r[i + 1] + max(rewards_plus_v[i], vpred_t[i + 1])
        batch_r = batch_r[:-1]
        # print(batch_r)

    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_r, rollout.terminal, features)


Batch = namedtuple("Batch", ["si", "a", "r", "terminal", "features"])


class PartialRollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)


class RunnerThread(threading.Thread):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""

    def __init__(self, env, policy, num_local_steps, visualise, min_lambda, max_expl_steps):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise
        self.min_exploration_rate = min_lambda
        self.max_exploration_steps = max_expl_steps

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer, self.visualise,
                                      self.min_exploration_rate, self.max_exploration_steps)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)


def env_runner(env, policy, num_local_steps, summary_writer, render, min_exploration_rate, max_exploration_steps):
    """
The logic of the thread runner.  In brief, it constantly keeps on running
the policy, and as long as the rollout exceeds a certain length, the thread
runner appends the policy to the queue.
"""
    last_state = env.reset()
    last_features = policy.get_initial_features()
    length = 0
    rewards = 0
    global_step = policy.global_step.eval()
    current_exploration_rate = max(min_exploration_rate, 1 - global_step/max_exploration_steps)
    possible_actions = list(range(env.action_space.n))

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            fetched = policy.act(last_state, *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]

            if np.random.random() > current_exploration_rate:
                action = action[0]
            else:
                action = np.zeros(env.action_space.n)
                action[np.random.choice(possible_actions)] = 1

            # argmax to convert from one-hot
            state, reward, terminal, info = env.step(action.argmax())
            if render:
                env.render()

            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal, last_features)
            length += 1
            rewards += reward

            last_state = state
            last_features = features

            global_step = policy.global_step.eval()
            current_exploration_rate = max(min_exploration_rate, 1 - global_step / max_exploration_steps)

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, global_step)
                summary_writer.flush()

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or length >= timestep_limit:
                terminal_end = True
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                last_features = policy.get_initial_features()
                print("Episode finished, exploration %4.2f. Sum of rewards: %d. Length: %d" % (current_exploration_rate, rewards, length))
                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout


class DQN(object):
    def __init__(self, env, task, visualise, min_exploration_rate, max_exploration_steps):
        """
An implementation of the DQN algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""

        self.env = env
        self.task = task
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = QNetwork(env.observation_space.shape, env.action_space.n)
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
            with tf.variable_scope("global_target"):
                self.target_network = QNetwork(env.observation_space.shape, env.action_space.n)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = QNetwork(env.observation_space.shape, env.action_space.n)
                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")  # 1hot action tensor
            # self.r = tf.placeholder(tf.float32, [None], name="r")

            # losses!
            self.target_q = tf.placeholder('float32', [None], name='target_q')
            self.q_acted = tf.reduce_sum(pi.q_vals * self.ac, reduction_indices=1, name='q_acted')
            bs = tf.to_float(tf.shape(pi.x)[0])
            self.loss = tf.reduce_mean(tf.square(self.target_q - self.q_acted), name='loss')

            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(env, pi, 20, visualise, min_exploration_rate, max_exploration_steps)

            grads = tf.gradients(self.loss, pi.var_list)

            tf.summary.scalar("model/loss", self.loss / bs)
            tf.summary.image("model/state", pi.x)
            tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
            tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
            self.summary_op = tf.summary.merge_all()

            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(1e-4)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0

            # copy global online network to global target network
            self.sync_target = tf.group(*[v1.assign(v2) for v1, v2 in
                                          zip(self.network.var_list, self.target_network.var_list)])

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
self explanatory:  take a rollout from the queue of the thread runner.
"""
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
process grabs a rollout that's been produced by the thread runner,
and updates the parameters.  The update is then sent to the parameter
server.
"""

        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        if self.local_network.lstm:
            feed_dict = {
                self.local_network.x: batch.si,
                self.ac: batch.a,
                self.target_q: batch.r,
                self.local_network.state_in[0]: batch.features[0],
                self.local_network.state_in[1]: batch.features[1],
            }
        else:
            feed_dict = {
                self.local_network.x: batch.si,
                self.ac: batch.a,
                self.target_q: batch.r
            }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1
