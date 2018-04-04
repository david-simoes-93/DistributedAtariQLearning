# While training is taking place, statistics on agent performance are available from Tensorboard. To launch it use:
# 
#   tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'
#   tensorboard --logdir=worker_0:'./train_0'
#   tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3',worker_4:'./train_4',worker_5:'./train_5',worker_6:'./train_6',worker_7:'./train_7',worker_8:'./train_8',worker_9:'./train_9',worker_10:'./train_10',worker_11:'./train_11'


import argparse
import os
import tensorflow as tf
from Maze2.QNetwork1step import QNetwork1Step
from Maze2.DQNSlave import Worker
import matplotlib.pyplot as mpl
from Maze2.Bridge import GymBridge

max_episode_length = 1000
bridge_size = 40

gamma = 1  # discount rate for advantage estimation and reward discounting
a_size = 4  # Agent can move Left, Right, up down
learning_rate = 1e-4  #
load_model = False
model_path = './model_dist'

max_steps = 500000
exploration_anealing_steps = max_steps * 0.75

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
    "--task_index",
    type=int,
    default=0,
    help="Index of task within the job"
)
parser.add_argument(
    "--slaves_per_url",
    type=str,
    default="1",
    help="Comma-separated list of maximum tasks within the job"
)
parser.add_argument(
    "--urls",
    type=str,
    default="localhost",
    help="Comma-separated list of hostnames"
)

FLAGS, unparsed = parser.parse_known_args()

# Create a cluster from the parameter server and worker hosts.
hosts = []
for (url, max_per_url) in zip(FLAGS.urls.split(","), FLAGS.slaves_per_url.split(",")):
    for i in range(int(max_per_url)):
        hosts.append(url + ":" + str(2410 + i))
print(hosts)
cluster = tf.train.ClusterSpec({"dqn": hosts})
server = tf.train.Server(cluster, job_name="dqn", task_index=FLAGS.task_index)

tf.reset_default_graph()

# Create a directory to save models and episode playback gifs
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device(tf.train.replica_device_setter(worker_device="/job:dqn/task:%d" % FLAGS.task_index, cluster=cluster)):
    global_episodes = tf.contrib.framework.get_or_create_global_step()
    trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    master_network = QNetwork1Step(bridge_size, a_size, 'global', None)  # Generate global network

    # Master declares worker for all slaves
    for i in range(len(hosts)):
        print("Initializing variables for slave ", i)
        if i == FLAGS.task_index:
            worker = Worker(GymBridge(size=bridge_size), i, bridge_size, a_size, trainer, model_path, global_episodes)
        else:
            Worker(GymBridge(size=bridge_size), i, bridge_size, a_size, trainer, model_path, global_episodes)

print("Starting session", server.target, FLAGS.task_index)
hooks = [tf.train.StopAtStepHook(last_step=max_steps)]
with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0),
                                       config=tf.ConfigProto(),
                                       save_summaries_steps=None, save_summaries_secs=None,
                                       save_checkpoint_secs=600, checkpoint_dir=model_path, hooks=hooks) as mon_sess:
    print("Started session")
    try:
        worker.work(max_steps, exploration_anealing_steps, max_episode_length, gamma, mon_sess)
    except RuntimeError:
        print("Puff")
    if FLAGS.task_index == 0:
        mpl.savefig("plot.pdf")

print("Done")
