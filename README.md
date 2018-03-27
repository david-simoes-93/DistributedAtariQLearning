# Distributed Asynchronous n-step QLearning for Atari Environment with TensorFlow

Implementation of Asynchronous n-step Q-Learning from [Mnih et al. 2016](http://proceedings.mlr.press/v48/mniha16.pdf), using TensorFlow, Keras, and a distributed implementation (unlike most implementations out there, which use threads, this uses separate processes. It can be run in different machines, and runs concurrently, as opposed to Python threads).

Requirements: install [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/), along with Gym, the Atari environments, and Scikit's Image module.

    pip install gym gym[atari] scikit-image
    
For a single worker, with GUI, just

    python3 async_dqn.py --experiment breakout --game "Breakout-v0" --show_training True
    
To run 4 workers on your CPU, just

    ./start-dqn.sh 4
    
You can run TensorBoard on statistics from each worker with

    tensorboard --logdir=worker_0:'./summaries0',worker_1:'./summaries1',worker_2:'./summaries2',worker_3:'./summaries3',worker_4:'./summaries4'
    
Original repository [here](https://github.com/coreylynch/async-rl).
