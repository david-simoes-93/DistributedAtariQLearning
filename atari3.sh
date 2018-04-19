workon tf
source ~/tensorflow/bin/activate
export PYTHONPATH=$(pwd)
python Atari3/train.py --num-workers 12 --env-id PongDeterministic-v3 --log-dir ./logs
