max="$1"
count=`expr $max - 1`

export PYTHONPATH=$(pwd)

echo 0 to "$count"
sleep 5

for i in `seq 0 $count`
do
    python async_dqn.py --experiment breakout --game "Pong-v0" --slaves_per_url "$max" --thread_index="$i" &
    echo python async_dqn.py --experiment breakout --game "Breakout-v0" --slaves_per_url "$max" --thread_index="$i"
    sleep 5
done
