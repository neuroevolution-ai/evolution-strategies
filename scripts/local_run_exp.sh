#!/bin/sh
NAME=exp_`date "+%m_%d_%H_%M_%S"`
EXP_FILE=$1
tmux new -s $NAME -d

# Start the Master, listening on socket /tmp/redis.sock
tmux send-keys -t $NAME 'python3 -m es_distributed.main master --master_socket_path=/tmp/redis.sock --exp_file '"$EXP_FILE" C-m

tmux split-window -t $NAME

# Start the worker
tmux send-keys -t $NAME 'python3 -m es_distributed.main workers --master_host localhost --relay_socket_path /tmp/redis.sock --num_workers 1' C-m
tmux a -t $NAME
