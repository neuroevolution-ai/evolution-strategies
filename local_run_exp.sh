#!/bin/sh
NAME=exp_`date "+%m_%d_%H_%M_%S"`
EXP_FILE=$1
tmux new -s $NAME -d
#tmux send-keys -t $NAME '. scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python3 -m es_distributed.main master --master_socket_path /tmp/es_redis_master.sock --exp_file '"$EXP_FILE" C-m
tmux split-window -t $NAME
#tmux send-keys -t $NAME '. scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python3 -m es_distributed.main workers --master_host localhost --relay_socket_path /tmp/es_redis_master.sock' C-m
tmux a -t $NAME