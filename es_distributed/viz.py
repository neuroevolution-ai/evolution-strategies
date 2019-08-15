import click

import os

import gym
import roboschool
from gym import wrappers
import tensorflow as tf
from policies import MujocoPolicy
import numpy as np

from multiprocessing import Process

def run_policy(env_id, policy_file, record, stochastic, extra_kwargs):
    env = gym.make(env_id)
    if record:
        import uuid
        env = wrappers.Monitor(env, '/tmp/' + str(uuid.uuid4()), force=True)

    if extra_kwargs:
        import json
        extra_kwargs = json.loads(extra_kwargs)

    with tf.Session():
        pi = MujocoPolicy.Load(policy_file, extra_kwargs=extra_kwargs)
        # while True:
        rews, t = pi.rollout(env, render=True,
                             random_stream=None)  # random_stream=np.random if stochastic else None)
        print('return={:.4f} len={}'.format(rews.sum(), t))

        if record:
            env.close()
            return
    tf.reset_default_graph()
    env.close()

@click.command()
@click.argument('env_id')
@click.argument('policies_path')
@click.option('--record', is_flag=True)
@click.option('--stochastic', is_flag=True)
@click.option('--extra_kwargs')
def main(env_id, policies_path, record, stochastic, extra_kwargs):
    os.chdir(policies_path)
    policy_files = [file for file in os.listdir(policies_path) if file.endswith(".h5")]
    policy_files.sort()

    # Skip 20 files
    policy_files = policy_files[::20]

    # Use a new process for every policy because there is a bug in the roboschool where on a second run the environment
    # does not get properly reset
    for policy in policy_files:
        p = Process(target=run_policy, args=(env_id, policy, record, stochastic, extra_kwargs))
        p.start()
        p.join()

if __name__ == '__main__':
    main()