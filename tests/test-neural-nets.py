from es_distributed.policies import MujocoPolicy
from es_distributed.es import RunningStat
import gym, roboschool

import tensorflow as tf
import numpy as np


def main():
    sess = tf.InteractiveSession()

    env = gym.make("RoboschoolAnt-v1")

    ac_space = env.action_space
    ob_space = env.observation_space

    policy = MujocoPolicy(ob_space, ac_space, "uniform:10", 0.01, 'tanh', [256, 256], 'ff')

    ob_stat = RunningStat(
        env.observation_space.shape,
        eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
    )

    policy.set_ob_stat(ob_stat.mean, ob_stat.std)
    ob = np.ones(ob_space.shape)#env.reset()
    sess.run(tf.initialize_variables(tf.all_variables()))

    a = policy.act(ob[None])
    print(a)

if __name__ == '__main__':
    main()