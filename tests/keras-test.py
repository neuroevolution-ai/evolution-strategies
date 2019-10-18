from es_distributed.policies import MujocoPolicy
from es_distributed.es import RunningStat
import gym, roboschool

import tensorflow as tf
import numpy as np

def jupyter_cell():
    ob_stat = RunningStat(
        env.observation_space.shape,
        eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
    )

    rews, lens = [], []

    for _ in range(1000):
        model = create_model(ob_mean=ob_stat.mean, ob_std=ob_stat.std)
        r, t = rollout(env, model)
        rews.append(r.sum())
        lens.append(t)

    print("RewMean", np.mean(rews))
    print("RewStd", np.std(rews))
    print("LenMean", np.mean(lens))

def main():
    """
    Tests if the outputs of the Keras model is equal (or similar) to the original implementation
    which used plain TensorFlow operations.

    The code in this main function starts 1000 iterations of the version with the TensorFlow Operations.
    The code in the function jupyter_cell can be started in the training Jupyter Notebook.
    :return:
    """

    env = gym.make("RoboschoolAnt-v1")

    ac_space = env.action_space
    ob_space = env.observation_space


    ob_stat = RunningStat(
        env.observation_space.shape,
        eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
    )
    rews, lens = [], []
    for _ in range(1000):
        sess = tf.InteractiveSession()
        policy = MujocoPolicy(ob_space, ac_space, "continuous:", 0.01, 'tanh', [256, 256], 'ff')
        policy.set_ob_stat(ob_stat.mean, ob_stat.std)
        sess.run(tf.initialize_variables(tf.all_variables()))
        r, t = policy.rollout(env)
        rews.append(r.sum())
        lens.append(t)
        tf.reset_default_graph()
        sess.close()

    print("RewMean", np.mean(rews))
    print("RewStd", np.std(rews))
    print("LenMean", np.mean(lens))

if __name__ == '__main__':
    main()