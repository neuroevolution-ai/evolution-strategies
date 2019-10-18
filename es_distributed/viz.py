import click

import os

import gym
import roboschool
from gym import wrappers
from policies import MujocoPolicy
import numpy as np

from multiprocessing import Process

def run_policy(env_id, policy_file, record, stochastic, extra_kwargs):
    import tensorflow as tf
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

def act(ob, model, random_stream=None):
    return model.predict(ob)

def rollout(env, model, *, render=False, timestep_limit=None, save_obs=False, random_stream=None):
    """
    If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
    Otherwise, no action noise will be added.
    """

    env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
    rews = []
    t = 0
    if save_obs:
        obs = []
    ob = env.reset()
    for _ in range(timestep_limit):
        ac = act(ob[None], model, random_stream=random_stream)[0]
        if save_obs:
            obs.append(ob)
        ob, rew, done, _ = env.step(ac)
        rews.append(rew)
        t += 1
        if render:
            env.render()
        if done:
            break
    rews = np.array(rews, dtype=np.float32)
    if save_obs:
        return rews, t, np.array(obs)
    return rews, t



def create_model(env, initial_weights=None, model_name="model", save_path=None):
    import tensorflow as tf
    args = {
        "ac_bins": "continuous:",
        "ac_noise_std": 0.01,
        # "connection_type": "ff",
        "hidden_dims": [
            256,
            256
        ],
        "nonlin_type": "tanh"
    }


    ob_space = env.observation_space
    ac_space = env.action_space
    ac_bins = args["ac_bins"]
    ac_noise_std = args["ac_noise_std"]
    hidden_dims = args["hidden_dims"]
    nonlin = args["nonlin_type"]
    # tf.keras.backend.clear_session()
    import tensorflow as tf
    nonlin = tf.tanh
    print("PID " + str(os.getpid()) + ": " + "Model entry")
    with tf.variable_scope("RoboschoolPolicy/" + model_name):
        # Observation normalization
        # ob_mean = tf.get_variable(
        #    'ob_mean', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
        # ob_std = tf.get_variable(
        #    'ob_std', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
        # in_mean = tf.placeholder(tf.float32, ob_space.shape)
        # in_std = tf.placeholder(tf.float32, ob_space.shape)
        # self._set_ob_mean_std = U.function([in_mean, in_std], [], updates=[
        # tf.assign(ob_mean, in_mean),
        # tf.assign(ob_std, in_std),
        # ])

        # Normalize observation space and clip to [-5.0, 5.0]
        # o = tf.clip_by_value((o - ob_mean) / ob_std, -5.0, 5.0)

        # Policy network

        input = x = tf.keras.Input(ob_space.shape, dtype=tf.float32)

        for hd in hidden_dims:
            x = tf.keras.layers.Dense(
                hd, activation=nonlin,
                kernel_initializer=tf.initializers.random_normal,
                bias_initializer=tf.initializers.zeros)(x)

        # Map to action
        adim = ac_space.shape[0]

        a = tf.keras.layers.Dense(
            adim,
            kernel_initializer=tf.initializers.random_normal,
            bias_initializer=tf.initializers.zeros)(x)
        model = tf.keras.Model(inputs=input, outputs=a, name=model_name)

        # Initializer for the newly created weights. TODO possible replacement tf.keras.initializers.RandomNormal
        # out = np.random.randn(*adim).astype(np.float32)
        # out *=  0.01 / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        # initializer= tf.constant(out)

    # if initial_weights is not None:
    #     set_from_flat(model, initial_weights)

    if save_path:
        model.save_weights(save_path)

    return model

def _process_viz(env_id, policy, model):
    env = gym.make(env_id)
    env.reset()

    if model:
        model = load_model(policy)
    else:
        model = create_model(env)
        model.load_weights(policy)

    rollout(env, model, render=True)

def load_model(model_path):
    import numpy as np
    import tensorflow as tf

    class Normc_initializer(tf.keras.initializers.Initializer):
        """
        Create a TensorFlow constant with random numbers normed in the given shape.
        :param std:
        :return:
        """

        def __init__(self, std=1.0):
            self.std = std

        def __call__(self, shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= self.std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)

    with tf.keras.utils.CustomObjectScope({'Normc_initializer': Normc_initializer(std=1.0)}):
        return tf.keras.models.load_model(model_path)

def visualize_no_redis(policy_files, env_id, jupyter, model):
    for policy in policy_files:
        if not jupyter:
            p = Process(target=run_policy, args=(env_id, policy, False, False, None))
            p.start()
            p.join()
        else:
            if model:
                p = Process(target=_process_viz, args=(env_id, policy, True))
                p.start()
                p.join()
            else:
                p = Process(target=_process_viz, args=(env_id, policy, False))
                p.start()
                p.join()


#from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QPushButton

@click.command()
@click.argument('env_id')
@click.argument('policies_path')
@click.option('--record', is_flag=True)
@click.option('--stochastic', is_flag=True)
@click.option('--jupyter', is_flag=True)
@click.option('--model', is_flag=True)
@click.option('--extra_kwargs')
def main(env_id, policies_path, record, stochastic, jupyter, model, extra_kwargs):
    #app = QApplication([])
    #
    #file_names = QFileDialog.getOpenFileNames(directory='/home/pdeubel/PycharmProjects/saved-weights/', filter='Policies (*.h5)')
    #app.closeAllWindows()

    #

    #env = gym.make("RoboschoolHumanoid-v1")
    #ac = env.action_space
   # os = env.observation_space

    #policy_files = [file for file in os.listdir(policies_path) if file.endswith(".h5")]
    #
    #policy_files.sort()
    #policy_files = [policy_files[25], policy_files[138]]
    #policy_files = policy_files[::10]
    policy_files = [policies_path]
    #os.chdir(policies_path)
    #
    if not jupyter and not model:
        visualize_no_redis(policy_files, env_id, False, False)

    if jupyter and not model:
        visualize_no_redis(policy_files, env_id, True, False)

    if jupyter and model:
        visualize_no_redis(policy_files, env_id, True, True)
    #
    # window = QWidget()
    # layout = QVBoxLayout()
    #
    # button_no_redis = QPushButton('No-Redis')
    # button_no_redis.clicked.connect(no_redis_click)
    # button_no_redis.clicked.connect(window.close)
    #
    #
    # button_jupyter_weights = QPushButton('Jupyter Notebook Weights')
    # button_jupyter_weights.clicked.connect(jupyter_weights_click)
    # button_jupyter_weights.clicked.connect(window.close)
    #
    # button_jupyter_model = QPushButton('Jupyter Notebook Model')
    # button_jupyter_model.clicked.connect(jupyter_model_click)
    # button_jupyter_model.clicked.connect(window.close)
    #
    # layout.addWidget(button_no_redis)
    # layout.addWidget(button_jupyter_weights)
    # layout.addWidget(button_jupyter_model)
    # window.setLayout(layout)
    # window.show()



    # Skip 20 files


    # Use a new process for every policy because there is a bug in the roboschool where on a second run the environment
    # does not get properly reset
    # for policy in policy_files:
    #     #print(policy)
    #     p = Process(target=_process_viz, args=(env_id, policy))
    #     #p = Process(target=run_policy, args=(env_id, policy, False, False, None))
    #     p.start()
    #     p.join()

    #_process_viz(env_id, policy_files[-1])


    #model = load_model(policy_files[-1])
    #env = gym.make(env_id)
    #rollout(env, model, render=True)



   #  import gym
   #  from gym import wrappers
   #  import tensorflow as tf
   #  from es_distributed.policies import MujocoPolicy
   #  import numpy as np
   #
   #  env = gym.make(env_id)
   #  if record:
   #      import uuid
   #      env = wrappers.Monitor(env, '/tmp/' + str(uuid.uuid4()), force=True)
   #
   #  if extra_kwargs:
   #      import json
   #      extra_kwargs = json.loads(extra_kwargs)
   #
   #  with tf.Session():
   #      pi = MujocoPolicy.Load(policy_files[-1], extra_kwargs=extra_kwargs)
   #      while True:
   #          rews, t = pi.rollout(env, render=True, random_stream=np.random if stochastic else None)
   #          print('return={:.4f} len={}'.format(rews.sum(), t))
   #
   #          if record:
   #              env.close()
   #              return



if __name__ == '__main__':
    main()