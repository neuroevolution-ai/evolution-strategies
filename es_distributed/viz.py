import click

import os

import gym
import roboschool
from gym import wrappers
from policies import MujocoPolicy
import numpy as np

from multiprocessing import Process

# def run_policy(env_id, policy_file, record, stochastic, extra_kwargs):
#     env = gym.make(env_id)
#     if record:
#         import uuid
#         env = wrappers.Monitor(env, '/tmp/' + str(uuid.uuid4()), force=True)
#
#     if extra_kwargs:
#         import json
#         extra_kwargs = json.loads(extra_kwargs)
#
#     with tf.Session():
#         pi = MujocoPolicy.Load(policy_file, extra_kwargs=extra_kwargs)
#         # while True:
#         rews, t = pi.rollout(env, render=True,
#                              random_stream=None)  # random_stream=np.random if stochastic else None)
#         print('return={:.4f} len={}'.format(rews.sum(), t))
#
#         if record:
#             env.close()
#             return
#     tf.reset_default_graph()
#     env.close()

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

def _process_viz(env_id, policy):
    env = gym.make(env_id)
    env.reset()

    model = create_model(env)
    model.load_weights(policy)

    rollout(env, model, render=True)

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
    #policy_files = policy_files[::20]

    # Use a new process for every policy because there is a bug in the roboschool where on a second run the environment
    # does not get properly reset
    for policy in policy_files:
        p = Process(target=_process_viz, args=(env_id, policy)
        p.start()
        p.join()

if __name__ == '__main__':
    main()