import click
import imageio
import os


@click.command()
@click.argument('env_id')
@click.argument('policy_file')
@click.option('--record', is_flag=True)
@click.option('--stochastic', is_flag=True)
@click.option('--extra_kwargs')
def main(env_id, policy_file, record, stochastic, extra_kwargs):
    import gym
    from gym import wrappers
    import tensorflow as tf
    from es_distributed.policies import MujocoPolicy
    import numpy as np

    env = gym.make(env_id)
    if record:
        import uuid
        env = wrappers.Monitor(env, '/tmp/' + str(uuid.uuid4()), force=True)

    if extra_kwargs:
        import json
        extra_kwargs = json.loads(extra_kwargs)

    with tf.Session():
        pi = MujocoPolicy.Load(policy_file, extra_kwargs=extra_kwargs)
        for i in range(1):
            rews, t, renders = pi.rollout(env, render=True, random_stream=np.random if stochastic else None)
            print('return={:.4f} len={}'.format(rews.sum(), t))

            if renders.size > 0:
                imageio.mimwrite(policy_file + ".mp4", renders, fps=24)

            if record:
                env.close()
                return


if __name__ == '__main__':
    main()
