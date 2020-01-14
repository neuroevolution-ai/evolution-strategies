import imageio
from IPython.display import Video
import warnings
import gym


def create_video(self, genome, config, fout="./out.mp4", fps=30, quality=7, render=False):
    net = self.make_net(genome, config)
    env = gym.make(self._env_name)

    state = env.reset()
    done = False
    imgs = []

    while not done:
        if render:
            env.render()
        action = self.activate_net(net, state)
        state, _, done, _ = env.step(action)
        imgs.append(state)

    env.close()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imageio.mimwrite(fout, imgs, fps=fps, quality=quality)

    return Video(fout)