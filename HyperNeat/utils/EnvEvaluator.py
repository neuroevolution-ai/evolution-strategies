
import gym
import time
from typing import List, Tuple
from neat import DefaultGenome, Config
from utils.Profiling import profile_func, profile_section
from multiprocessing import Pool

class EnvEvaluator:
    def __init__(self, env_name:str, max_steps:int, n_workers:int=1, seed=None):
        self._seed = seed
        self._n_workers = n_workers
        self._env_name = env_name
        self._max_steps = max_steps

    def make_net(self, genome:DefaultGenome, config:Config): ...

    def activate_net(self, net, observation): ...


  #  def eval_all_genomes(self, genomes:List[Tuple[int, DefaultGenome]], config:Config):
  #      def worker(args):
  #          g_id, genome = args
  #          genome.fitness = self.eval_genome(genome, config)
  #          print(g_id, genome.fitness)

  #      parallel(worker, genomes, self._n_workers)


    def eval_all_genomes(self, genomes:List[Tuple[int, DefaultGenome]], config:Config):
        genome_config = [(g,config) for (i,g) in genomes]
        pool = Pool(self._n_workers)
        fits = pool.map(self.eval_genome, genome_config)
        for fit, (i, genome) in zip(fits, genomes):
            genome.fitness = fit
        pool.close()



    def eval_genome(self, args):
        genome, config = args
        net = self.make_net(genome, config)
        env = profile_func("env_make")(gym.make)(self._env_name)
        if not self._seed is None:
            env.seed(self._seed)

        fitness = 0
        observation = env.reset()
        done = False
        i = 0

        while not done and i < self._max_steps:
            action = self.activate_net(net, observation)
            observation, reward, done, _ = profile_func("env_step")(env.step)(action)
            fitness += reward
            i += 1

        env.close()
        #print(fitness)
        return fitness


    def show(self, genome:DefaultGenome, config:Config, delay=0, random=True):
        net = self.make_net(genome, config)
        env = gym.make(self._env_name)
        if not random:
            env.seed(self._seed)

        state = env.reset()
        done = False

        while not done:
            env.render()
            action = self.activate_net(net, state)
            state, _, done, _ = env.step(action)

            if delay:
                time.sleep(delay)

        env.close()


import neat
import numpy as np

class NeatPythonCPPN:
    def __init__(self, genome, config:neat.Config):
      #  assert(config.genome_config['num_outputs'] == 2)
        self._net = neat.nn.FeedForwardNetwork.create(genome, config)

    @profile_func("cppn_get_weights")
    def get_weights(self, coords):
        return self.get_weights_sequential(coords)

    def get_weights_sequential(self, coords):
        weights = np.empty(coords.shape[:2])
        biases  = np.zeros((coords.shape[1]))
        for j in range(coords.shape[1]):
            for i in range(coords.shape[0]):
                weights[i, j] = self._net.activate(coords[i, j])[0]
            biases[j] = self._net.activate([0, 0, 0, *coords[0, j, 3:]])[1]

        return weights, biases



class HyperNeatEnvEvaluator(EnvEvaluator):

    def __init__(self, env_name: str, max_steps: int, n_workers:int=1, seed=None):
        super().__init__(env_name, max_steps, n_workers=n_workers, seed=seed)


    def make_substrate(self): ...

    def make_net(self, genome: DefaultGenome, config: Config):
        with profile_section("make_cppn"):
            cppn = NeatPythonCPPN(genome, config)
        with profile_section("make_substrate"):
            self._sub = self.make_substrate()
            assert(self._sub)
        self._sub.set_weights(cppn)
        return self._sub






