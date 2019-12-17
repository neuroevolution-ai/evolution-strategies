import numpy as np
import matplotlib.pyplot as plt
from neat.reporting import BaseReporter
from IPython.display import clear_output


class PyPlotReporter(BaseReporter):

    def __init__(self, figsize=(20,5), marker='.'):
        self._marker = marker
        self._figsize = figsize
        self._hist_max_fitness = []
        self._hist_mean_fitness = []
        self._hist_generations = []
        self._hist_best_neurons = []
        self._hist_best_connections = []
        self._hist_population = []
        self._hist_species = []

    def start_generation(self, generation):
        self._hist_generations.append(generation)

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [c.fitness for c in population.values()]
        self._hist_mean_fitness.append(np.mean(fitnesses))
        self._hist_max_fitness.append(best_genome.fitness)

        neurons, connections = best_genome.size()
        self._hist_best_neurons.append(neurons)
        self._hist_best_connections.append(connections)

        self._hist_population.append(len(population))
        self._hist_species.append(len(species.species))

        self._plot()

    def _plot(self):
        clear_output(wait=True)
        fig, ax =plt.subplots(1,3,figsize=self._figsize)

        ax[0].plot(self._hist_generations, self._hist_mean_fitness, label='mean fitness', marker=self._marker)
        ax[0].plot(self._hist_generations, self._hist_max_fitness, label='max fitness', marker=self._marker)
        ax[0].legend()

        ax[1].plot(self._hist_generations, self._hist_best_neurons, label='neurons', marker=self._marker)
        ax[1].plot(self._hist_generations, self._hist_best_connections, label='connections', marker=self._marker)
        ax[1].legend()

        #        ax[2].plot(self._hist_generations, self._hist_population , label='neurons', marker=self._marker)
        ax[2].plot(self._hist_generations, self._hist_species, label='# species', marker=self._marker)
        ax[2].legend()

        plt.show()





