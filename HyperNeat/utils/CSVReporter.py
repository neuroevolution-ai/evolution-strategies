import pandas as pd
import numpy as np
from neat.reporting import BaseReporter
import datetime
import matplotlib.pyplot as plt

class CSVReporter(BaseReporter):

    def __init__(self, file):
        self.file = file
        self.df = pd.DataFrame(columns=['generation', 'mean_fitness', 'max_fitness', 'best_neurons', 'best_connections', 'population', 'species', 'time'])


    def start_generation(self, generation):
        self.gen = generation

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [c.fitness for c in population.values()]
        neurons, connections = best_genome.size()
        data = {
            'generation': self.gen,
            'mean_fitness': np.mean(fitnesses),
            'max_fitness': best_genome.fitness,
            'best_neurons': neurons,
            'best_connections': connections,
            'population': len(population),
            'species':len(species.species),
            'time': datetime.datetime.now().isoformat()
        }
        self.df = self.df.append(data, ignore_index=True)
        self.df.to_csv(self.file, index=False)


    @staticmethod
    def plot(file, figsize=(14, 8), marker='.'):
        df = pd.read_csv(file)

        plt.figure(figsize=figsize)

        ax1 = plt.subplot(211)
        ax1.plot(df['generation'], df['mean_fitness'], label='mean fitness', marker=marker)
        ax1.plot(df['generation'], df['max_fitness'], label='max fitness', marker=marker)
        ax1.legend()

        ax2 = plt.subplot(223)
        ax2.plot(df['generation'], df['best_neurons'], label='neurons', marker=marker)
        ax2.plot(df['generation'], df['best_connections'], label='connections', marker=marker)
        ax2.legend()

        ax3 = plt.subplot(224)
        ax3.plot(df['generation'], df['species'], label='# species', marker=marker)
        ax3.legend()

        plt.show()

