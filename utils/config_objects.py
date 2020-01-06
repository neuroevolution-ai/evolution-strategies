from collections import namedtuple


Config = namedtuple('Config', [
    'env_id',
    'population_size',
    'timesteps_per_gen',
    'num_workers',
    'learning_rate',
    'noise_stdev',
    'snapshot_freq',
    'return_proc_mode',
    'calc_obstat_prob',
    'l2coeff',
    'eval_prob'
])

Optimizations = namedtuple('Optimizations', [
    'mirrored_sampling',
    'fitness_shaping',
    'weight_decay',
    'discretize_actions',
    'gradient_optimizer',
    'observation_normalization',
    'divide_by_stdev'
])

ModelStructure = namedtuple('ModelStructure', [
    'ac_noise_std',
    'ac_bins',
    'hidden_dims',
    'nonlin_type',
    'optimizer',
    'optimizer_args'
])
