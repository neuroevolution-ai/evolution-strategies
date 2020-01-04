import json
from collections import namedtuple
import os

from pathlib import Path
from enum import Enum

class GradientOptimizer(Enum):
    OPTIMIZER_ADAM = 'adam'
    OPTIMIZER_SGD = 'sgd'

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

class InvalidTrainingError(Exception):
    # TODO own file
    pass

def validate_config(config_input):
    # TODO support dict as input or make multiple methods for optimizations, config, model_structure
    if isinstance(config_input, os.DirEntry):
        with open(config_input.path, encoding='utf-8') as f:
            try:
                config_dict = json.load(f)
            except json.JSONDecodeError:
                raise InvalidTrainingError("The config file {} cannot be parsed.".format(config_input.path))
    elif isinstance(config_input, dict):
        config_dict = config_input
    else:
        # TODO raise InvalidTrainingError maybe
        return

    # Load the entry for the optimizations
    try:
        optimization_dict = config_dict["optimizations"]
    except KeyError:
        raise InvalidTrainingError("The loaded config does not have an entry for optimizations.")

    # Check if the values for the optimizations are valid
    if not all(isinstance(v, bool) for v in optimization_dict.values()):
        raise InvalidTrainingError("The values {} cannot be used to initialize an Optimization object".format(optimization_dict.values()))

    # Create an Optimizations object with the valid values and their keys. The keys have not been checked, they could
    # be false, too few or too many
    try:
        optimizations = Optimizations(**optimization_dict)
    except TypeError:
        raise InvalidTrainingError("Cannot initialize the Optimizations object from {}".format(optimization_dict))

    # Load the entry for the model structure
    try:
        model_structure_dict = config_dict["model_structure"]
    except KeyError:
        raise InvalidTrainingError("The loaded config does not have an entry for the model structure")

    # Create the ModelStructure object first to be able to easier check the values afterwards
    try:
        model_structure = ModelStructure(**model_structure_dict)
    except TypeError:
        raise InvalidTrainingError("Cannot initialize the ModelStructure object from {}".format(model_structure_dict))

    # Validate values for the ModelStructure object
    try:
        assert model_structure.ac_noise_std >= 0
        assert isinstance(model_structure.hidden_dims, list)
        assert all(isinstance(entry, int) for entry in model_structure.hidden_dims)
        assert all(hd > 0 for hd in model_structure.hidden_dims)
        assert isinstance(model_structure.nonlin_type, str)
        if optimizations.gradient_optimizer:
            # Other values in the optimizer_args dict are not checked only the mandatory stepsize is there
            # Could potentially lead to false arguments. They are handled in the Optimizer class
            assert model_structure.optimizer == GradientOptimizer.OPTIMIZER_ADAM or model_structure.optimizer == GradientOptimizer.OPTIMIZER_SGD
            stepsize = model_structure.optimizer_args['stepsize']
            assert stepsize > 0
        if optimizations.discretize_actions:
            assert model_structure.ac_bins > 0
    except KeyError:
        raise InvalidTrainingError("The model structure is missing the stepsize for the gradient optimizer.")
    except TypeError or AssertionError:
        raise InvalidTrainingError("One or more of the given values for the model structure is not valid.")




    # TODO rest of config

    optimizations = Optimizations._make(config_dict["optimizations"].values())
    opt2 = Optimizations._make([True, True, True, True, True, True])
    config = Config._make(config_dict["config"].values())
    model_structure = ModelStructure._make(config_dict["model_structure"].values())



    # validate every part of config
    # return optimizations, model_structure, config object

def index_training_folder(training_folder):
    '''
    Indexes a given training folder and returns a resulting TrainingRun object.

    The function first searches for the config, validates it and creates a dictionary from it. If no valid config
    is created a InvalidTrainingException is raised, because the config is mandatory to determine a TrainingRun.
    After that the log, evaluation, model and other saved files are indexed and validated. If they are valid, they
    get saved as attributes in the TrainingRun object, which then gets returned.

    :param training_folder: The folder which contains a started training which shall be loaded
    :return: A TrainingRun object with the attributes set to valid objects found in the training_folder
    :raises: InvalidTrainingException if the config files is not found or invalid
    '''

    # 1. Check if folder
    # 2. Indexiere Dateien
    # -> alles als Objekt speichern
    # 3. Validiere config

    #return TrainingRun()

    if not os.path.isdir(training_folder):
        raise InvalidTrainingError("Cannot load training, {} is not a directory.".format(training_folder))

    with os.scandir(training_folder) as it:
        for entry in it:
            if entry.name.endswith(".h5") and entry.is_file():
                # TODO
                pass
            elif entry.name == "config.json" and entry.is_file():
                config_file = entry

    validate_config(config_file)

    p = Path(training_folder)

    config_file = p / "config.json"

    if not config_file.is_file():
        raise InvalidTrainingError("Cannot load training, no config.json found in directory {}".format(training_folder))

    with config_file.open() as f:
        test = f

    model_files = list(p.glob("*.h5"))

    index = {}
    # for root, dirs, files in os.walk(main_directory):
    #     if 'log.csv' in files and 'config.json' in files:
    #         index[root] = files
    #
    # training_runs = []
    # for sub_dir in index:
    #     models, log, evaluation, config, video_file = [], None, None, None, None
    #     for file in index[sub_dir]:
    #         if file.endswith('.h5'):
    #             models.append(file)
    #             continue
    #         elif file.endswith('log.csv'):
    #             try:
    #                 log = pd.read_csv(os.path.join(sub_dir, file))
    #             except pd.errors.EmptyDataError:
    #                 print("The log file {} is empty. Skipping this folder({}).".format(
    #                     file, sub_dir))
    #             continue
    #         elif file.endswith('evaluation.csv'):
    #             try:
    #                 evaluation = pd.read_csv(os.path.join(sub_dir, file))
    #             except pd.errors.EmptyDataError:
    #                 print("The evaluation file {} is empty. Continuing.".format(file))
    #             continue
    #         elif file.endswith('config.json'):
    #             with open(os.path.join(sub_dir, file), encoding='utf-8') as f:
    #                 try:
    #                     config = json.load(f)
    #                 except json.JSONDecodeError as e:
    #                     print("The config file {} is empty or cannot be parsed. Skipping this folder ({}).".format(
    #                         file, sub_dir))
    #             continue
    #         elif file.endswith('.mp4'):
    #             video_file = os.path.join(sub_dir, file)
    #             continue
    #     models.sort()
    #     if log is not None and config is not None:
    #         training_runs.append(TrainingRun(sub_dir, log, config, models, evaluation, video_file))
    #
    # configs_and_runs = []
    # for run in training_runs:
    #     found = False
    #     for c in configs_and_runs:
    #         if c[0] == run.config:
    #             c[1].append(run)
    #             found = True
    #             break
    #
    #     if not found:
    #         configs_and_runs.append((run.config, [run]))
    #
    # return [Experiment(config, runs) for (config, runs) in configs_and_runs]
    #

def main():
    training_folder = "training_runs/11_11_2019-17h_19m_00s"
    index_training_folder(training_folder)

if __name__ == "__main__":
    main()

class TrainingRun:
    pass