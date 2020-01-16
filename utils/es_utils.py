import json
import gym
import os
import pandas as pd

from config_objects import Optimizations, ModelStructure, Config
from config_values import ConfigValues, LogColumnHeaders, EvaluationColumnHeaders
from es_errors import InvalidTrainingError
from experiments import TrainingRun


def validate_config_file(config_file):
    """
    Creates an Optimizations, ModelStructure and Config object from config_input and validates their attributes.

    config_input must be of type os.DirEntry or dict. It will try to create the respective objects from the dict
    entries it reads from the input. Afterwards the values get validated. If everything is valid the objects get
    returned, otherwise an InvalidTrainingError is raised.

    :param config_file: Configuration input of type os.DirEntry or dict
    :raises InvalidTrainingError: Will be raised when no valid objects can be created
    :return: A Valid Optimizations, ModelStructure and Config object, in this order
    """
    # Only os.DirEntry and dict are supported as input files. Could be easily extended if needed
    if isinstance(config_file, os.DirEntry):
        with open(config_file.path, encoding='utf-8') as f:
            try:
                config_dict = json.load(f)
            except json.JSONDecodeError:
                raise InvalidTrainingError("The config file {} cannot be parsed.".format(config_file.path))
    elif isinstance(config_file, dict):
        config_dict = config_file
    else:
        raise InvalidTrainingError("The input format of {} is not valid.".format(config_file))

    # Load the dictionary entries for the optimizations, model structure and the overall config
    try:
        optimization_dict = config_dict["optimizations"]
        model_structure_dict = config_dict["model_structure"]
        _config_dict = config_dict["config"]
    except KeyError:
        raise InvalidTrainingError("The loaded config does not have an entry for either the optimizations, model structure or the config.")

    # Create the Optimizations, ModelStructure and Config objects from the respective dict. If the keys in these
    # dicts are valid no exception is raised
    try:
        optimizations = Optimizations(**optimization_dict)
        model_structure = ModelStructure(**model_structure_dict)
        config = Config(**_config_dict)
    except TypeError:
        raise InvalidTrainingError("Cannot initialize the Optimizations object from {}".format(optimization_dict))

    # Now check the values for the created objects
    try:
        validate_config_objects(optimizations, model_structure, config)
    except InvalidTrainingError:
        raise

    return optimizations, model_structure, config


def validate_config_objects(optimizations, model_structure, config):
    """
    Validates the values of already created configuration objects.

    The inputs must be of Type Optimizations, ModelStructure and Config. If at least one of their values is invalid,
    a InvalidTrainingError is raised.

    :param optimizations: An object of type Optimizations for which the values shall be validated
    :param model_structure: An object of type ModelStructure for which the values shall be validated
    :param config: An object of type Config for which the values shall be validated
    :raises InvalidTrainingError: When there is at least one invalid value
    """

    if not isinstance(optimizations, Optimizations) or not isinstance(model_structure, ModelStructure) or not isinstance(config, Config):
        raise InvalidTrainingError("One of the given arguments has a false type.")

    # Check if the values for the optimizations are valid
    if not all(isinstance(v, bool) for v in optimizations):
        raise InvalidTrainingError("The values from {} cannot be used to initialize an Optimization object".format(optimizations))

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
            assert model_structure.optimizer == ConfigValues.OPTIMIZER_ADAM or model_structure.optimizer == ConfigValues.OPTIMIZER_SGD
            stepsize = model_structure.optimizer_args['stepsize']
            assert stepsize > 0
        if optimizations.discretize_actions:
            assert model_structure.ac_bins > 0
    except KeyError:
        raise InvalidTrainingError("The model structure is missing the stepsize for the gradient optimizer.")
    except TypeError or AssertionError:
        raise InvalidTrainingError("One or more of the given values for the model structure is not valid.")

    # Validate values for the config
    try:
        # Testing if the ID is valid by creating an environment with it
        gym.make(config.env_id)
    except:
        raise InvalidTrainingError("The provided environment ID {} is invalid.".format(config.env_id))

    try:
        assert config.population_size > 0
        assert config.timesteps_per_gen > 0
        assert config.num_workers > 0
        assert config.learning_rate > 0
        assert config.noise_stdev != 0
        assert config.snapshot_freq >= 0
        assert config.eval_prob >= 0

        assert (config.return_proc_mode == ConfigValues.RETURN_PROC_MODE_CR
                or config.return_proc_mode == ConfigValues.RETURN_PROC_MODE_SIGN
                or config.return_proc_mode == ConfigValues.RETURN_PROC_MODE_CR_SIGN)

        if optimizations.observation_normalization:
            assert config.calc_obstat_prob > 0

        if optimizations.gradient_optimizer:
            assert config.l2coeff > 0
    except TypeError or AssertionError:
        raise InvalidTrainingError("One or more of the given values for the config is not valid.")


def validate_log(log_input):
    """
    Reads a log file into a pandas DataFrame and validates the header columns.

    If log_input is not a valid .csv file or the header columns do not match a None object is returned and an error
    message printed.

    :param log_input: A .csv file with the to be validated log file
    :return: A pandas DataFrame containing the loaded log if it is valid, None instead
    """
    log = None

    try:
        log = pd.read_csv(log_input)
    except pd.errors.EmptyDataError:
        print("The log file {} is empty. Continuing.".format(log_input))
    except pd.errors.ParserError:
        print("The log file {} cannot be parsed. Continuing.".format(log_input))
    except FileNotFoundError:
        print("The log file {} does not exist. Continuing.".format(log_input))
    else:
        # Compare with the column headers which are set for the whole program, in the right order
        for a, b in zip(list(log), [e.value for e in LogColumnHeaders]):
            if a != b:
                log = None
                print("The log file {} does not have valid column headers. Continuing.".format(log_input))
                break
    return log


def validate_evaluation(evaluation_input):
    """
    Loads an evaluation from a .csv file into a pandas DataFrame and validates it.

    If it is invalid or not a valid .csv file a None object will be returned and a warning message is printed.
    Note that after the initial 5 header names, more columns start with Rew_i and Len_i indicating the reward and
    count of timesteps in the i-th evaluation. These will be not validated.

    :param evaluation_input: The file containing an evaluation which shall be validated
    :return: If the validation is successful a pandas DataFrame, instead None
    """
    evaluation = None

    try:
        evaluation = pd.read_csv(evaluation_input)
    except pd.errors.EmptyDataError:
        print("The evaluation file {} is empty. Continuing.".format(evaluation_input))
    except pd.errors.ParserError:
        print("The evaluation file {} cannot be parsed. Continuing.".format(evaluation_input))
    except FileNotFoundError:
        print("The evaluation file {} does not exist. Continuing.".format(evaluation_input))
    else:
        # Due to saving issues the Generation column got duplicated as the first column and is unnamed. Deleting it
        # here to avoid issues when comparing. TODO remove this when older evaluations are no longer used
        if "Unnamed: 0" in evaluation:
            del evaluation["Unnamed: 0"]

        # Compare with the column headers which are set for the whole program, in the right order
        # Take only the first 5 entries since after that individual reward and lengths are saved which are not validated
        for a, b in zip(list(evaluation)[:5], [e.value for e in EvaluationColumnHeaders][:5]):
            if a != b:
                evaluation = None
                print("The evaluation file {} does not have valid column headers. Continuing.".format(evaluation_input))
                break

    return evaluation


def index_training_folder(training_folder):
    """
    Indexes a given training folder and returns a resulting TrainingRun object.

    The function first searches for the config, validates it and creates a dictionary from it. If no valid config
    is created a InvalidTrainingException is raised, because the config is mandatory to determine a TrainingRun.
    After that the log, evaluation, model and other saved files are indexed. Then a TrainingRun object is created
    where the values of the respective files are validated. If they are valid the object is returned, otherwise
    an InvalidTrainingError is raised.

    :param training_folder: The folder which contains a started training which shall be loaded
    :return: A TrainingRun object with the attributes set to valid objects found in the training_folder
    :raises InvalidTrainingError: Will be raised if the config files is not found or invalid
    """

    if not os.path.isdir(training_folder):
        raise InvalidTrainingError("Cannot load training, {} is not a directory.".format(training_folder))

    model_files = []
    video_files = []
    ob_normalization_files = []
    optimizer_files = []

    with os.scandir(training_folder) as it:
        for entry in it:
            if entry.name.endswith(".h5") and entry.is_file():
                model_files.append(entry.path)
            elif entry.name.startswith("ob_normalization_") and entry.is_file():
                ob_normalization_files.append(entry.path)
            elif entry.name.startswith("optimizer_") and entry.is_file():
                optimizer_files.append(entry.path)
            elif entry.name.endswith(".mp4") and entry.is_file():
                video_files.append(entry.path)
            elif entry.name == "config.json" and entry.is_file():
                config_file = entry
            elif entry.name == "log.csv" and entry.is_file():
                log_file = entry
            elif entry.name == "evaluation.csv" and entry.is_file():
                evaluation_file = entry

    try:
        training_run = TrainingRun(config_file,
                                   log_file,
                                   evaluation_file,
                                   video_files,
                                   model_files,
                                   ob_normalization_files,
                                   optimizer_files)
    except InvalidTrainingError:
        raise
    else:
        return training_run


def index_experiments(experiments_folder):

    """
    TODO
    1. Check if folder
    2. List all subfolders
    3. index TrainingRun objects for each subfolder
    4. Create experiments from the subfolders and return them
    """
    pass