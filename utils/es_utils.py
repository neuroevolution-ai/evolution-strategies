import json
import gym
import numpy as np
import os
import pandas as pd
import time

from config_objects import Optimizations, ModelStructure, Config
from config_values import ConfigValues, LogColumnHeaders, EvaluationColumnHeaders
from es_errors import InvalidTrainingError
from experiments import TrainingRun, Experiment


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
    4. Compare TrainingRun configurations and create Experiments for same configs
    """

    if os.isdir(experiments_folder):
        # This will get the first entry in walk and ouput the directories which are stored in the second entry of the
        # tuple
        sub_directories = next(os.walk(experiments_folder))[1]
        training_runs = []

        for sub_dir in sub_directories:
            try:
                training_run = index_training_folder(os.path.join(experiments_folder, sub_dir))
            except InvalidTrainingError:
                continue
            else:
                training_runs.append(training_run)
                # TODO check here if config of this training_run already occured, if yes add to list else create new list

        # TODO create experiments of previously created lists and return a list of these experiments
        # TODO check empty lists before returning
    else:
        # TODO remove else when else in if clause is set
        return []



def act(ob, model, random_stream=None, ac_noise_std=0):
    """Takes an observation and a model with which an action shall be calculated.

    This means the action is a prediction of the model based on the observation. In addition if one provides a random
    stream and action noise, a random factor is added to the output prediction which can help generalisation but is
    in theory more difficult to learn.

    :param ob: The observation from which the action shall be predicted
    :param model: The model which is used to predict the output
    :param random_stream: If action noise shall be used this random stream provides the random number for it, defaults
        to None
    :param ac_noise_std: Will be multiplied with the random number to generate noise, defaults to 0
    :return: The predicted action based on the observation and the model
    """
    # Calculate prediction and measure the time, on batch prediction usually faster in es context
    time_predict_s = time.time()
    action = model.predict_on_batch(ob)
    time_predict_e = time.time() - time_predict_s

    if random_stream is not None and ac_noise_std != 0:
        # Action noise can make the learned model more robust but is generally more difficult
        action += random_stream.randn(*action.shape) * ac_noise_std
    return action, time_predict_e


def rollout(
        env, model,
        env_seed=None, render=False, timestep_limit=None, save_obs=False, random_stream=None, ac_noise_std=0):
    """Steps an episode in an environment with the policy provided through the model parameter.

    One call to the rollout function will run through an episode in the provided environment. Based on the model
    observations from the environment will be used to calculate actions, i.e. a step in the environment in one
    timestep. If render is True the environment will be rendered. If timestep_limit is provided the epsiode will only
    last this number of timesteps long or shorter. If save_obs is True all observations from the environment will be
    saved and returned. If random_stream is not None and ac_noise_std is higher than 0, noise will be added to the
    prediction of the model which is known as action noise.

    :param env: The environment in which the episode will be done
    :param model: A model which will be used to predict actions. Input and output dimensions must match the environment
    :param env_seed: Sets the seed for the environment, defaults to None
    :param render: True if the environment shall be rendered, False if not, defaults to False
    :param timestep_limit: If provided the episode will run this long or shorter, defaults to None
    :param save_obs: If True, the observations get saved, defaults to False
    :param random_stream: If provided and ac_noise_std is larger than 0 action noise is added to the prediction,
        defaults to None
    :param ac_noise_std: If larger than 0 and random_stream is provided, adds action noise to the prediction, defaults
        to 0
    :return: Returns the summed reward, timesteps of the episode and the prediction time measurements. If save_obs
        is True, the observations are returned as well
    """
    env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
    rews = []
    times_predict = []
    t = 0

    if env_seed:
        try:
            env.seed(env_seed)
        except gym.error.Error as e:
            print(e)
            print("Using random seed.")
    if save_obs:
        obs = []
    if render:
        # For PyBullet environments render() must be called before reset() otherwise rendering does not work
        env.render()

    ob = env.reset()
    for _ in range(timestep_limit):
        # The model wants an input in shape (X, ob_shape). With ob[None] this will be (1, ob_shape)
        ac, time_predict = act(ob[None], model, random_stream=random_stream, ac_noise_std=ac_noise_std)
        times_predict.append(time_predict)
        if save_obs:
            obs.append(ob)
        try:
            # Similar to the shape of the observation we get (1, ac_shape) as output of the model but the environment
            # wants (ac_shape) as input. Therefore we use ac[0].
            ob, rew, done, _ = env.step(ac[0])
        except AssertionError:
            # Is thrown when for example ac is a list which has at least one entry with NaN
            raise
        rews.append(rew)
        t += 1
        if render:
            env.render()
        if done:
            break
    rews = np.array(rews, dtype=np.float32)
    if save_obs:
        return rews, t, times_predict, np.array(obs)
    return rews, t, times_predict


def load_model(model_file_path):
    # TODO paste this function from notebook
    pass
