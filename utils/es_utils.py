import json
import gym
from gym import wrappers
import numpy as np
import os
import pandas as pd
import re
import time

# Needed for registering the environments of these packages to the OpenAI Gym
import pybullet_envs

from .config_objects import Optimizations, ModelStructure, Config
from .config_values import ConfigValues, LogColumnHeaders, EvaluationColumnHeaders
from .es_errors import InvalidTrainingError
from . import experiments


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
        raise InvalidTrainingError(
            "The loaded config does not have an entry for either the optimizations, model structure or the config.")

    # Create the Optimizations, ModelStructure and Config objects from the respective dict. If the keys in these
    # dicts are valid no exception is raised
    try:
        optimizations = Optimizations(**optimization_dict)
        model_structure = ModelStructure(**model_structure_dict)

        # In an earlier version env_seed was not used. This will add the parameter to the config
        if not "env_seed" in _config_dict:
            _config_dict["env_seed"] = None

        config = Config(**_config_dict)
    except TypeError:
        raise InvalidTrainingError("Cannot initialize the configuration objects. The given values do not match the"
                                   " constructor.")

    # Now check the values for the created objects
    try:
        optimizations, model_structure, config = validate_config_objects(optimizations, model_structure, config)
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
        raise InvalidTrainingError(
            "The values from {} cannot be used to initialize an Optimization object".format(optimizations))

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
            assert (
                    model_structure.optimizer == ConfigValues.OPTIMIZER_ADAM.value or
                    model_structure.optimizer == ConfigValues.OPTIMIZER_SGD.value)
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
        assert isinstance(config.env_id, str)

        if config.env_id.startswith("Roboschool"):
            # Replace all Roboschool environments with PyBullet environments since they are equal
            env_name = config.env_id.replace("Roboschool", "").split("-v")[0]
            if env_name == "Walker2d":
                env_name = "Walker2D"

            # It could be that the Roboschool Atlas or Pong environment have been used, these are currently not present
            # in PyBullet but that is not a problem since then an error will be raised when trying to create the
            # environment which is caught
            new_env_id = env_name + "BulletEnv-v0"

            new_config_values = config._asdict()
            new_config_values["env_id"] = new_env_id

            # Recreate the config object with the new value
            config = Config(**new_config_values)

        # Testing if the ID is valid by creating an environment with it
        gym.make(config.env_id)
    except:
        raise InvalidTrainingError("The provided environment ID {} is invalid.".format(config.env_id))

    try:
        if config.env_seed is not None:
            assert config.env_seed >= 0

        assert config.population_size > 0
        assert config.timesteps_per_gen > 0
        assert config.num_workers > 0
        assert config.learning_rate > 0
        assert config.noise_stdev != 0
        assert config.snapshot_freq >= 0
        assert config.eval_prob >= 0

        assert (config.return_proc_mode == ConfigValues.RETURN_PROC_MODE_CR.value
                or config.return_proc_mode == ConfigValues.RETURN_PROC_MODE_SIGN.value
                or config.return_proc_mode == ConfigValues.RETURN_PROC_MODE_CR_SIGN.value)

        if optimizations.observation_normalization:
            assert config.calc_obstat_prob > 0

        if optimizations.gradient_optimizer:
            assert config.l2coeff > 0
    except TypeError or AssertionError:
        raise InvalidTrainingError("One or more of the given values for the config is not valid.")

    return optimizations, model_structure, config


def validate_log(log_input):
    """
    Reads a log file into a pandas DataFrame and validates the header columns.

    If log_input is not a valid .csv file or the header columns do not match a None object is returned and an error
    message printed.

    :param log_input: A .csv file with the to be validated log file
    :return: A pandas DataFrame containing the loaded log if it is valid, None instead
    """
    log = None
    if log_input:
        try:
            log = pd.read_csv(log_input)
        except pd.errors.EmptyDataError:
            print("The log file {} is empty. Continuing.".format(log_input))
        except pd.errors.ParserError:
            print("The log file {} cannot be parsed. Continuing.".format(log_input))
        except FileNotFoundError:
            print("The log file {} does not exist. Continuing.".format(log_input))
        except ValueError:
            print("The log file {} has a false type. Continuing.".format(log_input))
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
    # TODO support pd Dataframe as input
    evaluation = None
    if evaluation_input:
        try:
            evaluation = pd.read_csv(evaluation_input)
        except pd.errors.EmptyDataError:
            print("The evaluation file {} is empty. Continuing.".format(evaluation_input))
        except pd.errors.ParserError:
            print("The evaluation file {} cannot be parsed. Continuing.".format(evaluation_input))
        except FileNotFoundError:
            print("The evaluation file {} does not exist. Continuing.".format(evaluation_input))
        except ValueError:
            print("The evaluation file {} has a false type. Continuing.".format(evaluation_input))
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


def validate_plot_values(x_value, y_value, y_std=None, log=None, evaluation=None):
    """Checks if the given keys (x_value, y_value and y_std) can be used to access the given log or evaluation file.
    If so the data is loaded and returned. Mainly used for plotting data, therefore they are called x and y.

    :param x_value: Key for the data on the x axis
    :param y_value: Key of the data on the y axis
    :param y_std: Key for the data to represent the standard deviation through the y axis
    :param log: The log file from which the data shall be loaded
    :param evaluation: The evaluation file from which the data shall be loaded
    :return: None, None, None if on of the keys do not match, even if another would match. Otherwise the data on the
        x and y axis and if wanted the y_std data.
    """
    _x, _y, _y_std = None, None, None

    # It can occur that there is no log file or no evaluation file and the provided parameters do not match
    # the data. This will be checked first before plotting can happen
    if isinstance(x_value, LogColumnHeaders) and log is not None:
        _x = log[x_value.value]
    elif isinstance(x_value, EvaluationColumnHeaders) and evaluation is not None:
        _x = evaluation[x_value.value]

    if isinstance(y_value, LogColumnHeaders) and log is not None:
        _y = log[y_value.value]
    elif isinstance(y_value, EvaluationColumnHeaders) and evaluation is not None:
        _y = evaluation[y_value.value]

    if isinstance(y_std, LogColumnHeaders) and log is not None:
        _y_std = log[y_std.value]
    elif isinstance(y_std, EvaluationColumnHeaders) and evaluation is not None:
        _y_std = evaluation[y_std.value]

    if (_x is None or _y is None) or (y_std and _y_std is None):
        return None, None, None

    return _x, _y, _y_std


def parse_generation_number(model_file_path):
    """Parses the generation number from a given file path, for example from a model file.

    Only works with video files and files created by this program. If you plan to use it look at the code of this
    function.

    :param model_file_path: A string containing a file path from where a generation number shall be extracted.
    :return: The generation number as an integer
    """
    # We are only interested in the file name to avoid getting the false generation number remove the whole path
    model_file_path = model_file_path.split("/")[-1]

    # Get all numbers in the string, the position depends on the file extension
    match = re.findall(r"\d+", model_file_path)
    
    if match:

        if model_file_path.endswith(".mp4"):
            # Second number is the generation number in video files from the OpenAI Gym Monitor
            if len(match) < 2:
                return None
            return int(match[1])
        else:
            # First number is the generation number in files created from evolution-strategies
            # Last entry is not possible otherwise the number from the file extension is returned
            if len(match) < 1:
                return None
            return int(match[0])


def sort_dict(dictionary):
    """Sorts a dictionary with only Integer keys ascending by these keys.

    If the dictionary is not valid, i.e. does not exclusively contain Integer keys, an Error is raised.

    :param dictionary: The to be sorted dictionary
    :return: The sorted dictionary, ascending by Integer keys.
    """
    try:
        sorted_dict = dict(sorted(dictionary.items(), key=lambda x: int(x[0])))
    except ValueError or AttributeError:
        raise
    else:
        return sorted_dict


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

    config_file = None
    log_file = None
    evaluation_file = None
    model_files = {}
    video_files = {}
    ob_normalization_files = {}
    optimizer_files = {}

    with os.scandir(training_folder) as it:
        for entry in it:
            if entry.is_file():
                is_model_file = entry.name.endswith(".h5")
                is_ob_normalization_file = entry.name.startswith("ob_normalization_")
                is_optimizer_file = entry.name.startswith("optimizer_")

                if is_model_file or is_ob_normalization_file or is_optimizer_file:
                    generation = parse_generation_number(entry.path)
                    if generation is not None:
                        if is_model_file:
                            model_files[generation] = entry.path
                        elif is_ob_normalization_file:
                            ob_normalization_files[generation] = entry.path
                        elif is_optimizer_file:
                            optimizer_files[generation] = entry.path
                elif entry.name == "config.json":
                    config_file = entry
                elif entry.name == "log.csv":
                    log_file = entry
                elif entry.name == "evaluation.csv":
                    evaluation_file = entry
            elif entry.is_dir() and entry.name == "videos":
                video_files = index_video_files(entry.path)

    try:
        training_run = experiments.TrainingRun(
            config_file,
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
    """Creates a list of Experiment objects from multiple training runs located in sub directories of
    experiments_folder.

    For each valid sub directory in experiments_folder a TrainingRun object will be created. Then the configurations
    of these runs are compared and if they match an Experiment object is created from them. A list of Experiments
    gets then returned. If no experiments can be found an empty list is returned.

    :param experiments_folder: A folder which contains sub directories with training runs.
    :return: A list of Experiment objects if valid training runs are found, an empty list otherwise.
    """
    indexed_experiments = []

    if os.path.isdir(experiments_folder):
        # This will get the first entry in walk and ouput the directories which are stored in the second entry of the
        # tuple
        sub_directories = next(os.walk(experiments_folder))[1]
        different_experiments = {}

        # Only checks direct sub directories for simplicity
        i = 0
        for sub_dir in sub_directories:
            try:
                training_run = index_training_folder(os.path.join(experiments_folder, sub_dir))
            except InvalidTrainingError:
                continue
            else:
                for key, value in different_experiments.items():
                    current_item = value[-1]
                    # Matching configuration objects mean the TrainingRun's are part of the same experiment
                    if (training_run.optimizations == current_item.optimizations and
                            training_run.model_structure == current_item.model_structure and
                            training_run.config == current_item.config):
                        different_experiments[key].append(training_run)
                        break
                else:
                    # TrainingRun's with the current config not found, therefore create a new entry
                    different_experiments[i] = [training_run]
                    i += 1
        for training_run_list in different_experiments.values():
            sample_training_run = training_run_list[-1]
            indexed_experiments.append(experiments.Experiment(
                sample_training_run.optimizations, sample_training_run.model_structure, sample_training_run.config,
                training_run_list))

    return indexed_experiments


def index_video_files(video_directory):
    """Indexes the video sub directory in a TrainingRun folder. It simply searches all sub directories for .mp4 files
    and adds them to a dictionary with the key being the name of the folder. These are the generation numbers of the
    respective video file.

    :param video_directory: A path to a directory containing sub directories with numbers as directory names. These
        in turn contain the video files
    :return: A dictionary with the key being the generation number and the value being the video file for this
        generation
    """
    video_files = {}
    with os.scandir(video_directory) as it:
        # Search all sub directories
        for entry in it:
            if entry.is_dir():
                with os.scandir(entry.path) as _it:
                    # Search all files for a sub directory of a generation
                    for _entry in _it:
                        if _entry.is_file() and _entry.name.endswith(".mp4"):
                            video_files[int(entry.name)] = _entry.path
    return video_files


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
    env_timestep_limit = env.spec.max_episode_steps
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
        return rews, t, times_predict, np.array(obs, dtype=np.float32)
    return rews, t, times_predict


def load_model(model_file_path):
    """Loads a tf.keras model from the given file path and returns it.

    If an OSError is raised it will be printed and None is returned.
    Note that this function imports tensorflow which can interfere with multiprocessing in combination with TensorFlow.

    :param model_file_path: The file path to the model which shall be loaded.
    :return: A model if the provided file is a valid model, None if not
    """
    # The imports need to be inside the function definition since TensorFlow sessions interfere with multiprocessing
    # and the program does not work
    import tensorflow as tf

    from .es_custom_layers import Normc_initializer, ObservationNormalizationLayer, DiscretizeActionsUniformLayer

    custom_objects = {"Normc_initializer": Normc_initializer,
                      "ObservationNormalizationLayer": ObservationNormalizationLayer,
                      "DiscretizeActionsUniformLayer": DiscretizeActionsUniformLayer}

    try:
        model = tf.keras.models.load_model(model_file_path, custom_objects=custom_objects)
    except OSError as e:
        print(e)
        return None
    return model


def rollout_helper(
        env_id, model_file_path,
        record=False, record_force=False,
        env_seed=None, render=False, timestep_limit=None):
    """Wraps rollout in a helper function where an environment id and a model file path can be provided instead of the
    corresponding objects. This will be used for example in the evaluation where model file paths are processed in a
    multiprocessing pool. In addition, the environment can be recorded for visualising purposes.

    In a multiprocessing pool a function is given to the pool. When iterating through the model file paths, it is
    simpler to use this helper function for creating the environment and model object.

    :param env_id: A valid environment ID for which an environment shall be created
    :param model_file_path: A valid file path to the model file which is used for the rollout
    :param record: If True, the episode will be recorded as a .mp4 video and saved in the same folder as the model
    :param record_force: If True all recordings in the folder of the model will be deleted
    :param env_seed: The seed for the environment, is used in rollout()
    :param render: If True, the environment will be rendered during the episode, is used in rollout()
    :param timestep_limit: Specifies the maximum timesteps for the episode, is used in rollout()
    :return: A NumPy array with the reward for the episode and the number of timesteps
    """
    env = gym.make(env_id)
    env.reset()

    if record:
        generation = parse_generation_number(model_file_path)
        video_directory = os.path.join(os.path.dirname(model_file_path), "videos", str(generation))

        env.env._render_width = 1024
        env.env._render_height = 768

        env = wrappers.Monitor(env, video_directory, force=record_force)

    model = load_model(model_file_path)

    try:
        rewards, length, _ = rollout(
            env, model,
            env_seed=env_seed, render=render, timestep_limit=timestep_limit)
    except AssertionError:
        if record:
            return None
        # Is thrown when for example ac is a list which has at least one entry with NaN
        return np.array([None, None])

    if record:
        # Variables are present since they are created inside an "if record" above
        for file in os.listdir(video_directory):
            if file.endswith(".mp4"):
                return os.path.join(video_directory, file)

    return np.array([rewards.sum(), length])
