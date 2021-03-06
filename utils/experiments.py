import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd

from .es_errors import InvalidTrainingError
from . import es_utils
from . import config_values


class TrainingRun:
    """A TrainingRun is one execution of the evolution-strategies.ipynb Notebook. Therefore it represents a result of
    the evolution strategies with a given config file.

    To initialize a TrainingRun object it is recommended to use the index_training_run() method in es_utils.py, since
    it automatically validates all the files in the folder, indexes them and returns a TrainingRun object.

    Such a TrainingRun object can be used to evaluate, visualize and plot the TrainingRun.
    """
    def __init__(self,
                 config_file,
                 log_file=None,
                 evaluation_file=None,
                 video_files=None,
                 model_files=None,
                 ob_normalization_files=None,
                 optimizer_files=None):
        """Creates a TrainingRun object.

        :param config_file: The config file created during the execution of the TrainingRun
        :param log_file: The log file created during the execution of the TrainingRun
        :param evaluation_file: The evaluation file created during the evaluation of a TrainingRun
        :param video_files: The video files created during different visualizations from a TrainingRun
        :param model_files: The model files created during the execution of a TrainingRun
        :param ob_normalization_files: The observation normalization files created during the execution of a TrainingRun
        :param optimizer_files: The optimizer files created during the execution of a TrainingRun
        """
        try:
            self.optimizations, self.model_structure, self.config = es_utils.validate_config_file(config_file)
        except InvalidTrainingError:
            raise

        # Could return None objects
        self.log = es_utils.validate_log(log_file)
        self.evaluation = es_utils.validate_evaluation(evaluation_file)

        # Could be None, therefore initialize empty list if so
        try:
            self.optimizer_files = es_utils.sort_dict(dict(optimizer_files)) if optimizer_files else {}
            self.ob_normalization_files = \
                es_utils.sort_dict(dict(ob_normalization_files)) if ob_normalization_files else {}
            self.model_files = es_utils.sort_dict(dict(model_files)) if model_files else {}
            self.video_files = es_utils.sort_dict(dict(video_files)) if video_files else {}
        except ValueError or AttributeError:
            # The sorting method will throw one of these errors if the input or the keys are invalid
            raise InvalidTrainingError(
                "Saving the file dictionaries raised an issue, possibly when sorting it by the keys. Aborting.")

    def evaluate(self, env_seed=None, num_evaluations=5, num_workers=os.cpu_count(), force=False, save=True):
        """Evaluate the TrainingRun by running num_evaluation episodes per model file from this TrainingRun. The
        rewards and lengths (timesteps per episode) get saved and median and standard deviation of the results are
        calculated and saved in self.evaluation in a pandas DataFrame.

        When using env_seed, num_evaluation will be set to 1 automatically since the reward and timesteps are always
        the same for the same environment seed. When using force the current evaluation save in the TrainingRun
        is overwritten. With the save parameter the evaluation is stored as a .csv file.

        :param env_seed: The environment seed which shall be used, if it is set, num_evaluations is set to 1,
            defaults to None
        :param num_evaluations: The number of episodes which shall be run per model, defaults to 5
        :param num_workers: Defines the number of workers used to calculate the evaluations, defaults to os.cpu_count()
        :param force: If True, the saved evaluation, if one exists, will be overwritten, defaults to False
        :param save: If True, the evaluation will be saved as a .csv file, defaults to True
        :return: A pandas DataFrame containing the evaluation or None if an error occured
        """
        if not force:
            # if [pd.DataFrame] throws an error therefore use sightly longer version
            if self.evaluation is not None:
                print("This TrainingRun has already been evaluated. If a new evaluation shall be done, set force=True.")
                return self.evaluation

        if not self.model_files:
            print("This TrainingRun cannot be evaluated because it does not have any model files.")
            return None

        if env_seed:
            if not isinstance(env_seed, int) or env_seed < 0:
                env_seed = None
            else:
                num_evaluations = 1
        else:
            assert num_evaluations > 0

        assert num_workers > 0

        # Is later needed for the head row in the pd.DataFrame
        columns = [col.value for col in config_values.EvaluationColumnHeaders]

        for i in range(num_evaluations):
            columns.append("Rew_{}".format(i))
            columns.append("Len_{}".format(i))

        evaluation_data = []
        with mp.Pool(processes=num_workers) as pool:
            results = []
            # First generate the reward data by running num_evaluations episodes in parallel
            for i, (generation, model_file) in enumerate(self.model_files.items()):
                generation_results = []
                for _ in range(num_evaluations):
                    generation_results.append(
                        pool.apply_async(
                            func=es_utils.rollout_helper, args=(
                                self.config.env_id, model_file, False, True, env_seed, False, None)))

                results.append((generation, generation_results))

            # In a mp.Pool the calculated data must be gathered using .get() on the results
            for (gen, generation_results) in results:
                for i, generation_result in enumerate(generation_results):
                    generation_results[i] = generation_result.get()
                    if generation_result == [None, None]:
                        return None  # Assertion error in rollout

                # Remember rollout_helper returns one array with two entries. First the reward, second the timesteps
                generation_rewards = np.array(generation_results)[:, 0]
                generation_lengths = np.array(generation_results)[:, 1]

                # TODO maybe match EvaluationColumnHeaders. This is kind of arbitrary here
                generation_row = [
                    gen, num_evaluations, np.mean(generation_rewards), np.std(generation_rewards),
                    np.mean(generation_lengths)
                ]

                assert len(generation_rewards) == len(generation_lengths)
                for reward, length in zip(generation_rewards, generation_lengths):
                    # Extend with tuple is faster than append or extend with list:
                    # https://stackoverflow.com/a/14446207/11162327
                    generation_row.extend((reward, length))

                evaluation_data.append(generation_row)

        self.evaluation = pd.DataFrame(data=evaluation_data, columns=columns)

        if save:
            # Access any model file (at least one is existing otherwise we have already returned) and get the directory
            directory = os.path.dirname(next(iter(self.model_files.values())))
            self.evaluation.to_csv(os.path.join(directory, 'evaluation.csv'))

        # TODO run validate_evaluation when pd dataframe is supported as input
        return self.evaluation

    def visualize(self, env_seed=None, generation=-1, force=False):
        """This will visualize a generation by running an episode and recording it. The resulting video file will
        be returned as a path to the video file.

        :param env_seed: This sets the environment seed for the episode, defaults to None
        :param generation: Specify which generation shall be visualized, defaults to -1 which means the last generation
            in the model_files dict
        :param force: If True, an already saved video of the generation will be overwritten, defaults to False
        :return: The path to the video file or None if an error occured, for example the generation is invalid or a
            model file for the specified generation does not exist
        """
        if not self.model_files:
            return None

        if generation == -1:
            # We are using a dict so we need to do a little more to get the last generation
            generation = list(self.model_files.keys())[-1]

        video_file_path = None
        model_file_path = None
        try:
            video_file_path = self.video_files[generation]
        except KeyError:
            # Video file not found
            pass

        # Video file for this generation is not present, check if a model file for this generation exists
        try:
            model_file_path = self.model_files[generation]
        except KeyError:
            # Model file not found
            pass

        if not video_file_path and not model_file_path:
            # Force is now irrelevant
            # We have no visualization and no model file for this generation
            print(
                "There is no model file for generation {}.",
                "Please provide another generation for the visualization.".format(generation))
            return None

        if video_file_path:
            if not force:
                # Can return the current video file
                return video_file_path
            else:
                if not model_file_path:
                    # Wanted to redo the visualization but the model file could not be found, therefore return the old
                    # one
                    print(
                        "Cannot create another visualization for generation {}, because the model file is no longer"
                        " present. Returning the current visualization.".format(generation))
                    return video_file_path

        if env_seed:
            if not isinstance(env_seed, int) or env_seed < 0:
                env_seed = None

        video_file_path = es_utils.rollout_helper(
            self.config.env_id, model_file_path, record=True, record_force=force, env_seed=env_seed)

        # Could return None if not, add it to the dict
        if video_file_path:
            self.video_files[generation] = video_file_path

        return video_file_path

    def plot_training_run(
            self, x_value, y_value, y_std=None, x_label=None, y_label=None, plot_label=None, fig=None, ax=None):
        """Plots the given x_value and y_value keys with possible shaded area by providing the y_std key.

        The parameters x_value, y_value and y_std must be of type config_values.LogColumnHeaders or
        config_values.EvaluationColumnHeaders.

        :param x_value: The key for data from the log or evaluation file for the x-axis
        :param y_value: The key for data from the log or evaluation file for the y-axis
        :param y_std: The key for data from the log or evaluation file for the shaded area indicating the
            standard deviation, defaults to None
        :param x_label: A string to label the x-axis, defaults to None
        :param y_label: A string to label the y-axis, defaults to None
        :param plot_label: When using mutiple plots in one Axes this is the name shown in the legend
        :param fig: A matplotlib figure which can be used to have multiple plots in one figure
        :param ax: A matplotlib axes, which can be used to have multiple plots in one figure
        :return: Nothing, the plot will be automatically shown when no errors occurred, otherwise an error message
            is printed.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        _x, _y, _y_std = es_utils.validate_plot_values(
            x_value, y_value, y_std=y_std, log=self.log, evaluation=self.evaluation)

        if _x is None and _y is None and _y_std is None:
            print(
                "'{}', '{}' and/or '{}' are invalid keys for the log and/or evaluation or the log and/or evaluation"
                " does not exist for this training run, and can therefore not be plotted."
                " Please provide valid keys.".format(x_value, y_value, y_std))
            return

        ax.plot(_x, _y, label=plot_label)
        if y_std:
            # If the color of the plots and the shaded area shall be changed, this is the place to do so
            ax.fill_between(_x, _y - _y_std, _y + _y_std, alpha=0.5)

        if x_label and isinstance(x_label, str):
            plt.xlabel(x_label)

        if y_label and isinstance(y_label, str):
            plt.ylabel(y_label)

        return fig, ax

    def delete_files(self, interval=1, model_files=False, ob_normalization_files=False, optimizer_files=False):
        """Deletes the files from disk depending on which interval and which type of files is given.

        :param interval: Iterates in this interval through the files and deletes them, defaults to 1
        :param model_files: If True the model files get deleted, defaults to False
        :param ob_normalization_files: If True the observation normalization files get deleted, defaults to False
        :param optimizer_files: If True the optimizer files get deleted, defaults to False
        :return: None
        """
        if interval <= 0:
            return

        dict_list = []
        if model_files and self.model_files:
            dict_list.append(self.model_files)
        if ob_normalization_files and self.ob_normalization_files:
            dict_list.append(self.ob_normalization_files)
        if optimizer_files and self.optimizer_files:
            dict_list.append(self.optimizer_files)

        for _dict in dict_list:
            for k, v in list(_dict.items())[::interval]:
                try:
                    os.remove(v)
                    # Don't forget to delete the entry from the dict to avoid further referencing.
                    del _dict[k]
                except FileNotFoundError:
                    print("File not found, continuing.")
                    continue

    def get_training_state(self, generation=-1):
        """Returns the training state from the given generation. If the model and other files do not exist for this
        generation, only the configuration is returned.
        :param generation: Indicates the point from which the training state shall be returned, defaults to -1 (latest)
        :return: Optimizations, ModelStructure, Config objects first, then model files, observation normalization files,
            and optimizer files if they exist for the given generation
        """
        if generation == -1:
            # We are using a dict so we need to do a little more to get the last generation
            generation = list(self.model_files.keys())[-1]

        try:
            model_file = self.model_files[generation]
        except KeyError or NameError:
            return self.optimizations, self.model_structure, self.config, None, None, None

        try:
            ob_normalization_file = self.ob_normalization_files[generation]
        except KeyError or NameError:
            ob_normalization_file = None

        try:
            optimizer_file = self.optimizer_files[generation]
        except KeyError or NameError:
            optimizer_file = None

        return self.optimizations, self.model_structure, self.config, model_file, ob_normalization_file, optimizer_file


class Experiment:
    """An Experiment is a collection of multiple identical TrainingRuns, therefore resembling multiple instances of
    an execution of the evolution-strategies.ipynb with equal configurations.

    Identical TrainingRuns mean that the optimizations, model_structure and config file of each TrainingRun have
    matching values.

    This can be used to validate a result over multiple runs to scientifically strengthen the result. As with
    the TrainingRun class it is recommended to initialize an object using validate_experiment_folder() from es_utils.py.
    This way the files get validated, indexed and an Experiment object is returned.

    When the Experiments is evaluated, visualized and plotted, usually all contained TrainingRun objects get used, but
    this can be configured in the method parameters.
    """

    def __init__(self, optimizations, model_structure, config, training_runs):
        """Creates an Experiment object.

        :param optimizations: The optimizations config file used to create each TrainingRun
        :param model_structure: The model structure config file used to create each TrainingRun
        :param config: The config file used to create each TrainingRun
        :param training_runs: The TrainingRuns of this Experiment
        """
        # Experiment object should only be created through index_experiments() method from es_util. There the objects
        # get validated
        self.optimizations = optimizations
        self.model_structure = model_structure
        self.config = config
        self.training_runs = training_runs

    def evaluate(
            self,
            env_seed=None, num_evaluations=5, num_workers=os.cpu_count(), force=False, save=True):
        """Evaluates every TrainingRun object of this Experiment. The parameters are directly given to the evaluate
        method of TrainingRun.

        :param env_seed: The environment seed which shall be used, if it is set, num_evaluations is set to 1,
            defaults to None
        :param num_evaluations: The number of episodes which shall be run per model, defaults to 5
        :param num_workers: Defines the number of workers used to calculate the evaluations, defaults to os.cpu_count()
        :param force: If True, the saved evaluation, if one exists, will be overwritten, defaults to False
        :param save: If True, the evaluation will be saved as a .csv file, defaults to True
        :return: A list of the evaluations
        """
        evaluations = []
        for training_run in self.training_runs:
            try:
                evaluation = training_run.evaluate(
                    env_seed=env_seed, num_evaluations=num_evaluations, num_workers=num_workers, force=force, save=save)
            except AssertionError as e:
                print("One of the parameters for the evaluation is false and threw an AssertionError:", e)
                return []
            else:
                if evaluation is not None:
                    evaluations.append(evaluation)

        return evaluations

    def visualize(self, env_seed=None, generation=-1, force=False):
        """Visualizes every TrainingRun object of this Experiment by running the visualization method from TrainingRun
        with the parameters given here.

        :param env_seed: This sets the environment seed for the episode, defaults to None
        :param generation: Specify which generation shall be visualized, defaults to -1 which means the last generation
            in the model_files dict
        :param force: If True, an already saved video of the generation will be overwritten, defaults to False
        :return: A list of the video files, if a TrainingRun could not be visualized, the object is ignored
        """
        video_files = []
        for training_run in self.training_runs:
            video_file = training_run.visualize(env_seed=env_seed, generation=generation, force=force)
            if video_file:
                video_files.append(video_file)

        return video_files

    def delete_files(self, interval=1, model_files=False, ob_normalization_files=False, optimizer_files=False):
        """Deletes the files in the stored TrainingRun objects of this experiment, depending on the parameters.

        :param interval: Iterates in this interval through the files and deletes them, defaults to 1
        :param model_files: If True the model files get deleted, defaults to False
        :param ob_normalization_files: If True the observation normalization files get deleted, defaults to False
        :param optimizer_files: If True the optimizer files get deleted, defaults to False
        :return: None
        """
        for training_run in self.training_runs:
            training_run.delete_files(
                interval=interval, model_files=model_files, ob_normalization_files=ob_normalization_files,
                optimizer_files=optimizer_files)

    def plot_experiment(self, x_value, y_value, y_std=None, x_label=None, y_label=None, plot_label=None, fig=None, ax=None):
        """Plots the given x_value and y_value keys with possible shaded area by providing the y_std key.

        Calculates the mean value across the indexed TrainingRun objects of this experiment.
        The parameters x_value, y_value and y_std must be of type config_values.LogColumnHeaders or
        config_values.EvaluationColumnHeaders.

        :param x_value: The key for data from the log or evaluation file for the x-axis
        :param y_value: The key for data from the log or evaluation file for the y-axis
        :param y_std: The key for data from the log or evaluation file for the shaded area indicating the
            standard deviation, defaults to None
        :param x_label: A string to label the x-axis, defaults to None
        :param y_label: A string to label the y-axis, defaults to None
        :param plot_label: When using mutiple plots in one Axes this is the name shown in the legend
        :param fig: A matplotlib figure which can be used to have multiple plots in one figure
        :param ax: A matplotlib axes, which can be used to have multiple plots in one figure
        :return: Nothing, the plot will be automatically shown when no errors occurred, otherwise an error message
            is printed.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        x_data = []
        y_data = []
        y_std_data = []
        for training_run in self.training_runs:
            assert isinstance(training_run, TrainingRun)
            _x, _y, _y_std = es_utils.validate_plot_values(
                x_value, y_value, y_std=y_std, log=training_run.log, evaluation=training_run.evaluation)

            if _x is None and _y is None and _y_std is None:
                continue

            x_data.append(_x)
            y_data.append(_y)
            if y_std:
                y_std_data.append(_y_std)

        if x_data and y_data:
            # It can happen that the individual TrainingRuns require a different amount of generations for the same
            # number of timesteps. For the standard deviation from NumPy it is however required to have the same
            # amount of rows, therefore we drop the additional rows.
            # x_data, y_data and y_std_data are identical in this terms
            min_count_rows = np.inf
            for data in x_data:
                min_count_rows = min(min_count_rows, len(data))

            for i, data in enumerate(x_data):
                try:
                    x_data[i] = data.drop(min_count_rows)
                except KeyError:
                    # This gets thrown when a Series with the least amount of rows is currently viewed
                    # And we do not want to drop any lines here
                    continue

            for i, data in enumerate(y_data):
                try:
                    y_data[i] = data.drop(min_count_rows)
                except KeyError:
                    # This gets thrown when a Series with the least amount of rows is currently viewed
                    # And we do not want to drop any lines here
                    continue

            x = np.mean(x_data, axis=0)
            y = np.mean(y_data, axis=0)
            ax.plot(x, y, label=plot_label)

            if y_std:
                for i, data in enumerate(y_std_data):
                    try:
                        y_std_data[i] = data.drop(min_count_rows)
                    except KeyError:
                        # This gets thrown when a Series with the least amount of rows is currently viewed
                        # And we do not want to drop any lines here
                        continue

                _y_std = np.std(y_std_data, axis=0)
                # If the color of the plots and the shaded area shall be changed, this is the place to do so
                ax.fill_between(x, y - _y_std, y + _y_std, alpha=0.5)

            if x_label and isinstance(x_label, str):
                plt.xlabel(x_label)

            if y_label and isinstance(y_label, str):
                plt.ylabel(y_label)

        return fig, ax
