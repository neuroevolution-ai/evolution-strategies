import multiprocessing as mp
import numpy as np
import os
import pandas as pd

from es_errors import InvalidTrainingError
import es_utils
import config_values

class TrainingRun:
    def __init__(self,
                 config_file,
                 log_file=None,
                 evaluation_file=None,
                 video_files=None,
                 model_files=None,
                 ob_normalization_files=None,
                 optimizer_files=None):
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
            if self.evaluation:
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
        columns = []
        for col in config_values.EvaluationColumnHeaders:
            columns.append(col.value)

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

        # TODO run validate_evaluation
        return self.evaluation

    def visualize(self, env_seed=None, generation=-1, force=False):

        """
        TODO
        1. assert env_seed integer; assert generation integer, -1 oder >= 0;
        2. Wenn nicht force nur weitermachen wenn visualisierung noch nicht existiert
        3. Für angegebene Generation visualisierung durchführen, ggf. seed setzen
        4. video Objekt erstellen, filepath setzen und returnen
        """
        pass

    def plot_training_run(self, x_value, y_value, y_std=None, x_label=None, y_label=None):

        """
        TODO
        1. x_value, y_value, ggf y_std checken ob diese Spaltennamen exisitieren
        2. assert x_label, y_label string oder None
        3. Values plotten mit matplotlib
        Auf eventuelle Fehler achten:
        - Datenspalten leer
        - Evaluation möglicherweise nicht existent falls daraus geplottet werden soll
        """
        pass

    def delete_files(self, interval=1, model_files=False, ob_normalization_files=False, optimizer_files=False):

        """
        TODO
        1. assert interval > 0
        2. Checken welche files gelöscht werden sollen, dann im interval die Dateien löschen
        Achtung:
        - möglicherweise leere dicts bei den files
        - interval könnte irgendwie überlaufen oder zu klein sein
        """
        pass

    def get_training_state(self, generation=-1):

        """
        TODO
        1. assert generation >= 0 oder -1
        2. Entsprechende files returnen. Wenn es model file nicht gibt -> error. Wenn es ob_norm oder optimizer nicht gibt,
        jeweils None returnen muss enstprechend beim Aufrufer gehandled werden
        """
        pass


        #
        # if not isinstance(video_files, list):
        #     self.video_files = []
        # else:
        #     self.video_files = video_files
        #
        # if not isinstance(model_files, list):
        #     self.model_files = []
        # else:
        #     self.model_files = model_files
        #
        # if not isinstance(ob_normalization_files, list):
        #     self.ob_normalization_files = []
        # else:
        #     self.ob_normalization_files = ob_normalization_files
        #
        # if not isinstance(optimizer_files, list):
        #     self.optimizer_files = []
        # else:
        #     self.optimizer_files = optimizer_files

    # def get_training_state(self):
    #
    #     current_model_file, current_optimizer_file, current_ob_normalization_file = None, None, None
    #
    #     if self.model_files is not None:
    #         current_model_file = self.model_files[-1]
    #
    #     if self.current
    #
    #     return self.optimizations, self.model_structure, self.config
    #     pass

    # def __init__(self, save_directory, log, config, model_file_paths, evaluation=None, video_file=None):
    #     self.save_directory = save_directory
    #     self.log = log
    #     self.config = config
    #
    #     if not model_file_paths:
    #         self.no_models = True
    #         self.model_file_paths = None
    #     else:
    #         self.no_models = False
    #         self.model_file_paths = [os.path.join(save_directory, model) for model in model_file_paths]
    #
    #     if evaluation is not None:
    #         self.no_evaluation = False
    #         self.evaluation = evaluation
    #         self.data = self.merge_log_eval()
    #     else:
    #         self.no_evaluation = True
    #         self.evaluation = None
    #         self.data = None
    #
    #     if video_file is not None:
    #         self.no_video = False
    #     else:
    #         self.no_video = True
    #     self.video_file = video_file
    #
    #     if self.log is None or self.config is None:
    #         print("This TrainingRun is missing either the log file or the configuration file. It will not "
    #               + "work as expected.")
    #
    # def merge_log_eval(self):
    #     if self.log is not None and self.evaluation is not None:
    #         return self.log.merge(self.evaluation[['Generation', 'Eval_Rew_Mean', 'Eval_Rew_Std', 'Eval_Len_Mean']],
    #                               on='Generation')
    #     return None
    #
    # def parse_generation_number(self, model_file_path):
    #     try:
    #         number = int(model_file_path.split('snapshot_')[-1].split('.h5')[0])
    #         return number
    #     except ValueError:
    #         return None
    #
    # def evaluate(self, force=False, eval_count=5, skip=None, save=False, delete_models=False):
    #     if not force:
    #         if self.data is not None:
    #             return self.data
    #
    #     if self.no_models:
    #         print("No models given for that training run, so no new evaluation is possible. You can still plot" +
    #               " your data if you have an evaluation.csv or log.csv.")
    #         return None
    #
    #     head_row = ['Generation', 'Eval_per_Gen', 'Eval_Rew_Mean', 'Eval_Rew_Std', 'Eval_Len_Mean']
    #
    #     for i in range(eval_count):
    #         head_row.append('Rew_' + str(i))
    #         head_row.append('Len_' + str(i))
    #
    #     data = []
    #
    #     results_list = []
    #     pool = Pool(os.cpu_count())
    #
    #     for model_file_path in self.model_file_paths[::skip]:
    #         results = []
    #         gen = self.parse_generation_number(model_file_path)
    #
    #         for _ in range(eval_count):
    #             results.append(pool.apply_async(func=self.run_model, args=(model_file_path,)))
    #         results_list.append((results, gen))
    #
    #     for (results, gen) in results_list:
    #         for i in range(len(results)):
    #             results[i] = results[i].get()
    #             if results[i] == [None, None]:
    #                 print("The provided model file produces non finite numbers. Stopping.")
    #                 return
    #
    #         rewards = np.array(results)[:, 0]
    #         lengths = np.array(results)[:, 1]
    #
    #         row = [gen,
    #                eval_count,
    #                np.mean(rewards),
    #                np.std(rewards),
    #                np.mean(lengths)]
    #
    #         assert len(rewards) == len(lengths)
    #         for i in range(len(rewards)):
    #             row.append(rewards[i])
    #             row.append(lengths[i])
    #
    #         data.append(row)
    #
    #     pool.close()
    #     pool.join()
    #
    #     self.evaluation = pd.DataFrame(data, columns=head_row)
    #     if save:
    #         self.save_evaluation()
    #     # Only copy the mean values in the merged data
    #     self.data = self.merge_log_eval()
    #
    #     if delete_models:
    #         self.delete_model_files
    #
    #     return self.data
    #
    # def delete_model_files(self, save_last=False):
    #     if save_last:
    #         self.model_file_paths = self.model_file_paths[:-1]
    #     for model_file_path in self.model_file_paths:
    #         os.remove(model_file_path)
    #
    # def plot_reward_timestep(self):
    #     if self.data is not None:
    #         plot(self.data.TimestepsSoFar, 'Timesteps', self.data.Eval_Rew_Mean, 'Cummulative reward')
    #     else:
    #         print("You did not evaluate these results. The evaluated mean reward displayed was computed during training"
    #               + "and can have missing values!")
    #         plot(self.log.TimestepsSoFar, 'Timesteps', self.log.EvalGenRewardMean, 'Cummulative reward')
    #
    # def save_evaluation(self):
    #     if self.evaluation is not None:
    #         self.evaluation.to_csv(os.path.join(self.save_directory, 'evaluation.csv'))
    #
    # def visualize(self, force=False):
    #     if self.no_models:
    #         # Error message in Experiment
    #         return None
    #     if not force:
    #         if self.video_file is not None:
    #             return self.video_file
    #
    #     latest_model = self.model_file_paths[-1]
    #
    #     with Pool(os.cpu_count()) as pool:
    #         pool.apply(func=self.run_model, args=(latest_model, True))
    #
    #     for file in os.listdir(self.save_directory):
    #         if file.endswith('.mp4'):
    #             self.video_file = os.path.join(self.save_directory, file)
    #
    #     return self.video_file
    #



class Experiment:
    # TODO docstring

    def __init__(self, optimizations, model_structure, config, training_runs):
        # TODO docstring
        # Experiment object should only be created through index_experiments() method from es_util. There the objects
        # get validated
        self.optimizations = optimizations
        self.model_structure = model_structure
        self.config = config
        self.training_runs = training_runs

    def evaluate(
            self,
            env_seed=None, num_evaluations=5, num_workers=os.cpu_count(), force=False, save=True):

        """
        TODO
        1. Evaluiere jedes TrainingRun mit den Parametern
        2. Auf assertions / fehler achten eingehen
        """
        pass

    def visualize(self, env_seed=None, generation=-1, force=False):

        """
        TODO
        1. Jedes Training run visualize mit den Parametern aufrufen
        2. Auf assertions / fehler achten
        """
        pass

    def delete_files(self, interval=1, model_files=False, ob_normalization_files=False, optimizer_files=False):

        """
        TODO
        1. Auf jedem TrainingRun delete_files mit den Parametern aufrufen
        2. Auf fehler /Assertions achten
        """
        pass

    def plot_experiment(self, x_value, y_value, y_std=None, x_label=None, y_label=None):

        """
        TODO
        1. x_value, y_value, ggf. y_std überprüfen
        2. ggf x_label, y_label String überprüfen
        3. Dann Daten von allen TrainingRuns holen -> Mittelwert bilden und darstellen
        4. Für y_std Standardabweichung bilden und dann die Daten darstellen
        """
        pass

    #
    # def __init__(self, config, training_runs):
    #     self.config = config
    #     self.training_runs = training_runs
    #     self.num_training_runs = len(self.training_runs)
    #     self.mean_data = None
    #     self.std_data = None
    #
    #     self.runs_evaluated = True
    #     for run in self.training_runs:
    #         if run.no_evaluation:
    #             self.runs_evaluated = False
    #
    #     # Every run has already an evaluation, therefore initialize self.mean_data and self.std_data with it
    #     if self.runs_evaluated is True:
    #         self.evaluate()
    #
    # def evaluate(self, force=False, eval_count=5, skip=None, save=False, delete_models=False):
    #     data = []
    #     no_models = False
    #     if not self.runs_evaluated:
    #         for training_run in self.training_runs:
    #             no_models = training_run.no_models
    #             if no_models is True:
    #                 break
    #
    #     if no_models:
    #         print("The training runs do not provide model files, therefore the experiment cannot be evaluated." +
    #               "Please provide at least one .h5 file.")
    #     else:
    #         for training_run in self.training_runs:
    #             d = training_run.evaluate(force, eval_count, skip, save, delete_models)
    #             if d is None:
    #                 return
    #             data.append(d)
    #         concatenated = pd.concat([d for d in data])
    #         self.mean_data = concatenated.groupby(by='Generation', level=0).mean()
    #         self.std_data = concatenated.groupby(by='Generation', level=0).std()
    #
    # def visualize(self, force=False):
    #     for run in self.training_runs:
    #         self.video_file = run.visualize(force=force)
    #         if self.video_file is not None:
    #             break
    #     if self.video_file is None:
    #         print("The training runs do not provide model files, therefore the experiment cannot be visualized." +
    #               "Please provide at least one .h5 file so a video can be recorded.")
    #     return self.video_file
    #
    # def delete_model_files(self, save_last=False):
    #     for run in self.training_runs:
    #         run.delete_model_files(save_last)
    #
    # def get_num_training_runs(self):
    #     return self.num_training_runs
    #
    # def get_all_training_runs(self):
    #     return [run for run in self.training_runs]
    #
    # def get_all_logs(self):
    #     return [run.log for run in self.training_runs]
    #
    # def get_all_evaluations(self):
    #     return [run.evaluation for run in self.training_runs]
    #
    # def print_config(self):
    #     print(json.dumps(self.config, indent=4))
    #
    # def plot_reward_timestep(self):
    #     if self.mean_data is None:
    #         print("You did not evaluate the results. Please run evaluate() on this experiment. The plotted results"
    #               + " are used from the log file.")
    #         for run in self.training_runs:
    #             run.plot_reward_timestep()
    #     else:
    #         y_std = None
    #         # If we only have one training run the standard deviation will be NaN across all values and therefore
    #         # not be plotted. Use standard deviation from the only evaluation we have
    #         if self.num_training_runs > 1:
    #             y_std = self.std_data.Eval_Rew_Mean
    #         plot(self.mean_data.TimestepsSoFar, 'Timesteps',
    #              self.mean_data.Eval_Rew_Mean, 'Cummulative reward',
    #              y_std)
    #         print("Displayed is the mean reward of {} different runs over timesteps with different random seeds." +
    #               " If there was more than one run, the shaded region is the standard deviation of the mean reward.")
    #
    # def plot_reward_generation(self):
    #     if self.mean_data is None:
    #         print("You did not evaluate the results. Please run evaluate() on this experiment.")
    #     else:
    #         y_std = None
    #         # If we only have one training run the standard deviation will be NaN across all values and therefore
    #         # not be plotted. Use standard deviation from the only evaluation we have
    #         if self.num_training_runs > 1:
    #             y_std = self.std_data.Eval_Rew_Mean
    #         plot(self.mean_data.Generation, 'Generation',
    #              self.mean_data.Eval_Rew_Mean, 'Cummulative reward',
    #              y_std)
    #         print("Displayed is the mean reward of {} different runs over timesteps with different random seeds." +
    #               " If there was more than one run, the shaded region is the standard deviation of the mean reward.")
    #
    # def plot_timesteps_timeelapsed(self):
    #     if self.mean_data is None:
    #         print("You did not evaluate the results. Please run evaluate() on this experiment.")
    #     else:
    #         y_std = None
    #         # If we only have one training run the standard deviation will be NaN across all values and therefore
    #         # not be plotted. Use standard deviation from the only evaluation we have
    #         if self.num_training_runs > 1:
    #             y_std = self.std_data.TimestepsSoFar
    #         plot(self.mean_data.TimeElapsed, 'Time elapsed (s)',
    #              self.mean_data.TimestepsSoFar, 'Timesteps',
    #              y_std)
    #         print("Displayed is the mean reward of {} different runs over timesteps with different random seeds." +
    #               " If there was more than one run, the shaded region is the standard deviation of the mean reward.")