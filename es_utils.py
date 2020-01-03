import json
import os

from pathlib import Path

class InvalidTrainingError(Exception):
    pass

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


def validate_config(config_file):
    assert isinstance(config_file, os.DirEntry)
    with open(config_file.path, encoding='utf-8') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            raise InvalidTrainingError("The config file {} cannot be parsed.".format(config_file.path))

    # validate every part of config
    # return optimizations, model_structure, config object




class TrainingRun:
    pass