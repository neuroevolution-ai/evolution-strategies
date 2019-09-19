import click
import roboschool, gym
import numpy as np
import os
import csv

from gym import wrappers
from itertools import zip_longest
from multiprocessing import Pool
from policies import MujocoPolicy

def run_policy(env_id, policy_file, record, stochastic):
    import tensorflow as tf
    env = gym.make(env_id)
    if record:
        import uuid
        env = wrappers.Monitor(env, '/tmp/' + str(uuid.uuid4()), force=True)

    with tf.Session():
        pi = MujocoPolicy.Load(policy_file)
        # while True:
        rews, t = pi.rollout(env, render=False, random_stream=np.random if stochastic else None)
        print('return={:.4f} len={}'.format(rews.sum(), t))

        rews = np.array(rews, dtype=np.float32)
        if record:
            env.close()
            return [rews.sum(), t]


    tf.reset_default_graph()
    env.close()

    return [rews.sum(), t]


def index_save_directory(save_directory):
    if not os.path.isdir(save_directory):
        return None

    model_file_paths, log_file_path = [], None

    for file in os.listdir(save_directory):
        if file.endswith('.h5'):
            model_file_paths.append(file)
        elif file.endswith('log.txt'):
            log_file_path = save_directory + file

    model_file_paths.sort()

    return model_file_paths, log_file_path

def parse_generation_number(model_file_path):
    try:
        number = int(model_file_path.split('snapshot_iter')[-1].split('_rew')[0])
        return number
    except ValueError:
        return None

def evaluate_to_csv(save_directory, model_file_paths, env_id, csv_eval_file_path, eval_count=5):
    writer = csv.writer(open(csv_eval_file_path, 'w'))

    head_row = ['Generation',
                'Eval_per_Gen',
                'Eval_Rew_Mean',
                'Eval_Rew_Std',
                'Eval_Len_Mean']

    for i in range(eval_count):
        head_row.append('Rew_' + str(i))
        head_row.append('Len_' + str(i))

    writer.writerow(head_row)

    rows = []
    rows.append(head_row)
    for model_file_path in model_file_paths:
        results = []
        with Pool(os.cpu_count()) as pool:
            for _ in range(eval_count):
                results.append(pool.apply_async(func=run_policy, args=(env_id,
                                                                       save_directory + model_file_path,
                                                                       False,
                                                                       False)))

            for i in range(len(results)):
                results[i] = results[i].get()

        rewards = np.array(results)[:, 0]
        lengths = np.array(results)[:, 1]

        row = [parse_generation_number(model_file_path),
               eval_count,
               np.mean(rewards),
               np.std(rewards),
               np.mean(lengths)]

        assert len(rewards) == len(lengths)
        for i in range(len(rewards)):
            row.append(rewards[i])
            row.append(lengths[i])

        writer.writerow(row)
        rows.append(row)

    return rows

def parse_log_to_csv(log_file, csv_file):
    with open(log_file) as f:
        content = f.readlines()

    groups = temp = []
    i = 0
    for line in content:
        line = line.split()

        if not line:
            continue

        if '**********' in line:
            temp = [line[2]]
            groups.append(temp)
        elif '----------------------------------' in line or 'snapshot' in line:
            continue

        else:
            temp.append(line[3])

    writer = csv.writer(open(csv_file, 'w'))

    head_row = ['Generation',
                 'EpRewMean',
                 'EpRewStd',
                 'EpLenMean',
                 'EvalEpRewMean',
                 'EvalEpRewStd',
                 'EvalEpLenMean',
                 'EvalPopRank',
                 'EvalEpCount',
                 'Norm',
                 'GradNorm',
                 'UpdateRatio',
                 'EpisodesThisIter',
                 'EpisodesSoFar',
                 'TimestepsThisIter',
                 'TimestepsSoFar',
                 'UniqueWorkers',
                 'UniqueWorkersFrac',
                 'ResultsSkippedFrac',
                 'ObCount',
                 'TimeElapsedThisIter',
                 'TimeElapsed']

    writer.writerow(head_row)

    rows = []
    rows.append(head_row)
    for generation in groups:
        if len(generation) != 22: continue

        # Throw out save_directory and distinction line
        row = []

        for column in generation:
            row.append(column)

        writer.writerow(row)
        rows.append(row)

    return rows
@click.command()
@click.argument('env_id')
@click.argument('policies_path')
@click.option('--record', is_flag=True)
@click.option('--stochastic', is_flag=True)
def main(env_id, policies_path, record, stochastic):
    model_file_paths, log_file_path = index_save_directory(policies_path)

    log_rows = parse_log_to_csv(log_file_path, policies_path + 'log.csv')
    eval_rows = evaluate_to_csv(policies_path, model_file_paths, env_id, policies_path + 'evaluate.csv')

    # 
    # merged_rows = []
    # merged_rows.append(eval_rows[0] + ['TimeElapsedThisIter', 'TimeElapsed', 'TimestepsThisIter', 'TimestepsSoFar', 'EpisodesThisIter', 'EpisodesSoFar'])
    #
    # for i in range(len(eval_rows)):
    #     merged_row = eval_rows[i]
    #
    #     gen = int(merged_row[0])
    #
    #     merged_rows.append(eval_row)
    # writer = csv.writer(open(policies_path + 'evaluate.csv', 'w'))
    # for row in merged_rows:
    #     writer.writerow(row)

if __name__ == '__main__':
    main()