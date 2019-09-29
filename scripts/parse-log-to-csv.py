import csv

def parse_log_to_csv(log_file, csv_file):
    with open(log_file) as f:
        content = f.readlines()

    groups = temp = []
    for line in content:
        line = line.split()

        if not line:
            continue

        if "Generation" in line:
            temp = [line[-1]]
            groups.append(temp)
        else:
            temp.append(line[-1])

    writer = csv.writer(open(csv_file, 'w'))

    writer.writerow(['Generation',
                     'Rew_Mean',
                     'Rew_Std',
                     'Len_Mean',
                     'Eval_Rew_Mean',
                     'Eval_Rew_Std',
                     'Eval_Len_Mean',
                     'Eval_Count',
                     'Episodes_this_Gen',
                     'Episodes_overall',
                     'Timesteps_this_gen',
                     'Timesteps_overall',
                     'Unique_Workers',
                     'ResultsSkippedFrac',
                     'Observation_count',
                     'Time_elapsed_this_Gen',
                     'Time_elapsed_overall',
                     'TimePerMutationMin',
                     'TimePerMutationMax',
                     'TimePerMutationMean',
                     'TimePerMutationCount',
                     'TimeCreateModelMin',
                     'TimeCreateModelMax',
                     'TimeCreateModelMean',
                     'TimeCreateModelCount',
                     'TimeSetFlatMin',
                     'TimeSetFlatMax',
                     'TimeSetFlatMean',
                     'TimeSetFlatCount',
                     'TimeSampleMin',
                     'TimeSampleMax',
                     'TimeSampleMean',
                     'TimeSampleCount',
                     'TimeGetNoiseMin',
                     'TimeGetNoiseMax',
                     'TimeGetNoiseMean',
                     'TimeGetNoiseCount',
                     'TimePredictMin',
                     'TimePredictMax',
                     'TimePredictMean',
                     'TimePredictCount',
                     'TimeClearSessMin',
                     'TimeClearSessMax',
                     'TimeClearSessMean',
                     'TimeClearSessCount',
                     'TimeSetSessMin',
                     'TimeSetSessMax',
                     'TimeSetSessMean',
                     'TimeSetSessCount'])

    for generation in groups:
        if len(generation) != 51: continue

        # Throw out save_directory and distinction line
        generation = generation[:49]
        row = []

        for column in generation:
            row.append(column)

        writer.writerow(row)