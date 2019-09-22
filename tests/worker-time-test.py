def run_worker(num_jobs, theta, ob_mean=None, ob_std=None, generations=None):
    print("PID " + str(os.getpid()) + ": " + "Started worker with " + str(num_jobs) + "Jobs")

    assert isinstance(noise, SharedNoiseTable)
    assert num_jobs > 0

    # Setup
    # Create a new gym environment object because each worker needs its own one
    env = gym.make(config.env_id)

    # Initialize the model with the supplied weights 'theta' to calcualate based on the current generation
    model = create_model(initial_weights=theta, model_name=str(os.getpid()), ob_mean=ob_mean, ob_std=ob_std)

    # Random stream used for adding noise to the actions as well as deciding if the observation statistics shall be
    # updated
    rs = np.random.RandomState()

    task_ob_stat = RunningStat(env.observation_space.shape, eps=0.)  # eps=0 because we're incrementing only

    noise_inds, returns, signreturns, lengths, eval_returns, eval_lengths = [], [], [], [], [], []

    time_samples, time_get_noises, time_set_flat_pos, time_set_flat_neg, time_rollout_pos, time_rollout_neg = [], [], [], [], [], []
    for _ in range(num_jobs):

        if rs.rand() < config.eval_prob:
            # Evaluation sample
            set_from_flat(model, theta)
            eval_rews, eval_length = rollout(env, model)
            eval_returns.append(eval_rews.sum())
            eval_lengths.append(eval_length)

        # Rollouts with noise

        # Noise sample
        time_s_sample_noise = time.time()
        noise_idx = noise.sample_index(rs, num_params)
        time_e_sample_noise = time.time()

        time_s_get_noise = time.time()
        _temp = noise.get(noise_idx, num_params)
        time_e_get_noise = time.time()

        epsilon = config.noise_stdev * _temp

        # Evaluate the sampled noise

        time_s_set_flat_pos = time.time()
        set_from_flat(model, theta + epsilon)
        time_e_set_flat_pos = time.time()

        time_s_rollout_pos = time.time()
        rews_pos, len_pos = rollout_and_update_ob_stat(env, model, rs=rs, task_ob_stat=task_ob_stat)
        time_e_rollout_pos = time.time()

        # Gather results
        noise_inds.append(noise_idx)
        returns.append([rews_pos.sum()])
        signreturns.append([np.sign(rews_pos).sum()])
        lengths.append([len_pos])

        # Mirrored sampling also evaluates the noise by subtracting it
        if optimizations.mirrored_sampling:
            time_s_set_flat_neg = time.time()
            set_from_flat(model, theta - epsilon)
            time_e_set_flat_neg = time.time()

            time_s_rollout_neg = time.time()
            rews_neg, len_neg = rollout_and_update_ob_stat(env, model, rs=rs, task_ob_stat=task_ob_stat)
            time_e_rollout_neg = time.time()

            returns[-1].append(rews_neg.sum())
            signreturns[-1].append(np.sign(rews_neg).sum())
            lengths[-1].append(len_neg)

            time_set_flat_neg.append(time_e_set_flat_neg - time_s_set_flat_neg)
            time_rollout_neg.append(time_e_rollout_neg - time_s_rollout_neg)

        time_samples.append(time_e_sample_noise - time_s_sample_noise)
        time_get_noises.append(time_e_get_noise - time_s_get_noise)
        time_set_flat_pos.append(time_e_set_flat_pos - time_s_set_flat_pos)
        time_rollout_pos.append(time_e_rollout_pos - time_s_rollout_pos)

    times = [
        time_samples,
        time_get_noises,
        time_set_flat_pos,
        time_rollout_pos
    ]

    if optimizations.mirrored_sampling:
        times.append(time_set_flat_neg)
        times.append(time_rollout_neg)

    with open(save_directory + '/' + str(generations) + '_' + str(os.getpid()) + ".txt", "w") as output:
        output.write('Sample times ' + str(times[0]) + '\n')
        output.write('Get Noises times ' + str(times[1]) + '\n')
        output.write('Set flat pos times ' + str(times[2]) + '\n')
        output.write('Rollout pos times ' + str(times[3]) + '\n')
        if len(times) > 4:
            output.write('Set flat neg times ' + str(times[4]) + '\n')
            output.write('Rollout neg times ' + str(times[5]) + '\n')

    result = Result(
        noise_inds=np.array(noise_inds),
        returns=np.array(returns, dtype=np.float32),
        signreturns=np.array(signreturns, dtype=np.float32),
        lengths=np.array(lengths, dtype=np.int32),
        eval_returns=None if not eval_returns else np.array(eval_returns, dtype=np.float32),
        eval_lengths=None if not eval_lengths else np.array(eval_lengths, dtype=np.int32),
        ob_sum=None if task_ob_stat.count == 0 else task_ob_stat.sum,
        ob_sumsq=None if task_ob_stat.count == 0 else task_ob_stat.sumsq,
        ob_count=task_ob_stat.count
    )

    print("PID " + str(os.getpid()) + ": " + "Returned result")

    return result


'''
Tracks time per worker
Result object needs a time object. In master log then the time
'''


def run_worker(num_jobs, theta, ob_mean=None, ob_std=None):
    time_worker_start = time.time()
    print("PID " + str(os.getpid()) + ": " + "Started worker with " + str(num_jobs) + "Jobs")

    assert isinstance(noise, SharedNoiseTable)
    assert num_jobs > 0

    # Setup
    # Create a new gym environment object because each worker needs its own one
    env = gym.make(config.env_id)

    # Initialize the model with the supplied weights 'theta' to calcualate based on the current generation
    model = create_model(initial_weights=theta, model_name=str(os.getpid()), ob_mean=ob_mean, ob_std=ob_std)

    # Random stream used for adding noise to the actions as well as deciding if the observation statistics shall be
    # updated
    rs = np.random.RandomState()

    task_ob_stat = RunningStat(env.observation_space.shape, eps=0.)  # eps=0 because we're incrementing only

    noise_inds, returns, signreturns, lengths, eval_returns, eval_lengths = [], [], [], [], [], []

    for _ in range(num_jobs):

        if rs.rand() < config.eval_prob:
            # Evaluation sample
            set_from_flat(model, theta)
            eval_rews, eval_length = rollout(env, model)
            eval_returns.append(eval_rews.sum())
            eval_lengths.append(eval_length)

        # Rollouts with noise

        # Noise sample
        noise_idx = noise.sample_index(rs, num_params)
        epsilon = config.noise_stdev * noise.get(noise_idx, num_params)

        # Evaluate the sampled noise
        set_from_flat(model, theta + epsilon)
        rews_pos, len_pos = rollout_and_update_ob_stat(env, model, rs=rs, task_ob_stat=task_ob_stat)

        # Gather results
        noise_inds.append(noise_idx)
        returns.append([rews_pos.sum()])
        signreturns.append([np.sign(rews_pos).sum()])
        lengths.append([len_pos])

        # Mirrored sampling also evaluates the noise by subtracting it
        if optimizations.mirrored_sampling:
            set_from_flat(model, theta - epsilon)
            rews_neg, len_neg = rollout_and_update_ob_stat(env, model, rs=rs, task_ob_stat=task_ob_stat)

            returns[-1].append(rews_neg.sum())
            signreturns[-1].append(np.sign(rews_neg).sum())
            lengths[-1].append(len_neg)

    time_worker_end = time.time()

    result = Result(
        noise_inds=np.array(noise_inds),
        returns=np.array(returns, dtype=np.float32),
        signreturns=np.array(signreturns, dtype=np.float32),
        lengths=np.array(lengths, dtype=np.int32),
        eval_returns=None if not eval_returns else np.array(eval_returns, dtype=np.float32),
        eval_lengths=None if not eval_lengths else np.array(eval_lengths, dtype=np.int32),
        ob_sum=None if task_ob_stat.count == 0 else task_ob_stat.sum,
        ob_sumsq=None if task_ob_stat.count == 0 else task_ob_stat.sumsq,
        ob_count=task_ob_stat.count,
        time=time_worker_end - time_worker_start
    )

    print("PID " + str(os.getpid()) + ": " + "Returned result")

    return result