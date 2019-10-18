import time

def create-model-once():

    import tensorflow as tf

    tf.keras.backend.clear_session()

    ob_stat = RunningStat(
        env.observation_space.shape,
        eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
    )

    times_load_weights, times_predict = [], []
    obs = np.ones(ob_space.shape)

    time_model_s = time.time()
    model = create_model(ob_mean=ob_stat.mean, ob_std=ob_stat.std)
    time_model_e = time.time() - time_model_s

    for i in range(1000):
        if i % 120 == 0:
            tf.keras.backend.clear_session()
            model = create_model(ob_mean=ob_stat.mean, ob_std=ob_stat.std)

        new_weights = np.random.randn(num_params)

        time_load_neww_s = time.time()
        set_from_flat(model, new_weights)
        times_load_weights.append(time.time() - time_load_neww_s)

        time_predict_s = time.time()
        a = model.predict(obs[None])
        times_predict.append(time.time() - time_predict_s)

    print("CreateModelTime", time_model_e)

    print("LoadWeightsMin", np.amin(times_load_weights))
    print("LoadWeightsMax", np.amax(times_load_weights))
    print("LoadWeightsMean", np.mean(times_load_weights))
    print("LoadWeightsCount", len(times_load_weights))

    print("PredictMin", np.amin(times_predict))
    print("PredictMax", np.amax(times_predict))
    print("PredictMean", np.mean(times_predict))
    print("PredictCount", len(times_predict))
    for j in range(len(times_predict)):
        print(j, times_predict[j])

def create_model_every_time():
    ob_stat = RunningStat(
        env.observation_space.shape,
        eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
    )

    times_create_model, times_predict = [], []
    obs = np.ones(ob_space.shape)
    for i in range(1000):
        time_model_s = time.time()
        model = create_model(ob_mean=ob_stat.mean, ob_std=ob_stat.std)
        times_create_model.append(time.time() - time_model_s)

        time_predict_s = time.time()
        a = model.predict(obs[None])
        times_predict.append(time.time() - time_predict_s)

    print("CreateModelMin", np.amin(times_create_model))
    print("CreateModelMax", np.amax(times_create_model))
    print("CreateModelMean", np.mean(times_create_model))
    print("CreateModelCount", len(times_create_model))

    print("PredictMin", np.amin(times_predict))
    print("PredictMax", np.amax(times_predict))
    print("PredictMean", np.mean(times_predict))
    print("PredictCount", len(times_predict))
    for j in times_predict:
        print(j)