import tensorflow as tf
import numpy as np
import os

def set_from_flat(model, theta):
    print("Set from flat Start, PID:", os.getpid(), "Affinity:", os.sched_getaffinity(os.getpid()))

    old_theta = model.get_weights()
    print("Set from flat get weights, PID:", os.getpid(), "Affinity:", os.sched_getaffinity(os.getpid()))

    shapes = [v.shape for v in old_theta]
    total_size = theta.size

    start = 0
    reshapes = []
    print("Set from flat before for loop, PID:", os.getpid(), "Affinity:", os.sched_getaffinity(os.getpid()))

    for (shape, v) in zip(shapes, theta):
        size = int(np.prod(shape))
        reshapes.append(np.reshape(theta[start:start + size], shape))
        start += size
    print("Set from flat after for loop, PID:", os.getpid(), "Affinity:", os.sched_getaffinity(os.getpid()))

    assert start == total_size

    model.set_weights(reshapes)


def main():
    '''
    When the test fails there will be only one cpu core printed to stdout.

    Caused by the tensorflow and numpy packages used by conda. When installing in conda with -c intel, the intel packages
    get used which work.
    :return:
    '''
    input_layer = x = tf.keras.Input((None, 28), dtype=tf.float32)

    for hd in [256, 256]:
        x = tf.keras.layers.Dense(hd)(x)

    a = tf.keras.layers.Dense(8)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=a, name="2892")

    model.summary()

    set_from_flat(model, np.random.randn(75272))


if __name__ == '__main__':
    main()





