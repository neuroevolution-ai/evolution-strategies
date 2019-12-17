import neat
import numpy as np
import typing as t

import keras.layers
import keras.models
import tensorflow as tf
from keras import backend as K

from utils.Profiling import profile_func


# maybe replace with np.indices()
def create_matrix(i_vals, j_vals):
    i_matrix = np.repeat(np.expand_dims(i_vals, 1), len(j_vals), axis=1)
    j_matrix = np.repeat(np.expand_dims(j_vals, 0), len(i_vals), axis=0)
    return np.stack([i_matrix, j_matrix], axis=-1)

#def grid_mapping_center(shape:t.Tuple[int, int], z_cord:float, scale=1.):
#    return grid_mapping_bounds(shape, z_cord, (-scale, scale), (-scale, scale))
#
#
#def grid_mapping_bounds( shape:t.Tuple[int, int], z_cord:float, range_x=(-1,1), range_y=(-1,1)):
#    mapping = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
#    x_cords = np.linspace(range_x[0], range_x[1], shape[0]) if shape[0] > 1 else [0]
#    y_cords = np.linspace(range_y[0], range_y[1], shape[1]) if shape[1] > 1 else [0]
#    mapping[:,:,:2] = create_matrix(x_cords, y_cords)
#    mapping[:,:, 2] = z_cord
#    return np.reshape(mapping, (shape[0]*shape[1], 3))


class GridMapping:
    def __init__(self, shape:t.Tuple[int, int]):
        assert (len(shape) == 2)
        self.shape = shape

    def flatten(self, input):
        assert(input.shape[0] == self.shape[0])
        assert(input.shape[1] == self.shape[1])
        return np.reshape(input, (len(self), *input.shape[2:]))
        
    def get_mapping(self): ...

    def __len__(self):
        return self.shape[0]*self.shape[1]

class GridMappingBounds(GridMapping):
    def __init__(self, shape: t.Tuple[int, int], z_cord: float, range_x=(-1, 1), range_y=(-1, 1)):
        super().__init__(shape)
        self.mapping = self.flatten(self.mapping2d(z_cord, range_x, range_y))
        
    def mapping2d(self, z_cord, range_x, range_y):
        mapping = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.float32)
        x_cords = np.linspace(range_x[0], range_x[1], self.shape[0]) if self.shape[0] > 1 else [0]
        y_cords = np.linspace(range_y[0], range_y[1], self.shape[1]) if self.shape[1] > 1 else [0]
        mapping[:,:,:2] = create_matrix(x_cords, y_cords)
        mapping[:,:, 2] = z_cord
        return mapping

    def get_mapping(self):
        return self.mapping


class DirectMapping:
    def __init__(self, xy_coords, z_coord):
        xy_coords = np.array(xy_coords)
        assert(xy_coords.shape[1] == 2)
        self.mapping = np.concatenate((xy_coords, np.full((xy_coords.shape[0],1), z_coord)), axis=1)

    def flatten(self, input):
        return input

    def get_mapping(self):
        return self.mapping

    def __len__(self):
        return len(self.mapping)



class Input():

    def __init__(self, mapping):
        self._mapping = mapping

    def flatten_input(self, input):
        return self._mapping.flatten(input)

    def get_neuron_coords(self):
        return self._mapping.get_mapping()

    def compile(self):
        self._keras_layer = keras.layers.Input((len(self._mapping),))

    def set_weights(self, weights, biases):
        pass

    def __call__(self):
        return self._keras_layer


class Dense:

    def __init__(self, mapping, activation:str='sigmoid'):
        self._mapping= mapping
        self._activation = activation

    def get_neuron_coords(self):
        return self._mapping.get_mapping()

    def compile(self):
        self._keras_layer = keras.layers.Dense(len(self._mapping), activation=self._activation)

    def set_weights(self, weights, biases):
        self._keras_layer.set_weights((weights, biases))

    def __call__(self, prev_layer):
        return self._keras_layer(prev_layer)





class Sequential:

    def __init__(self, max_abs_weight=10,min_abs_weight=0.1, bias=True):
        self.min_abs_weight = min_abs_weight
        self.max_abs_weight = max_abs_weight
        self.bias = bias
        self._input = None
        self._layers = []

        #self._session = tf.Session(self._graph)

    def input(self, input:Input):
        input.compile()
        self._input = input

    def add(self, layers:Dense):
        layers.compile()
        self._layers.append(layers)

    def compile(self):
        k_output = k_input = self._input()
        for layer in self._layers:
            k_output = layer(k_output)

        self._model = keras.models.Model(inputs=[k_input], outputs=[k_output])
        self._model._make_predict_function()


    def set_weights(self, cppn):
        all_layers = [self._input] + self._layers
        for i in range(1, len(all_layers)):
            from_neurons = all_layers[i-1].get_neuron_coords()
            to_neurons   = all_layers[i].get_neuron_coords()

            weight_coords = create_matrix(from_neurons, to_neurons)
            weight_coords = np.concatenate((weight_coords[:,:,:, 0], weight_coords[:,:,:,1]),axis=2)

            weights, biases = cppn.get_weights(weight_coords)
            weights = (weights - 0.5) * 6
            weights[weights < self.min_abs_weight] = 0
            #weights = np.clip(weights, -self.max_abs_weight, self.max_abs_weight)
            #biases = np.clip(biases, -self.max_abs_weight, self.max_abs_weight)
            if self.bias:
                biases = (biases - 0.5) * 6
            else:
                biases *= 0

            all_layers[i].set_weights(weights, biases)



    def __del__(self):
        pass


    @profile_func("substrat_forward")
    def activate(self, input):
        inputs = np.array([self._input.flatten_input(input)])
        return self._model.predict(inputs)




if __name__ == '__main__':
    sub = Sequential()
    sub.input(Input(GridMappingBounds((4, 1), 0)))
    sub.add(Dense(GridMappingBounds((3, 3), 1, range_x=(-0.5, 0.5), range_y=(-0.5, 0.5))))
    sub.add(Dense(DirectMapping([(0, 0)], 2)))
    sub.compile()

    class CPPA:
        def get_weights(self, cords):
            return np.sum(cords, axis=2), np.zeros(cords.shape[1])
    sub.set_weights(CPPA())