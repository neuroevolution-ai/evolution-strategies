def load_model(model_path):   
    import tensorflow as tf

    class Normc_initializer(tf.keras.initializers.Initializer):
        def __init__(self, std=1.0):
            self.std=std

        def __call__(self, shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= self.std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
    
    class ObservationNormalizationLayer(tf.keras.layers.Layer):
        def __init__(self, ob_mean, ob_std, **kwargs):
            self.ob_mean = ob_mean
            self.ob_std = ob_std
            super(ObservationNormalizationLayer, self).__init__(**kwargs)

        def call(self, x):
            return tf.clip_by_value((x - self.ob_mean) / self.ob_std, -5.0, 5.0)
        
        # get_config and from_config need to implemented to be able to serialize the model
        def get_config(self):
            base_config = super(ObservationNormalizationLayer, self).get_config()
            base_config['ob_mean'] = self.ob_mean
            base_config['ob_std'] = self.ob_std
            return base_config
        
        @classmethod
        def from_config(cls, config):
            return cls(**config)
        
    class DiscretizeActionsUniformLayer(tf.keras.layers.Layer):
        def __init__(self, num_ac_bins, adim, ahigh, alow, **kwargs):
            self.num_ac_bins = num_ac_bins
            self.adim = adim
            # ahigh, alow are NumPy arrays when extracting from the environment, but when the model is loaded from a h5
            # File they get initialised as a normal list, where operations like subtraction does not work, thereforce
            # cast them explicitly
            self.ahigh = np.array(ahigh)
            self.alow = np.array(alow)
            super(DiscretizeActionsUniformLayer, self).__init__(**kwargs)

        def call(self, x):            
            # Reshape to [n x i x j] where n is dynamically chosen, i equals action dimension and j equals the number
            # of bins
            scores_nab = tf.reshape(x, [-1, self.adim, self.num_ac_bins])
            # This picks the bin with the greatest value
            a = tf.argmax(scores_nab, 2)
            
            # Then transform the interval from [0, num_ac_bins - 1] to [-1, 1] which equals alow and ahigh
            ac_range_1a = (self.ahigh - self.alow)[None, :]
            return 1. / (self.num_ac_bins - 1.) * tf.keras.backend.cast(a, 'float32') * ac_range_1a + self.alow[None, :]        
        
        # get_config and from_config need to implemented to be able to serialize the model
        def get_config(self):
            base_config = super(DiscretizeActionsUniformLayer, self).get_config()
            base_config['num_ac_bins'] = self.num_ac_bins
            base_config['adim'] = self.adim
            base_config['ahigh'] = self.ahigh
            base_config['alow'] = self.alow
            return base_config
        
        @classmethod
        def from_config(cls, config):
            return cls(**config)
    
    custom_objects = {'Normc_initializer' : Normc_initializer, 
                      'ObservationNormalizationLayer' : ObservationNormalizationLayer,
                      'DiscretizeActionsUniformLayer' : DiscretizeActionsUniformLayer}
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except OSError as e:
        print(e)
        return None
    return model