from tensorflow.keras.layers import Layer, Dense, Input, LSTM, Dense, Dropout, Multiply, Reshape, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

class WeightPredictor(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim  # This should match the feature dimension
        super(WeightPredictor, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weight_predictor = Dense(self.output_dim, activation='tanh')
        self.sparsity_network = Dense(self.output_dim, activation='sigmoid')
        super(WeightPredictor, self).build(input_shape)

    def call(self, inputs):
        inputs_reduced = K.mean(inputs, axis=1)
        predicted_weights = self.weight_predictor(inputs_reduced)
        sparse_weights = self.sparsity_network(predicted_weights)
        expanded_weights = K.expand_dims(sparse_weights, axis=1)
        repeated_weights = K.repeat_elements(expanded_weights, rep=inputs.shape[1], axis=1)
        return repeated_weights
