import keras
import numpy as np

class My_Callback(keras.callbacks.Callback):

    def __init__(self, max_norm):
        self.results = []
        self.max_norm = max_norm

    def set_validation(self, X_validation, Y_validation):
        self.X_validation = X_validation
        self.Y_validation = Y_validation

    def on_epoch_end(self, epoch, logs={}):

        validate_logs = self.model.evaluate(self.X_validation, self.Y_validation, verbose=1, sample_weight=None)

        full_log = {'train_acc': logs['acc'],
                    'train_loss': logs['loss'],
                    'valid_acc': validate_logs[1],
                    'valid_loss': validate_logs[0]}

        self.results.append(full_log)

        return

    def on_batch_end(self, batch, logs={}):

        for layer_name in ['convolution_3', 'convolution_4', 'convolution_5', 'dense_layer']:

            layer = self.model.get_layer(layer_name)

            weigths = np.array(layer.get_weights())
            original_shape = weigths[0].shape

            flatten_weights = weigths[0].flatten()
            norm_value = np.linalg.norm(flatten_weights)

            if (norm_value > self.max_norm):
                norm_flatten_weights = (flatten_weights / norm_value) * self.max_norm
                weigths[0] = np.reshape(norm_flatten_weights, original_shape)
                layer.set_weights(weigths)

        return
