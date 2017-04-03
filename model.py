from keras.models import Model
from keras.layers import Input, Dense, MaxPooling1D, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import SGD

import keras.layers


def get_model(max_len_pargraph: int) -> Model:

    input_sequence = Input(shape=(max_len_pargraph, 300), dtype='float32')

    x = Conv1D(filters=100,
               kernel_size=3,
               activation='relu',
               name='convolution_3',
               input_shape=(max_len_pargraph, 300))(input_sequence)

    convolution3 = MaxPooling1D(max_len_pargraph - 3 + 1)(x)

    x = Conv1D(filters=100,
               kernel_size=4,
               activation='relu',
               name='convolution_4',
               input_shape=(max_len_pargraph, 300))(input_sequence)

    convolution4 = MaxPooling1D(max_len_pargraph - 4 + 1)(x)

    x = Conv1D(filters=100,
               kernel_size=5,
               activation='relu',
               name='convolution_5',
               input_shape=(max_len_pargraph, 300))(input_sequence)

    convolution5 = MaxPooling1D(max_len_pargraph - 5 + 1)(x)

    x = keras.layers.concatenate([convolution3, convolution4, convolution5])
    x = Dropout(.5)(x)
    x = Dense(3, name='dense_layer')(x)
    x = Flatten()(x)
    results = Activation('softmax')(x)

    model = Model(input_sequence, results)
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    model.summary()

    return model
