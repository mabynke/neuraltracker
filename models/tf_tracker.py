import os
import sys

import tensorflow as tf
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from keras.engine.topology import Layer

# TODO: change padding from 'valid' to 'same'
# TODO: not hand-code sequence_length, may also change to None?
sequence_length = 12

class GRULayer(Layer):

    def __init__(self, state_size, **kwargs):
        self.state_size = state_size
        super(GRULayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.gru = tf.contrib.rnn.GRUCell(self.state_size)
        self._trainable_weights = self.gru._trainable_weights
        # TODO: add trainable weights
        super(GRULayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        x, coded_cords = inputs
        self.state = coded_cords
        outputs = []
        for frame in range(sequence_length):
            # assume x has shape [batch_size, sequence_length, interface_vector_length]
            output, self.state = self.gru(x[:, frame, :], self.state)
            outputs.append(output)
        x = tf.stack(outputs, axis=1)
        return x

    def compute_output_shape(self, input_shape):
        data_input = input_shape[0]
        output_shape = list(data_input)
        output_shape[-1] = self.state_size
        return tuple(output_shape)




def create_model(image_size, interface_vector_length, state_vector_length):
    # Ta innputt
    input_sequence = Input(shape=(None, image_size, image_size, 3), name="Innsekvens")
    input_coordinates = Input(shape=(4,), name="Innkoordinater")

    # Behandle bildesekvensen
    x = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding="valid"), name="Konv1")(input_sequence)
    x = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding="valid"), name="Konv2")(x)
    x = TimeDistributed(MaxPooling2D((2, 2)), name="maxpooling1")(x)
    x = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding="valid"), name="Konv3")(x)
    x = TimeDistributed(MaxPooling2D((2, 2)), name="maxpooling2")(x)

    x = TimeDistributed(Flatten(), name="Bildeutflating")(x)
    x = TimeDistributed(Dense(interface_vector_length, activation="relu"), name="Grensesnittvektorer")(x)
    # Kode og sette inn informasjon om startkoordinater
    coords = Dense(units=state_vector_length, activation="relu", name="Kodede_koordinater")(input_coordinates)

    # Behandle sekvens av grensesnittvektorer med LSTM
    x = GRU(units=state_vector_length,
            dropout=0.5,
            recurrent_dropout=0.0,
            return_sequences=True,
            name="GRU-lag1")(x, initial_state=coords)

    # Dekode til koordinater
    position = TimeDistributed(Dense(units=2, activation="linear"), name="Posisjon_ut")(x)
    size = TimeDistributed(Dense(units=2, activation="sigmoid"), name="Stoerrelse_ut")(x)

    return Model(inputs=[input_sequence, input_coordinates], outputs=[position, size])


def main():
    os.chdir(os.path.dirname(sys.argv[0])) # set working directory to that of the script
    sess = tf.Session()
    image_size = 32
    interface_vector_length = 512
    state_vector_length = 512
    log_dir = "tensorboard_out"
    model = create_model(image_size, interface_vector_length, state_vector_length)
    model.compile(optimizer=Adam(), loss=["mean_squared_error", "mean_squared_error"], loss_weights=[1, 1])
    #fit_history = model.fit(x=None, y=None, epochs=1,
    #                        callbacks=[TensorBoard(log_dir=log_dir)],
    #                        verbose=2)
    summary_writer = tf.summary.FileWriter(
        log_dir,
        graph=sess.graph)


if __name__ == "__main__":
    main()
