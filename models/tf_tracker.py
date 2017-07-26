import os
import sys
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf


def create_model(image_size, interface_vector_length, state_vector_length):
    # Ta innputt
    input_sequence = Input(shape=(None, image_size, image_size, 3), name="Innsekvens")
    input_coordinates = Input(shape=(4,), name="Innkoordinater")

    # Behandle bildesekvensen
    x = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"), name="Konv1")(input_sequence)
    x = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"), name="Konv2")(x)
    x = TimeDistributed(MaxPooling2D((2, 2)), name="maxpooling1")(x)
    x = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"), name="Konv3")(x)
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
