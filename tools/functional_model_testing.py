from keras.layers.core import Dense, Dropout, Flatten, Reshape, Masking
from keras.layers.recurrent import LSTM
# from keras.layers.convolutional import Conv2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Concatenate
from keras.layers import Input
from keras.callbacks import TerminateOnNaN, EarlyStopping, TensorBoard
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
# from keras import backend as K

import os

# import random
import numpy as np

from tools import data_io


def create_model(sequence_length, image_size, interface_vector_length, state_vector_length):
    # Ta innputt
    input_sequence = Input(shape=(sequence_length, image_size, image_size, 3), name="Innsekvens")
    input_coordinates = Input(shape=(4,), name="Innkoordinater")

    # Behandle bildesekvensen
    x = TimeDistributed(Flatten(), name="Bildeutflating")(input_sequence)
    x = TimeDistributed(Dense(units=512, activation="relu", name="Tett bildelag"))(x)
    x = TimeDistributed(Dense(interface_vector_length, activation="relu", name="Grensesnittvektorer"))(x)

    # Kode og sette inn informasjon om startkoordinater
    k = Dense(units=interface_vector_length, activation="relu", name="Kodede_koordinater")(input_coordinates)
    k = Reshape(target_shape=(1, interface_vector_length), name="Omforming")(k)
    x = Concatenate(axis=1, name="Sammensetting")([k, x])

    # Behandle sekvens av grensesnittvektorer med LSTM
    x = LSTM(units=state_vector_length, dropout=0.0, recurrent_dropout=0.0, return_sequences=True, name="LSTM-lag")(x)

    # # Fjerne det første ekstra tidssteget
    # x = K.gather(x, [None, 1])

    # Dekode til koordinater
    answers = TimeDistributed(Dense(units=4, activation="linear", name="Koordinater ut"))(x)

    return Model(inputs=[input_sequence, input_coordinates], outputs=answers)


def build_and_train_model(state_vector_length, image_size, interface_vector_length, sequence_length,
                          tensorboard_log_dir, training_epochs, training_examples, training_path):
    # Bygge modellen
    model = create_model(sequence_length, image_size, interface_vector_length, state_vector_length)
    model.compile(optimizer="rmsprop", loss="mean_squared_error")

    print("Oppsummering av nettet:")
    model.summary()  # Skrive ut en oversikt over modellen
    print()

    # Trene modellen
    train_seq, train_startcoords, train_labels = data_io.fetch_seq_startcoords_labels(training_path, training_examples)
    model.fit(x=[train_seq, train_startcoords], y=train_labels, epochs=training_epochs,
              callbacks=[TerminateOnNaN(), EarlyStopping(monitor="loss", patience=5),
                         TensorBoard(log_dir=tensorboard_log_dir)])
    return model


def print_results(example_labels, example_sequences, prediction, sequence_length):
    for i in range(len(example_sequences)):
        print("Eksempelsekvens:")
        for frame in range(sequence_length):
            print("Bilde:")
            correct_coords = example_labels[i][frame]
            calculated_coords = prediction[i][frame]

            squared_error = 0
            for coordinate in range(len(correct_coords)):
                squared_error += (correct_coords[coordinate] - calculated_coords[coordinate]) ** 2
            mean_squared_error = squared_error / len(correct_coords)

            print("{0:4.0f}, {1:4.0f} \t{2:4.0f}, {3:4.0f}".format(correct_coords[0], correct_coords[1],
                                                                   correct_coords[2], correct_coords[3]))
            print("{0:4.0f}, {1:4.0f} \t{2:4.0f}, {3:4.0f}".format(calculated_coords[0], calculated_coords[1],
                                                                   calculated_coords[2], calculated_coords[3]), end="")
            print("\tLoss: {0:5.2f}".format(mean_squared_error))


def main():
    sequence_length = 12
    image_size = 32  # Antar kvadrat og 3 kanaler
    interface_vector_length = 128
    state_vector_length = 256
    training_epochs = 1000

    tensorboard_log_dir = "/tmp/logg/logg3"

    train_path = "/home/mathias/inndata/generert/tilfeldig bevegelse/train"
    test_path = "/home/mathias/inndata/generert/tilfeldig bevegelse/test"
    example_path = train_path

    training_examples = 10
    testing_examples = 1000
    example_examples = 10

    model = build_and_train_model(state_vector_length, image_size, interface_vector_length, sequence_length,
                                  tensorboard_log_dir, training_epochs, training_examples, train_path)

    test_sequences, test_startcoords, test_labels = data_io.fetch_seq_startcoords_labels(test_path, testing_examples)
    evaluation = model.evaluate([test_sequences, test_startcoords], test_labels)
    print("Evaluering: ", evaluation, end="\n\n")

    # Lage og vise eksempler
    example_sequences, example_startcoords, example_labels = data_io.fetch_seq_startcoords_labels(example_path,
                                                                                                  example_examples)
    predictions = model.predict([example_sequences, example_startcoords])
    predictions = np.delete(predictions, 0, 1)  # Fjerne det første ("falske") tidssteget
    for sequence in range(len(predictions)):
        path = os.path.join(example_path, "seq{0:05d}".format(sequence))
        data_io.write_labels(file_names=data_io.get_image_file_names_in_dir(path), labels=predictions[sequence], path=path, json_file_name="predictions.json")
    print_results(example_labels, example_sequences, predictions, sequence_length)


main()