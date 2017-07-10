from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv2D
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input
from keras.models import Model

import random
import numpy as np

from tools import data_reader


def generate_example_io(count):
    xes = []
    ys = []

    for i in range(count):
        x = [random.random() for _ in range(2)]
        y = []
        weighted_sum = 0
        for j in range(len(x)):
            weighted_sum += x[j] * (j + 1)
        weighted_sum += 3.14159265
        y.append(1.234567890123456789 - x[0] - x[1])
        y.append(weighted_sum)
        xes.append(x)
        ys.append(y)

    return np.array(xes), np.array(ys)


def create_model(sequence_length, image_size, interface_vector_length, hidden_vector_length):
    if sequence_length == 1:
        inputs = Input(shape=(image_size, image_size, 3))
    else:
        inputs = Input(shape=(sequence_length, image_size, image_size, 3))

    # x = Conv2D(filters = 20,
    #            kernel_size = (3,3),
    #            data_format = "channels_last")(inputs)
    # x = Flatten()(inputs)
    x = TimeDistributed(Flatten())(inputs)
    # x = Dropout(0.2)(x)
    # x = Dense(units=512, activation="relu")(x)
    x = TimeDistributed(Dense(units=512, activation="relu"))(x)
    # frame_summary = Dense(hidden_vector_length, activation="linear")(x)
    x = TimeDistributed(Dense(interface_vector_length, activation="relu"))(x)

    x = LSTM(units=hidden_vector_length, dropout=0.0, recurrent_dropout=0.0, return_sequences=True)(x)
    answers = TimeDistributed(Dense(units=4, activation="linear"))(x)
    return Model(inputs=inputs, outputs=answers)


def main():
    sequence_length = 12
    image_size = 32     # Antar kvadrat og 3 kanaler
    interface_vector_length = 128
    hidden_vector_length = 256
    training_epochs = 10

    training_path = "/home/mathias/inndata/generert/tilfeldig bevegelse/train"
    testing_path = "/home/mathias/inndata/generert/tilfeldig bevegelse/test"
    example_path = testing_path

    training_examples = 1000
    testing_examples = 1000
    example_examples = 1


    # Bygge modellen
    model = create_model(sequence_length, image_size, interface_vector_length, hidden_vector_length)
    model.compile(optimizer="rmsprop",
                  loss="mean_squared_error")

    print("Oppsummering av det ferdige nettet:")
    model.summary()     # Skrive ut en oversikt over modellen
    print()

    # Trene modellen
    x_train, y_train = data_reader.fetch_x_y(training_path, training_examples, single_image=sequence_length == 1)
    model.fit(x_train, y_train, epochs=training_epochs)

    x_test, y_test = data_reader.fetch_x_y(testing_path, testing_examples, single_image=sequence_length == 1)
    evaluation = model.evaluate(x_test, y_test)
    print("Evaluering: ", evaluation)
    print()


    # Lage eksempler
    x_example, y_example = data_reader.fetch_x_y(example_path, example_examples, single_image=sequence_length == 1)

    prediction = model.predict(x_example)

    for i in range(len(x_example)):
        print("Eksempelsekvens:")
        for frame in range(sequence_length):
            print("Bilde:")
            correct_coords = y_example[i][frame]
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


main()