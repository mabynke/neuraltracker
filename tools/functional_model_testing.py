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
    x = Flatten()(inputs)
    # x = Dropout(0.2)(x)
    x = Dense(units=512, activation="relu")(x)
    frame_summary = Dense(hidden_vector_length, activation="linear")(x)

    sequence_after_frame_net = TimeDistributed(x)(input_sequence)
    position_sequence = LSTM(units=hidden_vector_length, dropout=0.0, recurrent_dropout=0.0)(sequence_after_frame_net)
    return Model(inputs=input_sequence, outputs=position_sequence)


def create_conv_net(image_size, output_values, sequence_length):
    """Enn så lenge ikke egentlig konvolusjonal"""


    return Model(inputs=inputs, outputs=predictions)


def main():
    sequence_length = 12
    image_size = 32     # Antar kvadrat og 3 kanaler
    interface_vector_length = 128
    hidden_vector_length = 256
    training_epochs = 10

    training_path = "/home/mathias/inndata/generert/tilfeldig bevegelse/train"
    testing_path = "/home/mathias/inndata/generert/tilfeldig bevegelse/test"
    example_path = testing_path

    training_examples = 0
    testing_examples = 0
    example_examples = 6


    # Bygge modellen
    model = create_model(sequence_length, image_size, interface_vector_length, hidden_vector_length)
    model.compile(optimizer="rmsprop",
                  loss="mean_squared_error")

    print("Oppsummering av det ferdige nettet:")
    model.summary()     # Skrive ut en oversikt over modellen
    print()
    exit()

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

    if sequence_length == 1:
        for i in range(len(x_example)):
            print("{0:4.1f}, {1:4.1f} \t\t{2:4.1f}, {3:4.1f}".format(y_example[i][0], y_example[i][1],
                                               prediction[i][0], prediction[i][1]))
    else:
        print("Sekvenser er ikke ferdig implementert")
        raise(NotImplementedError)


main()