from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, Reshape
from keras.models import Model, Sequential

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


def create_model(sequence_length, image_size, output_values):
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
    predictions = Dense(output_values, activation="linear")(x)

    return Model(inputs=inputs, outputs=predictions)


def main():
    sequence_length = 1
    image_size = 32     # Antar kvadrat og 3 kanaler
    output_values = 4
    training_epochs = 10

    training_path = "/home/mathias/inndata/generert/tilfeldig bevegelse/train"
    testing_path = "/home/mathias/inndata/generert/tilfeldig bevegelse/test"
    example_path = testing_path

    training_examples = 0
    testing_examples = 0
    example_examples = 6


    # Bygge modellen
    model = create_model(sequence_length, image_size, output_values)
    model.compile(optimizer="rmsprop",
                  loss="mean_squared_error")

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

    if sequence_length == 1:
        for i in range(len(x_example)):
            print("{0:4.1f}, {1:4.1f} \t\t{2:4.1f}, {3:4.1f}".format(y_example[i][0], y_example[i][1],
                                               prediction[i][0], prediction[i][1]))
    else:
        print("Sekvenser er ikke ferdig implementert")
        raise(NotImplementedError)


main()