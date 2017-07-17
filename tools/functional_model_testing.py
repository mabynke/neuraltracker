import os

# import random
import numpy as np
from keras.callbacks import TerminateOnNaN, EarlyStopping, TensorBoard
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Reshape
from keras.layers.merge import Concatenate
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model

# import keras.backend.tensorflow_backend as K

# from tools import data_io  # For kjøring fra PyCharm
import data_io  # For kjøring fra terminal


# from keras import backend as K


def create_model(sequence_length, image_size, interface_vector_length, state_vector_length):
    # Ta innputt
    input_sequence = Input(shape=(sequence_length, image_size, image_size, 3), name="Innsekvens")
    input_coordinates = Input(shape=(4,), name="Innkoordinater")

    # Behandle bildesekvensen
    # x = TimeDistributed(Dense(units=512, activation="relu", name="Tett bildelag"))(x)
    # x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3)), name="Konv1")(input_sequence)
    # x = TimeDistributed(MaxPooling2D((2, 2)), name="maxpooling1")(x)  # 16*16
    # x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3)), name="Konv2")(x)
    # x = TimeDistributed(MaxPooling2D((2, 2)), name="maxpooling2")(x)  # 8*8
    # x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3)), name="Konv3")(x)

    x = TimeDistributed(Flatten(), name="Bildeutflating")(input_sequence)
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
                          tensorboard_log_dir, training_epochs, training_examples, training_path,
                          test_path, testing_examples, save_weights_path, round_patience=2):
    # Bygge modellen
    # K._set_session(K.tf.Session(config=K.tf.ConfigProto(log_device_placement=True)))
    model = create_model(sequence_length, image_size, interface_vector_length, state_vector_length)
    model.compile(optimizer="adam", loss="mean_squared_error")

    print("Oppsummering av nettet:")
    model.summary()  # Skrive ut en oversikt over modellen
    print()

    # Trene modellen
    train_seq, train_startcoords, train_labels = data_io.fetch_seq_startcoords_labels(training_path, training_examples)

    epoches_per_test = 9
    best_loss = 10000
    num_of_rounds_without_improvement = 0
    max_num_of_rounds = int(1032 / epoches_per_test)
    time_at_start = os.times().elapsed
    for i in range(max_num_of_rounds):
        model.fit(x=[train_seq, train_startcoords], y=train_labels, epochs=epoches_per_test,
              callbacks=[TerminateOnNaN(),# EarlyStopping(monitor="loss", patience=4),
                         TensorBoard(log_dir=tensorboard_log_dir)])
        evaluation = evaluate_model(model, test_path, testing_examples)

        print("Fullført runde {0}/{1} ({2} epoker). Brukt {3} minutter.".format(i+1, max_num_of_rounds,
                                                                               (i+1) * epoches_per_test,
                                                                               round((os.times().elapsed - time_at_start)/60, 1)))

        if evaluation >= best_loss:
            num_of_rounds_without_improvement += 1
            print("Runder uten forbedring: {0}/{1}".format(num_of_rounds_without_improvement, round_patience))
            if num_of_rounds_without_improvement >= round_patience:
                break
        else:
            num_of_rounds_without_improvement = 0
            best_loss = evaluation
            model.save_weights(save_weights_path, overwrite=True)
        print()
    return model


def print_results(example_labels, example_sequences, prediction, sequence_length, max_count):
    for i in range(len(example_sequences)):
        if i >= max_count:
            break

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
    training_epochs = 1000  # Stoppes når loss ikke lenger forbedres, av EarlyStopping
    training_examples = 0
    testing_examples = 0
    example_examples = 100

    sequence_length = 12
    image_size = 32  # Antar kvadrat og 3 kanaler
    interface_vector_length = 512
    state_vector_length = 512

    tensorboard_log_dir = "/tmp/logg/logg01"

    default_train_path = "/home/mby/Grafikk/Tilfeldig bevegelse/train"
    default_test_path = "/home/mby/Grafikk/Tilfeldig bevegelse/test"
    default_save_weight_path = "/home/mby/neuraltracker/tools/saved_weights/last_run.h5"
    train_path = data_io.get_path_from_user(default_train_path, "mappen med treningssekvenser")
    test_path = data_io.get_path_from_user(default_test_path, "mappen med testsekvenser")
    save_weights_path = data_io.get_path_from_user(default_save_weight_path,
                                                   "filen som vektene skal lagres til (overskriver)")
    example_path = test_path

    model = build_and_train_model(state_vector_length, image_size, interface_vector_length, sequence_length,
                                  tensorboard_log_dir, training_epochs, training_examples, train_path,
                                  test_path, testing_examples, save_weights_path, round_patience=3)
    # TODO: Lagre modellens konfigurasjon som json

    # evaluate_model(model, test_path, testing_examples)

    # Lage og vise eksempler
    example_sequences, example_startcoords, example_labels = data_io.fetch_seq_startcoords_labels(example_path,
                                                                                                  example_examples)
    predictions = model.predict([example_sequences, example_startcoords])
    predictions = np.delete(predictions, 0, 1)  # Fjerne det første ("falske") tidssteget
    for sequence in range(len(predictions)):
        path = os.path.join(example_path, "seq{0:05d}".format(sequence))
        data_io.write_labels(file_names=data_io.get_image_file_names_in_dir(path), labels=predictions[sequence], path=path, json_file_name="predictions.json")
    # print_results(example_labels, example_sequences, predictions, sequence_length, 4)


def evaluate_model(model, test_path, testing_examples):
    test_sequences, test_startcoords, test_labels = data_io.fetch_seq_startcoords_labels(test_path, testing_examples)
    evaluation = model.evaluate([test_sequences, test_startcoords], test_labels)
    print("\nEvaluering: ", evaluation, end="\n\n")
    return evaluation


main()