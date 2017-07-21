import os
import sys

# from tools import data_io  # For kjøring fra PyCharm
import data_io  # For kjøring fra terminal
# import random
import numpy as np
from keras.callbacks import TerminateOnNaN, TensorBoard
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Reshape
from keras.layers.merge import Concatenate
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model

# from keras import backend as K
import plotting
import time


RUN_ID = 0


# import keras.backend.tensorflow_backend as K


def create_model(sequence_length, image_size, interface_vector_length, state_vector_length):
    # Ta innputt
    input_sequence = Input(shape=(sequence_length, image_size, image_size, 3), name="Innsekvens")
    input_coordinates = Input(shape=(4,), name="Innkoordinater")

    # Behandle bildesekvensen
    x = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"), name="Konv1")(input_sequence)
    x = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"), name="Konv2")(x)
    x = TimeDistributed(MaxPooling2D((2, 2)), name="maxpooling1")(x)
    if RUN_ID == 2:
        x = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"), name="Konv2")(x)
        x = TimeDistributed(MaxPooling2D((2, 2)), name="maxpooling2")(x)
    # x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3)), name="Konv3")(x)

    x = TimeDistributed(Flatten(), name="Bildeutflating")(x)
    if RUN_ID != 0:
        x = TimeDistributed(Dense(interface_vector_length, activation="relu"), name="Grensesnittvektorer")(x)
    else:
        interface_vector_length = 4608

    # Kode og sette inn informasjon om startkoordinater
    # k = Concatenate(name="pos_str_sammensetting")([input_position, input_size])
    k = Dense(units=interface_vector_length, activation="relu", name="Kodede_koordinater")(input_coordinates)
    k = Reshape(target_shape=(1, interface_vector_length), name="Omforming")(k)
    x = Concatenate(axis=1, name="Sammensetting")([k, x])

    # Behandle sekvens av grensesnittvektorer med LSTM
    if RUN_ID == 1:
        dropout = 0.5
    else:
        dropout = 0.0
    x = GRU(units=state_vector_length, dropout=dropout, recurrent_dropout=0.0, return_sequences=True, name="GRU-lag1")(x)
    # x = GRU(units=state_vector_length, dropout=0.0, recurrent_dropout=0.0, return_sequences=True, name="GRU-lag2")(x)

    # Dekode til koordinater
    position = TimeDistributed(Dense(units=2, activation="linear"), name="Posisjon_ut")(x)
    size = TimeDistributed(Dense(units=2, activation="sigmoid"), name="Stoerrelse_ut")(x)

    return Model(inputs=[input_sequence, input_coordinates], outputs=[position, size])


def build_and_train_model(state_vector_length, image_size, interface_vector_length, sequence_length,
                          tensorboard_log_dir, training_examples, training_path, test_path, testing_examples,
                          weights_path, run_name, round_patience=6, load_prevous_weigths=False, do_training=True):
    # Bygge modellen
    # K._set_session(K.tf.Session(config=K.tf.ConfigProto(log_device_placement=True)))
    model = create_model(sequence_length, image_size, interface_vector_length, state_vector_length)
    model.compile(optimizer="adam", loss=["mean_squared_error", "mean_squared_error"], loss_weights=[1, 1])

    print("Oppsummering av nettet:")
    model.summary()  # Skrive ut en oversikt over modellen
    print()

    if load_prevous_weigths:
        print("Laster inn vekter fra ", weights_path)
        model.load_weights(weights_path)
        evaluate_model(model, test_path, testing_examples)
    if do_training:
        print("Begynner trening.")
        train_model(model, round_patience, weights_path, tensorboard_log_dir, test_path, testing_examples,
                    training_examples, training_path, run_name)
    else:
        print("Hopper over trening.")
    return model


def train_model(model, round_patience, save_weights_path, tensorboard_log_dir, test_path, testing_examples,
                training_examples, training_path, run_name):
    # Trene modellen
    train_seq, train_startcoords, train_labels_pos, train_labels_size = data_io.fetch_seq_startcoords_labels(
        training_path, training_examples)
    # Debugging:
    # print("train_labels_pos[0]:")
    # print(train_labels_pos[0])
    # print("train_labels_size[0]:")
    # print(train_labels_size[0])
    epoches_per_round = 1

    loss_history = []  # Format: ((treningsloss, tr.loss_pos, tr.loss_str), (testloss, testloss_pos, testloss_str))

    best_loss = 10000
    num_of_rounds_without_improvement = 0
    max_num_of_rounds = int(1032 / epoches_per_round)
    time_at_start = os.times().elapsed
    for i in range(max_num_of_rounds):
        fit_history = model.fit(x=[train_seq, train_startcoords], y=[train_labels_pos, train_labels_size], epochs=epoches_per_round,
                  callbacks=[TerminateOnNaN(),  # EarlyStopping(monitor="loss", patience=4),
                             TensorBoard(log_dir=tensorboard_log_dir)],
                  verbose=1)

        evaluation = evaluate_model(model, test_path, testing_examples)
        # print(fit_history.history)

        # Plotte loss
        train_loss = (fit_history.history["loss"][0], fit_history.history["Posisjon_ut_loss"][0], fit_history.history["Stoerrelse_ut_loss"][0])
        test_loss = tuple(evaluation)
        loss_history.append((train_loss, test_loss))
        plotting.plot_loss_history(loss_history, run_name)

        print("Fullført runde {0}/{1} ({2} epoker). Brukt {3} minutter.".format(i + 1, max_num_of_rounds,
                                                                                (i + 1) * epoches_per_round,
                                                                                round((
                                                                                      os.times().elapsed - time_at_start) / 60,
                                                                                      1)))

        if evaluation[0] >= best_loss:
            num_of_rounds_without_improvement += 1
            print("Runder uten forbedring: {0}/{1}".format(num_of_rounds_without_improvement, round_patience))
            if num_of_rounds_without_improvement >= round_patience:
                print("Laster inn vekter fra ", save_weights_path)
                model.load_weights(save_weights_path)
                break
        else:
            num_of_rounds_without_improvement = 0
            best_loss = evaluation[0]
            model.save_weights(save_weights_path, overwrite=True)
            print("Lagret vekter til ", save_weights_path)
        print()


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
    training_examples = 100
    testing_examples = 100
    example_examples = 100

    for i in range(3):
        global RUN_ID
        RUN_ID = i
        try:
            do_run(example_examples, testing_examples, training_examples, False, True)
        except Exception:
            print("Det skjedde en feil med kjøring, ", RUN_ID)


def do_run(example_examples, testing_examples, training_examples, load_weights, do_training):
    # Parametre
    round_patience = 12

    interface_vector_length = 512
    state_vector_length = 512

    sequence_length = 12
    image_size = 32  # Antar kvadrat og 3 kanaler

    timestamp = time.localtime()
    run_name = "{0}-{1:02}-{2:02} {3:02}:{4:02}:{5:02}".format(timestamp[0], timestamp[1], timestamp[2], timestamp[3],
                                                               timestamp[4], timestamp[5])
    print_path = "saved_outputs"
    print("Skriver utdata til", os.path.join(print_path, run_name))
    orig_stdout = sys.stdout
    sys.stdout = open(os.path.join(print_path, run_name), 'w')

    print("run_name: ", run_name)
    # default_train_path = "/home/mby/Grafikk/tilfeldig_relativeKoordinater/train"
    # default_test_path = "/home/mby/Grafikk/tilfeldig_relativeKoordinater/test"
    # train_path = data_io.get_path_from_user(default_train_path, "mappen med treningssekvenser")
    # test_path = data_io.get_path_from_user(default_test_path, "mappen med testsekvenser")
    train_path = "../../Grafikk/tilfeldig_relativeKoordinater/train"
    print("Treningseksempler hentes fra ", train_path)
    test_path = "../../Grafikk/tilfeldig_relativeKoordinater/test"
    example_path = test_path
    print("Testeksempler hentes fra ", test_path)
    weights_path = os.path.join("saved_weights", run_name + ".h5")
    tensorboard_log_dir = "tensorboard_logs"

    model = build_and_train_model(state_vector_length, image_size, interface_vector_length, sequence_length,
                                  tensorboard_log_dir, training_examples, train_path, test_path, testing_examples,
                                  weights_path, run_name, round_patience=round_patience,
                                  load_prevous_weigths=load_weights, do_training=do_training)
    # TODO: Lagre modellens konfigurasjon som json
    make_examples(example_examples, example_path, model)

    sys.stdout = orig_stdout


def make_examples(example_examples, example_path, model):
    # Lage og vise eksempler
    example_sequences, example_startcoords, example_labels_pos, example_labels_size\
        = data_io.fetch_seq_startcoords_labels(example_path, example_examples)
    predictions = model.predict([example_sequences, example_startcoords])

    # print("predictions[1]:")
    # print(predictions[1])

    predictions = np.delete(predictions, 0, 2)  # Fjerne det første ("falske") tidssteget
    print("len(predictions):", len(predictions))
    print("len(predictions[0]):", len(predictions[0]))
    print("len(predictions[0][0]):", len(predictions[0][0]))

    # print("predictions[1]:")
    # print(predictions[1])

    for sequence_index in range(len(predictions[0])):
        path = os.path.join(example_path, "seq{0:05d}".format(sequence_index))
        data_io.write_labels(file_names=data_io.get_image_file_names_in_dir(path),
                             labels_pos=predictions[0][sequence_index],
                             labels_size=predictions[1][sequence_index],
                             path=path, json_file_name="predictions.json")
        # print_results(example_labels, example_sequences, predictions, sequence_length, 4)


def evaluate_model(model, test_path, testing_examples):
    test_sequences, test_startcoords, test_labels_pos, test_labels_size = data_io.fetch_seq_startcoords_labels(test_path, testing_examples)
    evaluation = model.evaluate([test_sequences, test_startcoords], [test_labels_pos, test_labels_size])
    print("\nEvaluering: ", evaluation, end="\n\n")
    return evaluation


main()