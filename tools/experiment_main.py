import os
import sys
import time
import numpy
import random
import math

# from tools import data_io  # For kjøring fra PyCharm
import data_io  # For kjøring fra terminal
import plotting
# import tf_tracker

# Tar GPU-enhet som første argument
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except IndexError:
    print("Trenger et argument: GPU-enhet (0-3)")
    exit()

# from keras.callbacks import TerminateOnNaN, TensorBoard
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.backend import get_value, set_value


RUN_ID = 0
USE_DENSENET = False

if USE_DENSENET:
    from densenet121 import DenseNet
# import keras.backend as K


def create_model(image_size, interface_vector_length, state_vector_length):
    # Ta innputt
    input_sequence = Input(shape=(None, image_size, image_size, 3), name="Innsekvens")
    input_coordinates = Input(shape=(4,), name="Innkoordinater")

    # Behandle bildesekvensen
    if USE_DENSENET:
        weights_path = 'imagenet_models/densenet121_weights_tf.h5'
        dense_model = DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
        for layer in dense_model.layers:
            layer.trainable = False
        x = TimeDistributed(dense_model)(input_sequence)

    else:
        for i in range(round(math.log(image_size, 2)-3)):  # Redusere til ca. 8*8
            print("Lager konvolusjon- og maxpoolinglag nr.", i)
            if i == 0:
                kernel_size = (5, 5)  # Første laget har større kjerne
                input = input_sequence
            else:
                kernel_size = (3, 3)
                input = x
            x = TimeDistributed(Conv2D(filters=32, kernel_size=kernel_size, activation="relu", padding="same"), name="Konv"+str(i+1))(input)
            x = TimeDistributed(MaxPooling2D((2, 2)), name="maxpooling"+str(i+1))(x)

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


def build_and_train_model(state_vector_length, image_size, interface_vector_length, tensorboard_log_dir,
                          training_examples, training_path, test_path, testing_examples, weights_path, run_name,
                          round_patience, load_prevous_weigths, do_training, save_results, batch_size):
    # Bygge modellen
    # model = create_model(image_size, interface_vector_length, state_vector_length)
    model = create_model(image_size, interface_vector_length, state_vector_length)
    model.compile(optimizer=Adam(), loss=["mean_squared_error", "mean_squared_error"], loss_weights=[1, 1])

    print("Oppsummering av nettet:")
    model.summary()  # Skrive ut en oversikt over modellen
    print()

    if load_prevous_weigths:
        print("Laster inn vekter fra ", weights_path)
        model.load_weights(weights_path)
        # evaluate_model(model, test_path, testing_examples)
    if do_training:
        print("Begynner trening.")
        train_model(model, round_patience, weights_path, tensorboard_log_dir, test_path, testing_examples,
                    training_examples, training_path, run_name, save_results=save_results, image_size=image_size,
                    batch_size=batch_size)
    else:
        print("Hopper over trening.")
    return model


def train_model(model, round_patience, save_weights_path, tensorboard_log_dir, test_path, testing_examples,
                training_examples, training_path, run_name, save_results, image_size, batch_size):

    train_seqs, train_startcoords, train_labels_pos, train_labels_size, _ =\
        data_io.fetch_seq_startcoords_labels(training_path, training_examples, output_size=image_size)
    test_seqs, test_startcoords, test_labels_pos, test_labels_size, _ = \
        data_io.fetch_seq_startcoords_labels(test_path, testing_examples, output_size=image_size)

    loss_history = []  # Format: ((treningsloss, tr.loss_pos, tr.loss_str), (testloss, testloss_pos, testloss_str))

    max_num_of_epoches = 10000
    best_loss = 10000  # Kan like gjerne være uendelig, bare den er høyere enn alle loss-verdier
    num_of_rounds_without_improvement = 0
    time_at_start = os.times().elapsed
    for epoch_index in range(max_num_of_epoches):
        train_loss_history = []
        test_loss_history = []

        # Trene
        print("Trener ...")
        for batch_start in range(0, len(train_seqs), batch_size):  # For hver objektsekvens i treningssettet
            batch_end = min(batch_start+batch_size, len(train_seqs) - 1)
            # print("Trener på sekvens {0}-{1} …".format(batch_start, batch_end))

            batch_sequences = train_seqs[batch_start:batch_end+1]
            batch_startcoords = train_startcoords[batch_start:batch_end+1]
            batch_labels_pos = train_labels_pos[batch_start:batch_end+1]
            batch_labels_size = train_labels_size[batch_start:batch_end+1]

            train_losses = model.train_on_batch(x=[numpy.array(batch_sequences),
                                                   numpy.array(batch_startcoords)],
                                                y=[numpy.array(batch_labels_pos),
                                                   numpy.array(batch_labels_size)])
            train_loss_history.append(train_losses)

        train_loss_history = numpy.array(train_loss_history)
        train_loss = (numpy.mean(train_loss_history[:, 0]),
                      numpy.mean(train_loss_history[:, 1]),
                      numpy.mean(train_loss_history[:, 2]))
        print("Treningsloss:", train_loss)

        # Teste
        print("Tester ...")
        for batch_start in range(0, len(test_seqs), batch_size):  # For hver objektsekvens i treningssettet
            batch_end = min(batch_start+batch_size, len(test_seqs) - 1)
            # print("Tester på sekvens {0}-{1} …".format(batch_start, batch_end))

            batch_sequences = test_seqs[batch_start:batch_end+1]
            batch_startcoords = test_startcoords[batch_start:batch_end+1]
            batch_labels_pos = test_labels_pos[batch_start:batch_end+1]
            batch_labels_size = test_labels_size[batch_start:batch_end+1]

            test_losses = model.test_on_batch(x=[numpy.array(batch_sequences),
                                                   numpy.array(batch_startcoords)],
                                                y=[numpy.array(batch_labels_pos),
                                                   numpy.array(batch_labels_size)])
            test_loss_history.append(test_losses)

        test_loss_history = numpy.array(test_loss_history)
        test_loss = (numpy.mean(test_loss_history[:, 0]),
                      numpy.mean(test_loss_history[:, 1]),
                      numpy.mean(test_loss_history[:, 2]))
        print("Testloss:", test_loss)

        if save_results:
            # Plotte loss
            loss_history.append((train_loss, test_loss))
            plotting.plot_loss_history(loss_history, run_name)

        print("Fullført epoke {0}/{1}. Brukt {2} minutter."
              .format(epoch_index + 1,
                      max_num_of_epoches,
                      round((os.times().elapsed - time_at_start) / 60, 1)))

        if train_loss[0] >= best_loss:
            num_of_rounds_without_improvement += 1
            print("Runder uten forbedring: {0}/{1}".format(num_of_rounds_without_improvement, round_patience))
            if num_of_rounds_without_improvement == round_patience:  # Senke læringsrate
                set_value(model.optimizer.lr, get_value(model.optimizer.lr) / 10)
                print("Senket læringsrate til", get_value(model.optimizer.lr))
            if num_of_rounds_without_improvement >= round_patience + 4:  # Avslutte treningen
                if save_results:
                    print("Laster inn vekter fra ", save_weights_path)
                    model.load_weights(save_weights_path)
                break
        else:
            num_of_rounds_without_improvement = 0
            best_loss = train_loss[0]
            if save_results:
                model.save_weights(save_weights_path, overwrite=True)
                print("Lagret vekter til ", save_weights_path)
        print("Beste treningsloss så langt:", best_loss)
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


def do_run(example_examples=100, testing_examples=0, training_examples=0, load_weights=False, do_training=True,
           make_predictions=True, round_patience=8, interface_vector_length=512, state_vector_length=512,
           save_results=True, image_size=32, batch_size=16):

    timestamp = time.localtime()
    run_name = "{0}-{1:02}-{2:02} {3:02}:{4:02}:{5:02}".format(timestamp[0], timestamp[1], timestamp[2], timestamp[3],
                                                               timestamp[4], timestamp[5])
    if save_results:
        print_path = "saved_outputs"
        print("Skriver utdata til", os.path.join(print_path, run_name))
        orig_stdout = sys.stdout
        sys.stdout = open(os.path.join(print_path, run_name), 'w', buffering=1)

    print("run_name: ", run_name)
    train_path = "../../Grafikk/tilfeldig_relativeKoordinater/train"
    print("Treningseksempler hentes fra ", train_path)
    test_path = "../../Grafikk/tilfeldig_relativeKoordinater/test"
    example_path = "../../Grafikk/tilfeldig_relativeKoordinater/test"
    print("Testeksempler hentes fra ", test_path)
    if load_weights:
        weights_path = os.path.join("saved_weights", input("Skriv filnavnet til vektene som skal lastes inn (ikke inkludert \".h5\"): ") + ".h5")
    else:
        weights_path = os.path.join("saved_weights", run_name + ".h5")
    tensorboard_log_dir = "tensorboard_logs"

    model = build_and_train_model(state_vector_length, image_size, interface_vector_length, tensorboard_log_dir,
                                  training_examples, train_path, test_path, testing_examples, weights_path, run_name,
                                  round_patience=round_patience, load_prevous_weigths=load_weights,
                                  do_training=do_training, save_results=save_results, batch_size=batch_size)
    # TODO: Lagre modellens konfigurasjon som json
    if make_predictions:
        make_example_jsons(example_examples, example_path, model, image_size=image_size)

    if save_results:
        sys.stdout = orig_stdout


def make_example_jsons(example_examples, example_path, model, image_size):
    # Lage og vise eksempler
    example_sequences, example_startcoords, _, _, json_paths\
        = data_io.fetch_seq_startcoords_labels(example_path, example_examples, output_size=image_size, frame_stride=1)

    for sequence_index in range(len(example_sequences)):
        predictions = model.predict([numpy.array(example_sequences[sequence_index:sequence_index+1]),
                                     numpy.array(example_startcoords[sequence_index:sequence_index+1])])

        json_label_path = json_paths[sequence_index]
        dir_path = os.path.split(json_label_path)[0]
        json_pred_name = "predictions" + os.path.split(json_label_path)[1][6:]

        data_io.write_labels(file_names=data_io.get_image_file_names_in_json(json_label_path),
                             labels_pos=predictions[0][0],
                             labels_size=predictions[1][0],
                             path=dir_path, json_file_name=json_pred_name)
        # print_results(example_labels, example_sequences, predictions, sequence_length, 4)


def evaluate_model(model, test_sequences, test_startcoords, test_labels_pos, test_labels_size):


    evaluation = model.evaluate([test_sequences, test_startcoords], [test_labels_pos, test_labels_size], verbose=0)
    print("\nEvaluering: ", evaluation, end="\n\n")
    return evaluation


def main():
    os.chdir(os.path.dirname(sys.argv[0]))  # set working directory to that of the script
    # Oppsett
    save_results = True  # Husk denne! Lagrer vekter, plott og stdout.
    load_saved_weights = False
    do_training = True
    make_example_jsons = True
    training_examples = 0
    testing_examples = 0
    sequences_to_predict = 100  # Antall sekvenser det skal lages prediksjons-json-filer til.
    patience_before_lowering_lr = 8
    batch_size = 16  # Antall sekvenser som skal inkluderes per parti (trening og testing)

    if USE_DENSENET:
        image_size = 224
    else:
        image_size = 32

    for i in range(1):  # Kjøre de angitte eksperimentene
        global RUN_ID
        RUN_ID = i
        try:
            # with tf.device("/gpu:0"):
            do_run(sequences_to_predict, testing_examples, training_examples, load_saved_weights, do_training,
                   make_example_jsons, round_patience=patience_before_lowering_lr, save_results=save_results,
                   image_size=image_size, batch_size=batch_size)
        except IOError as e:
            print("Det skjedde en feil med kjøring nr.", RUN_ID)
            print(e)


if __name__ == "__main__":
    main()
