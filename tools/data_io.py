import os
import numpy
import json

from PIL import Image


def fetch_seq_startcoords_labels(main_path, max_count=0):
    """
    Leser inn sekvensmapper og genererer to sequences- og labels-arrayer som kan brukes til trening.
    labels hentes fra en fil ved navn "label.json" i sekvensmappen.
    :param main_path: Full bane til mappen som inneholder sekvensmappene (og bare det)
    :param max_count: Maksimalt antall sekvenser som skal lastes inn. 0 betyr ingen begrensning.
    :return: sequences, labels
    """

    sequences = []
    startcoords = []
    labels = []

    file_list = os.listdir(main_path)  # os.listdir sorterer ikke alfabetisk.
    file_list.sort()  # For å være konsekvent og for å kunne lettere sammenligne eksemplene
    if max_count <= 0:
        max_count = len(file_list)  # Inkludere alle sekvensene
    else:
        max_count = min(len(file_list), max_count)  # Pass på at max_count ikke er større enn lengden på listen

    print("Henter {0}/{1} sekvenser fra mappe: {2}".format(max_count, len(file_list), main_path))
    file_list = file_list[:max_count]
    for seq_name in file_list:  # Iterere gjennom sekvensene i mappen
        seq_path = os.path.join(main_path, seq_name)
        sequence = []
        for image_name in os.listdir(seq_path):  # Iterere gjennom bildene i sekvensen
            # TODO: bruke get_image_file_names_in_dir
            image_path = os.path.join(seq_path, image_name)
            try:
                im = Image.open(image_path)
                image_array = numpy.array(im)
                sequence.append(image_array)  # Legger til bildet i sekvensen
            except OSError:  # Dette er ikke en bildefil.
                pass
            finally:
                im.close()

        sequences.append(sequence)  # Legger til sekvensen i listen over sekvenser

        # Hente merkelapper
        with open(os.path.join(seq_path, "labels.json")) as label_file:  # Åpne filen
            label_raw = json.load(label_file)  # Lese merkelapper fra filen

        sequence_labels = [(i["x1"], i["y1"], i["x2"], i["y2"]) for i in label_raw]
        sequence_labels.insert(0, sequence_labels[0])  # Doble den første merkelappen fordi vi får den dobbelt opp fra nettverket (dårlig løsning)
        labels.append(sequence_labels)
        startcoords.append(sequence_labels[0])

    sequences = numpy.array(sequences)
    startcoords = numpy.array(startcoords)
    return sequences, startcoords, labels


def write_labels(file_names, labels, path, json_file_name):
    # Skrive merkelapper til fil
    formatted_labels = [
        {"filename": file_names[i], "x1": float(labels[i][0]), "y1": float(labels[i][1]),
         "x2": float(labels[i][2]), "y2": float(labels[i][3])} for
        i in range(len(labels))]
    with open(os.path.join(path, json_file_name), "w") as label_file:
        json.dump(formatted_labels, label_file)


def get_image_file_names_in_dir(dir_path, suffix=".jpg"):
    files = os.listdir(dir_path)
    files = [i for i in files if i.endswith(suffix)]
    files.sort()
    return files


def get_path_from_user(default_path, path):
    while path is None:
        path = input("Mappe med treningssekvenser (trykk enter for \"{0}\"): >".format(default_path))
        if path == "":
            path = default_path
        if not os.access(path, os.F_OK):
            print("%s er ikke en mappe." % path)
    return path