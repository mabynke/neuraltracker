import os
import numpy
import json

from PIL import Image


def fetch_seq_startcoords_labels(main_path, max_count=0):
    """
    Leser inn sekvensmapper og genererer to sequences- og labels_pos-arrayer som kan brukes til trening.
    labels_pos hentes fra en fil ved navn "label.json" i sekvensmappen.
    :param main_path: Full bane til mappen som inneholder sekvensmappene (og bare det)
    :param max_count: Maksimalt antall sekvenser som skal lastes inn. 0 betyr ingen begrensning.
    :return: sequences, labels_pos, labels_size
    """

    sequences = []
    startcoords = []
    labels_pos = []
    labels_size = []

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

        sequence_labels_pos = [(i["x"], i["y"]) for i in label_raw]
        sequence_labels_size = [(i["w"], i["h"]) for i in label_raw]
        # Doble den første merkelappen fordi vi får den dobbelt opp fra nettverket
        sequence_labels_pos.insert(0, sequence_labels_pos[0])
        sequence_labels_size.insert(0, sequence_labels_size[0])
        labels_pos.append(sequence_labels_pos)
        labels_size.append(sequence_labels_size)

        startcoords.append((sequence_labels_pos[0][0],
                            sequence_labels_pos[0][1],
                            sequence_labels_size[0][0],
                            sequence_labels_size[0][1]))

    sequences = numpy.array(sequences)
    startcoords = numpy.array(startcoords)
    labels_pos = numpy.array(labels_pos)
    labels_size = numpy.array(labels_size)
    return sequences, startcoords, labels_pos, labels_size


def write_labels(file_names, labels_pos, labels_size, path, json_file_name):
    # Skrive merkelapper til fil
    formatted_labels = []
    for i in range(len(labels_pos)):
        formatted_labels.append({"filename": file_names[i],
                                 "x": float(labels_pos[i][0]),
                                 "y": float(labels_pos[i][1]),
                                 "w": float(labels_size[i][0]),
                                 "h": float(labels_size[i][1])})
    with open(os.path.join(path, json_file_name), "w") as label_file:
        json.dump(formatted_labels, label_file)


def get_image_file_names_in_dir(dir_path, suffix=".jpg"):
    files = os.listdir(dir_path)
    files = [i for i in files if i.endswith(suffix)]
    files.sort()
    return files


def get_path_from_user(default_path, description):
    path = None
    while path is None:
        path = input("Skriv inn banen til {1} (trykk enter for \"{0}\"): >".format(default_path, description))
        if path == "":
            path = default_path
        if not os.access(path, os.F_OK):
            print("%s er ikke en mappe eller fil." % path)
    return path