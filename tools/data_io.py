import os
import numpy
import json

from PIL import Image


def fetch_seq_startcoords_labels(main_path, max_count=0, output_size=224, frame_stride=1):
    """
    Leser inn sekvensmapper og genererer to sequences- og labels_pos-arrayer som kan brukes til trening.
    labels_pos hentes fra en fil ved navn "label.json" i sekvensmappen.
    :param main_path: Full bane til mappen som inneholder sekvensmappene (og bare det)
    :param max_count: Maksimalt antall sekvenser som skal lastes inn. 0 betyr ingen begrensning.
    :return: sequences, startcoords, labels_pos, labels_size, json_paths
    """

    sequences = []  # Sekvensene av bilder – ett element per sekvens
    startcoords = []  # Startkoordinatene for hver sekvens
    labels_pos = []  # x og y for hvert bilde i hver sekvens
    labels_size = []  # w og h for hvert bilde i hver sekvens
    json_paths = []

    list_of_sequence_folders = os.listdir(main_path)  # os.listdir sorterer ikke alfabetisk.
    list_of_sequence_folders.sort()  # For å være konsekvent og for å lagre eksemplene i riktige mapper

    # Kontrollere antall sekvenser som skal lastes inn
    if max_count <= 0:
        max_count = len(list_of_sequence_folders)  # Inkludere alle sekvensene i mappen
    else:
        max_count = min(len(list_of_sequence_folders), max_count)  # Pass på at max_count ikke er større enn lengden på listen
    print("Henter {0}/{1} sekvenser fra mappe: {2}".format(max_count, len(list_of_sequence_folders), main_path))
    if max_count != len(list_of_sequence_folders):
        list_of_sequence_folders = list_of_sequence_folders[:max_count]

    print("Bildestørrelse: {0}*{0}".format(output_size))

    for sequence_index in range(len(list_of_sequence_folders)):  # Iterere gjennom sekvensene i mappen
        if not sequence_index % 5000:
            print("Henter sekvens {0}/{1}...".format(sequence_index, max_count))
        sequence_dir = os.path.join(main_path, list_of_sequence_folders[sequence_index])

        files_in_sequence_dir = os.listdir(sequence_dir)
        label_files_in_sequence_dir = [i for i in files_in_sequence_dir if (i[:6] == "labels" and i[-5:] == ".json")]

        # print("Fant {0} objektsekvenser i {1}.".format(len(label_files_in_sequence_dir), sequence_name))

        for object_sequence_index in range(len(label_files_in_sequence_dir)):
            # print("Laster inn fra", label_files_in_sequence_dir[object_sequence_index])

            # Hente ut bildene som viser det gitte objektet, fra dets json-fil.
            image_names_in_object_sequence = []
            json_path = os.path.join(sequence_dir, label_files_in_sequence_dir[object_sequence_index])
            json_paths.append(json_path)
            with open(json_path) as label_file:
                labels = json.load(label_file)
            if frame_stride > 1:
                labels = labels[::frame_stride]
            for image_label in labels:
                try:
                    image_names_in_object_sequence.append(image_label["filename"])
                except KeyError:  # Denne oppføringen er ikke et bilde, men sannsynligvis en som inneholder "img_amount".
                    labels.remove(image_label)  # Fjerner denne oppføringen fordi den skaper problemer senere.

            sequence = []  # Bildene i denne objektsekvensen
            for image_name in image_names_in_object_sequence:  # Iterere gjennom bildene i objektsekvensen og legge dem til i sequence
                image_path = os.path.join(sequence_dir, image_name)
                try:
                    im = Image.open(image_path)
                    im = im.resize((output_size, output_size), Image.BILINEAR)
                    image_array = numpy.array(im)
                    sequence.append(image_array)  # Legger til bildet i sekvensen
                except OSError:  # Dette er ikke en bildefil.
                    pass
                finally:
                    im.close()

            sequences.append(sequence)  # Legge til sekvensen i listen over sekvensen

            sequence_labels_pos = [(i["x"], i["y"]) for i in labels]  # Posisjonsdata for denne sekvensen
            sequence_labels_size = [(i["w"], i["h"]) for i in labels]  # Størrelsesdata for denne sekvensen
            # # Doble den første merkelappen fordi vi får den dobbelt opp fra nettverket
            # sequence_labels_pos.insert(0, sequence_labels_pos[0])
            # sequence_labels_size.insert(0, sequence_labels_size[0])
            labels_pos.append(sequence_labels_pos)
            labels_size.append(sequence_labels_size)

            startcoords.append((sequence_labels_pos[0][0],
                                sequence_labels_pos[0][1],
                                sequence_labels_size[0][0],
                                sequence_labels_size[0][1]))

    # sequences = numpy.array(sequences)
    # startcoords = numpy.array(startcoords)
    # labels_pos = numpy.array(labels_pos)
    # labels_size = numpy.array(labels_size)

    return sequences, startcoords, labels_pos, labels_size, json_paths


def write_labels(path, json_file_name, labels_pos, labels_size, file_names):
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


def get_image_file_names_in_json(json_path):
    with open(json_path, "r") as json_file:
        json_content = json.load(json_file)
    image_file_names = []
    for entry in json_content:
        try:
            image_file_names.append(entry["filename"])
        except KeyError:
            pass
    image_file_names.sort()
    return image_file_names


def get_path_from_user(default_path, description):
    path = None
    while path is None:
        path = input("Skriv inn banen til {1} (trykk enter for \"{0}\"): >".format(default_path, description))
        if path == "":
            path = default_path
        if not os.access(path, os.F_OK):
            print("%s er ikke en mappe eller fil." % path)
    return path