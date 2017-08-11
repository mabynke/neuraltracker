import os
import numpy
import json
import random

from PIL import Image


def fetch_seq_startcoords_labels(main_path, max_count=0, max_length=0, output_size=224, frame_stride=1):
    """
    Leser inn sekvensmapper og genererer to sequences- og labels_pos-arrayer som kan brukes til trening.
    labels_pos hentes fra en fil ved navn "label.json" i sekvensmappen.
    :param main_path: Full bane til mappen som inneholder sekvensmappene (og bare det)
    :param max_count: Maksimalt antall sekvenser som skal lastes inn. 0 betyr ingen begrensning.
    :param max_length: Det maksimale antall bilder som skal hentes per sekvens. Kutter fra slutten. 0 betyr ingen begrensning.
    :param output_size: Størrelsen som bildene skal skaleres til (heltall). Alle bilder blir skalert til kvadrater.
    :param frame_stride: Kan brukes til å ikke laste inn alle bilder, men bare f.eks. hvert andre (ved å sette til 2).
    :return: sequences, startcoords, labels_pos, labels_size, json_paths
    """

    sequences = []  # Sekvensene av bilder – ett element per sekvens
    startcoords = []  # Startkoordinatene for hver sekvens
    labels_pos = []  # x og y for hvert bilde i hver sekvens
    labels_size = []  # w og h for hvert bilde i hver sekvens
    json_paths = []  # Liste over stiene til json-filene som hver sekvens er hentet fra

    list_of_sequence_folders = os.listdir(main_path)  # os.listdir sorterer ikke alfabetisk.
    list_of_sequence_folders.sort()  # Kanskje ikke nødvendig etter at vi lagrer json_paths og prediksjonene derfor uansett havner i riktig mappe?

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
            print("Henter sekvens {0}/{1} ...".format(sequence_index, max_count))
        sequence_dir = os.path.join(main_path, list_of_sequence_folders[sequence_index])

        files_in_sequence_dir = os.listdir(sequence_dir)
        label_files_in_sequence_dir = [i for i in files_in_sequence_dir if (i[:6] == "labels" and i[-5:] == ".json")]

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

            if max_length > 0 and max_length < len(image_names_in_object_sequence):
                image_names_in_object_sequence = image_names_in_object_sequence[:max_length]
                labels = labels[:max_length]

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


def write_labels(dir_path, json_file_name, labels_pos, labels_size, file_names):
    """Skrive merkelapper til json-fil."""

    formatted_labels = []
    for i in range(len(labels_pos)):
        formatted_labels.append({"filename": file_names[i],
                                 "x": float(labels_pos[i][0]),
                                 "y": float(labels_pos[i][1]),
                                 "w": float(labels_size[i][0]),
                                 "h": float(labels_size[i][1])})
    with open(os.path.join(dir_path, json_file_name), "w") as label_file:
        json.dump(formatted_labels, label_file)


def load_image_section(size_x, size_y, image_path=None, pos_x=None, pos_y=None, scale=1, numpy_array=True):
    if image_path is None:
        dir = "../../INRIAHolidays/jpg"
        file = random.choice(os.listdir(dir))
        image_path = os.path.join(dir, file)

    im = Image.open(image_path)
    orig_size_x, orig_size_y = im.size

    section_size_x = min(round(size_x / scale), orig_size_x)
    section_size_y = min(round(size_y / scale), orig_size_y)

    if pos_x is None:
        pos_x = random.randint(0, orig_size_x - section_size_x)
    if pos_y is None:
        pos_y = random.randint(0, orig_size_y - section_size_y)

    im = im.crop((pos_x, pos_y, pos_x + section_size_x, pos_y + section_size_y))
    im = im.resize((size_x, size_y), Image.BILINEAR)

    if numpy_array:
        im = numpy.array(im)
    return im


def get_existing_image_section(dir="../../INRIAHolidays/utsnitt"):
    """ Returnerer ferdiggenererte små bakgrunner i mappen dir. """

    file = random.choice(os.listdir(dir))
    image_path = os.path.join(dir, file)
    im = Image.open(image_path)

    return im


def get_image_file_names_in_json(json_path):
    """Henter ut en liste over filnavnene til alle bilder i en merkelapp-json-fil i riktig rekkefølge."""

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


def main():
    dir = "../../INRIAHolidays/jpg"
    out_dir = "../../INRIAHolidays/utsnitt"
    file_list = os.listdir(dir)
    for i in range(100000):
        if not i % 1000:
            print("Framgang: {0}/{1}".format(i, 100000))
        image = load_image_section(size_x=32, size_y=32, scale=0.01, image_path=os.path.join(dir, random.choice(file_list)), numpy_array=False)
        image.save(os.path.join(out_dir, str(i)+".bmp"))


if __name__ == "__main__":
    main()