import os
import numpy

from PIL import Image


def fetch_x_y(main_path, max_count=0, single_image=False):
    """
    Leser inn sekvensmapper og genererer to x- og y-arrayer som kan brukes til trening.
    y hentes fra en fil ved navn "label" i sekvensmappen.
    :param main_path: Full bane til mappen som inneholder sekvensmappene (og bare det)
    :param max_count: Maksimalt antall sekvenser som skal lastes inn. 0 betyr ingen begrensning.
    :param single_image: False: Laster inn hele sekvensen i en array av bilder. True: Laster inn bare det første bildet i sekvensen.
    :return: x, y
    """

    x = []
    y = []

    file_list = os.listdir(main_path)   # os.listdir sorterer ikke alfabetisk.
    file_list.sort()  # For å være konsekvent og for å kunne lettere sammenligne eksemplene
    if max_count <= 0:
        max_count = len(file_list)
    else:
        max_count = min(len(file_list), max_count)  # Pass på at max_count ikke er større enn lengden på listen

    print("Henter {0}/{1} sekvenser fra mappe: {2}".format(max_count, len(file_list), main_path))
    file_list = file_list[:max_count]
    for seq_name in file_list:
        seq_path = os.path.join(main_path, seq_name)
        sequence = []
        for image_name in os.listdir(seq_path):
            image_path = os.path.join(seq_path, image_name)
            try:
                im = Image.open(image_path)
                image_array = numpy.array(im)

                if single_image:
                    sequence = image_array
                    break
                else:
                    sequence.append(image_array)

            except OSError:
                # print("{0} er ikke et lesbart bilde.".format(image_path))
                pass
            finally:
                im.close()

        x.append(sequence)

        # Hente labels
        label_file = open(os.path.join(seq_path, "labels"))         # Åpne filen
        labels = label_file.readlines()                             # Lese linjer fra filen
        label_file.close()

        labels = [line.split(",") for line in labels]               # Dele opp hver linje etter komma
        labels = [[float(i) for i in line[1:]] for line in labels]  # Ignorere filnavnet i hver linje og konvertere hvert tall til flyttall
        if single_image:
            labels = labels[0]                                      # Evt. beholde data fra bare det første bildet
        y.append(labels)

    x = numpy.array(x)
    return x, y


if __name__ == "__main__":
    fetch_x_y("/home/mathias/inndata/generert/tilfeldig bevegelse/examples/", 0)