import os
import sys
import random

import numpy as np
import scipy.misc

# from tools.data_io import write_labels
from data_io import write_labels


# def generate_movement_binarylabel(category, binary_type="horizontal-vertical", frames=12, size_x=32, size_y=32, channels=3):
#     square_size = 1
#
#     sequence = np.zeros((frames, size_x, size_y, channels), dtype=np.int)
#
#     if binary_type == "horizontal-vertical":
#         speedX = (category - 1) * random.random() * size_x / 2 / frames
#         speedY = category       * random.random() * size_y / 2 / frames
#         posX = size_x / 2
#         posY = size_y / 2
#     elif binary_type == "fast-slow_right":
#         speedX = (size_x / frames) * (1 + random.random() + category)
#         speedY = 0
#         posX = random.randint(0, size_x - 1)
#         posY = size_y / 2
#
#     else:
#         print("FEIL i sekvensgenerator.generate_binary_linear_movement: Ukjent binary_type: ", binary_type)
#         return None
#
#     for frame in range(frames):
#         # TODO: Bruke draw_rectangle her dersom vi beholder denne funksjonen
#         for widthIndex in range(square_size):
#             for heightIndex in range(square_size):
#                 for channel in range(channels):
#                     sequence[frame,
#                              round(posY + heightIndex - square_size / 2) % size_y,
#                              round(posX  + widthIndex - square_size / 2) % size_x,
#                              channel] = 255
#         posX += speedX
#         posY += speedY
#
#     return sequence


def generate_movement_positionlabel(sequence, type="random", frames=12, size_x=32, size_y=32, channels=3):
    changing_color = False
    changing_size = True

    labels_pos = []
    labels_size = []

    # Initialisere
    if type == "random":
        pos_x = random.randint(0, size_x - 1)
        pos_y = random.randint(0, size_y - 1)
        speed_x = random.gauss(0, 0.05 * size_x)
        speed_y = random.gauss(0, 0.05 * size_y)
        rect_size_x = random.randint(3, int(size_x / 3))
        rect_size_y = random.randint(3, int(size_x / 3))
        color = [random.randint(1, 255) for i in range(3)]
        color_speed = [random.gauss(0, 10) for i in range(3)]
    else:
        raise ValueError("Ukjent verdi av 'type': {0}".format(type))

    for frame in range(frames):  # For hvert bilde i sekvensen
        draw_rectangle(sequence, frame, pos_x, pos_y, rect_size_x, rect_size_y, channels, color=color)  # Tegne inn firkant i bildet

        # Lagre merkelapp for dette bildet
        x_min = max(0.0, pos_x - rect_size_x / 2)  # Venstre kant av rektangelet
        y_min = max(0.0, pos_y - rect_size_y / 2)  # Øvre kant av rektangelet
        x_max = min(size_x, pos_x + rect_size_x / 2)  # Høyre kant av rektangelet
        y_max = min(size_y, pos_y + rect_size_y / 2)  # Nedre kant av rektangelet
        # Konvertere til relativt koordinatsystem
        x = (x_max + x_min) / size_x - 1
        y = (y_max + y_min) / size_y - 1
        w = (x_max - x_min) / size_x
        h = (y_max - y_min) / size_y
        labels_pos.append((x, y))
        labels_size.append((w, h))

        # Oppdatere posisjon og fart til neste bilde/tidssteg
        pos_x += speed_x
        pos_y += speed_y
        speed_x += random.gauss(0, 0.03 * size_x)
        speed_y += random.gauss(0, 0.03 * size_y)
        speed_x += 0.05 * (size_x/2 - pos_x)
        speed_y += 0.05 * (size_y/2 - pos_y)
        # Sørge for at firkanten holder seg innenfor bildet
        if pos_x < 0:
            pos_x = 0
            speed_x = 0
        elif pos_x >= size_x:
            pos_x = size_x - 1
            speed_x = 0
        if pos_y < 0:
            pos_y = 0
            speed_y = 0
        elif pos_y >= size_y:
            pos_y = size_y - 1
            speed_y = 0

        # Oppdatere størrelse til neste bilde/tidssteg
        if changing_size:
            rect_size_x += random.gauss(0, 0.1 * rect_size_x)
            rect_size_x = min(rect_size_x, 12)
            rect_size_x = max(rect_size_x, 2)
            rect_size_y += random.gauss(0, 0.1 * rect_size_y)
            rect_size_y = min(rect_size_y, 12)
            rect_size_y = max(rect_size_y, 2)

        # Oppdatere farge til neste bilde/tidssteg
        if changing_color:
            for ch in range(3):
                color[ch] = color[ch] + color_speed[ch]
                if color[ch] < 0:
                    color[ch] = 0
                    color_speed[ch] = 0
                elif color[ch] > 255:
                    color[ch] = 255
                    color_speed[ch] = 0
                color_speed[ch] += random.gauss(0, 6)

    return labels_pos, labels_size


def draw_rectangle(sequence, frame, pos_x, pos_y, rect_size_x=4, rect_size_y=4, channels=3, color=(255, 255, 255)):
    """
    Tegner et kvadrat i en numpy-array sequence på den gitte posisjonen.

    :param sequence: Sekvens av bilder (numpy-array)
    :param frame: Indeksen til bildet i sekvensen som skal endres
    :param pos_x: x-koordinaten til midten av rektangelet
    :param pos_y: y-koordinaten til midten av rektangelet
    :param rect_size_x: bredden til rektangelet
    :param rect_size_y: høyden til rektangelet
    :param channels: antallet fargekanaler
    :param color: fargen til rektangelet, som tuppel med oppføring for hver fargekanal
    :return:
    """

    assert len(color) == channels

    orig_x = round(pos_x - rect_size_x / 2)      # Venstre kant av rektangelet
    orig_y = round(pos_y - rect_size_y / 2)      # Øvre kant av rektangelet
    for widthIndex in range(round(rect_size_x)):           # Iterere over bredden til firkanten
        current_pos_x = orig_x + widthIndex
        if current_pos_x < 0 or current_pos_x > len(sequence[0][0]):    # Utenfor bildet
            continue
        for heightIndex in range(round(rect_size_y)):      # Iterere over høyden til firkanten
            current_pos_y = orig_y + heightIndex
            if current_pos_y < 0 or current_pos_y > len(sequence[0]):  # Utenfor bildet
                continue
            for channel in range(channels):         # Iterere over fargekanalene
                try:
                    sequence[frame, current_pos_y, current_pos_x, channel] = color[channel]
                except IndexError:
                    pass


def make_background(sequence, pattern="random_noise_constant"):
    if pattern == "random_noise_constant":
        for x in range(len(sequence[0])):
            for y in range(len(sequence[0][0])):
                for ch in range(3):
                    value = random.randint(0, 255)
                    for frame in range(len(sequence)):
                        sequence[frame][x][y][ch] = value
    else:
        raise ValueError("Kjenner ikke mønsteret " + pattern)


# def save_sequence_binarylabel(sequence, parent_path, type):
#     seq_number = 0
#     while True:
#         path_base = os.path.join(parent_path, "seq{0:05d}".format(seq_number))
#         for i in range(2):
#             if os.access(path_base + " " + str(i), os.F_OK):
#                 seq_number += 1
#                 break
#         else:   # Dersom det ikke ble funnet en eksisterende mappe
#             break
#         continue   # Dersom det ble funnet en eksisterende mappe
#
#     path = os.path.join(path_base + " " + type)
#     os.mkdir(path)
#
#     for frame in range(len(sequence)):
#         image_array = sequence[frame]
#         scipy.misc.imsave(os.path.join(path, "frame{0:05d}.jpg".format(frame)), image_array)


def save_sequence_labelfile(sequence, labels_pos, labels_size, parent_path, seq_number):
    path = os.path.join(parent_path, "seq{0:05d}".format(seq_number))
    os.mkdir(path)

    # Iterere gjennom bildene i sekvensen
    file_names = []
    for frame in range(len(sequence)):
        file_name = "frame{0:05d}.jpg".format(frame)
        file_names.append(file_name)
        image_array = sequence[frame]
        scipy.misc.imsave(os.path.join(path, file_name), image_array)

    write_labels(file_names=file_names, labels_pos=labels_pos, labels_size=labels_size, dir_path=path,
                 json_file_name="labels.json")


def create_train_test_examples(path, counts, figures=1):
    frames = 24
    image_size = 32
    channels = 3

    # type = bool(random.getrandbits(1))
    # sequence = generate_movement_binarylabel(type, binary_type="fast-slow_right", frames=frames, sizeX=image_size, sizeY=image_size)
    # save_sequence(sequence, "/home/mathias/inndata/generert/fast-slow_right/eksempel", "1" if type else "0")

    folder_names = ["train", "test"]

    for name_index in range(len(folder_names)):
        name = folder_names[name_index]
        try:
            os.mkdir(os.path.join(path, name))
        except IOError:
            print("Mappen \"{0}\" fins allerede.".format(os.path.join(path, name)))

        count = counts[name_index]
        for sequence_index in range(count):
            if not sequence_index % 2000:
                print("Skrevet {0}/{1} sekvenser til {2}".format(sequence_index, count, name))
            sequence = np.zeros((frames, image_size, image_size, channels), dtype=np.int)
            make_background(sequence)
            for figure in range(figures):
                # Kun den siste versjonen av labels blir beholdt og skrevet til fil
                labels_pos, labels_size = generate_movement_positionlabel(sequence,
                                                                          type="random",
                                                                          frames=frames,
                                                                          size_x=image_size,
                                                                          size_y=image_size,
                                                                          channels=channels)
            save_sequence_labelfile(sequence, labels_pos, labels_size, os.path.join(path, name), seq_number=sequence_index)


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    train_examples = 100000
    test_examples = 10000
    default_path = "../../Grafikk/stoybakgrunn"

    path = input("Mappe det skal skrives til (trykk enter for \"{0}\"): >".format(default_path))
    if path == "":
        path = default_path
    if not os.access(path, os.F_OK):
        raise IOError("%s er ikke en mappe." % path)
    create_train_test_examples(path,
                              [train_examples, test_examples],
                               figures=2)


if __name__ == "__main__":
    main()
