import numpy as np
import random
import scipy.misc
import os


def generate_movement_binarylabel(category, binary_type="horizontal-vertical", frames=12, sizeX=32, sizeY=32, channels=3):
    square_size = 1

    sequence = np.zeros((frames, sizeX, sizeY, channels), dtype=np.int)

    if binary_type == "horizontal-vertical":
        speedX = (category - 1) * random.random() * sizeX / 2 / frames
        speedY = category       * random.random() * sizeY / 2 / frames
        posX = sizeX / 2
        posY = sizeY / 2
    elif binary_type == "fast-slow_right":
        speedX = (sizeX / frames) * (1 + random.random() + category)
        speedY = 0
        posX = random.randint(0, sizeX-1)
        posY = sizeY / 2

    else:
        print("FEIL i sekvensgenerator.generate_binary_linear_movement: Ukjent binary_type: ", binary_type)
        return None

    for frame in range(frames):
        for widthIndex in range(square_size):
            for heightIndex in range(square_size):
                for channel in range(channels):
                    sequence[frame,
                             round(posY + heightIndex - square_size / 2) % sizeY,
                             round(posX  + widthIndex - square_size / 2) % sizeX,
                             channel] = 255
        posX += speedX
        posY += speedY

    return sequence


def generate_movement_positionlabel(type="random", frames=12, sizeX=32, sizeY=32, channels=3):
    square_size = 1

    sequence = np.zeros((frames, sizeX, sizeY, channels), dtype=np.int)
    labels = []

    # Initialisere
    if type == "random":
        posX = random.randint(0, sizeX-1)
        posY = random.randint(0, sizeY-1)
        speedX = random.gauss(0, 0.02*sizeX)
        speedY = random.gauss(0, 0.02*sizeY)
    else:
        raise ValueError("Ukjent verdi av 'type': {0}".format(type))

    for frame in range(frames):                     # For hvert bilde i sekvensen
        draw_rectangle(sequence, frame, posX, posY, square_size, channels, color=(255, 255, 255))      # Tegne inn firkant i bildet

        # Lagre label for dette bildet
        radius = square_size / 2
        labels.append((posX - radius, posY - radius, posX + radius, posY + radius))

        # Oppdatere posisjon og fart til neste tidssteg
        posX += speedX
        posY += speedY
        speedX += random.gauss(0, 0.03*sizeX)
        speedY += random.gauss(0, 0.03*sizeY)

        # Sørge for at firkanten holder seg innenfor bildet
        if posX < 0:
            posX = 0
            speedX = 0
        elif posX >= sizeX:
            posX = sizeX - 1
            speedX  = 0
        if posY < 0:
            posY = 0
            speedY = 0
        elif posY >= sizeY:
            posY = sizeY - 1
            speedY = 0

    return sequence, labels


def draw_rectangle(sequence, frame, posX, posY, square_size=4, channels=3, color=(255, 255, 255)):
    """
    Tegner et kvadrat i en numpy-array sequence på den gitte posisjonen.

    :param sequence: Sekvens av bilder (numpy-array)
    :param frame: Indeksen til bildet i sekvensen som skal endres
    :param posX: x-koordinaten til midten av rektangelet
    :param posY: y-koordinaten til midten av rektangelet
    :param square_size: sidelengden til kvadratet
    :param channels: antallet fargekanaler
    :param color: fargen til rektangelet, som tuppel med oppføring for hver fargekanal
    :return:
    """

    assert len(color) == channels

    orig_x = round(posX - square_size / 2)      # Venstre kant av rektangelet
    orig_y = round(posY - square_size / 2)      # Øvre kant av rektangelet
    for widthIndex in range(square_size):           # Iterere over bredden til firkanten
        current_pos_x = orig_x + widthIndex
        for heightIndex in range(square_size):      # Iterere over høyden til firkanten
            current_pos_y = orig_y + heightIndex
            for channel in range(channels):         # Iterere over fargekanalene
                try:
                    sequence[frame, current_pos_y, current_pos_x, channel] = color[channel]
                except IndexError:
                    pass


def save_sequence_binarylabel(sequence, parent_path, type):
    seq_number = 0
    while True:
        path_base = os.path.join(parent_path, "seq{0:05d}".format(seq_number))
        for i in range(2):
            if os.access(path_base + " " + str(i), os.F_OK):
                seq_number += 1
                break
        else:   # Dersom det ikke ble funnet en eksisterende mappe
            break
        continue   # Dersom det ble funnet en eksisterende mappe

    path = os.path.join(path_base + " " + type)
    os.mkdir(path)

    for frame in range(len(sequence)):
        image_array = sequence[frame]
        scipy.misc.imsave(os.path.join(path, "frame{0:05d}.jpg".format(frame)), image_array)


def save_sequence_labelfile(sequence, labels, parent_path):
    seq_number = 0
    while True:
        path = os.path.join(parent_path, "seq{0:05d}".format(seq_number))
        if os.access(path, os.F_OK):
            seq_number += 1
            continue
        break
    os.mkdir(path)

    # Lage label-fil
    label_file = open(os.path.join(path, "labels"), "w")

    # Iterere gjennom bildene i sekvensen
    for frame in range(len(sequence)):
        image_array = sequence[frame]
        file_name = "frame{0:05d}.jpg".format(frame)
        scipy.misc.imsave(os.path.join(path, file_name), image_array)

        # Skrive label til fil
        label_file.write(file_name + "," + ",".join(["{0:01f}".format(i) for i in labels[frame]]) + "\n")

    # Lukke label-fil
    label_file.close()


def create_train_test_examples(path, counts):
    frames = 12
    image_size = 32

    # type = bool(random.getrandbits(1))
    # sequence = generate_movement_binarylabel(type, binary_type="fast-slow_right", frames=frames, sizeX=image_size, sizeY=image_size)
    # save_sequence(sequence, "/home/mathias/inndata/generert/fast-slow_right/eksempel", "1" if type else "0")

    folder_names = ["train", "test", "examples"]

    for name in folder_names:
        try:
            os.mkdir(os.path.join(path, name))
        except IOError:
            print("Mappen \"{0}\" fins allerede.".format(os.path.join(path, name)))

        count = counts[folder_names.index(name)]
        for _ in range(count):
            sequence, labels = generate_movement_positionlabel(type="random",
                                                       frames=frames,
                                                       sizeX=image_size,
                                                       sizeY=image_size)
            save_sequence_labelfile(sequence, labels, os.path.join(path, name))



def main():
    create_train_test_examples("/home/mathias/inndata/generert/tilfeldig bevegelse/",
                              [1000, 1000, 24])


if __name__ == "__main__":
    main()