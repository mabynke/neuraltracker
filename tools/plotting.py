import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os


def plot_loss_history(loss_history, run_name):
    array = np.array(loss_history)
    colors = [["#0000ff", "#6060e0", "#6060c0"],
              ["#ff0000", "#e06060", "#c06060"]]
    for i in range(2):  # Trening og test
        # for j in range(3):  # loss, loss_pos, loss_str
        plt.plot(array[:, i, 0], colors[i][0])

    plt.clf()
    plt.xlabel("Runder")
    plt.ylabel("mean_squared_error")
    plt.title("Loss for trening (blå) og test (rød)")
    plt.grid(True)
    plt.savefig(os.path.join("saved_plots", run_name + ".png"))


def main():
    dummy_loss_history = [((i, i+1, i-1), (2*i, 2*i+1, 2*i-1)) for i in range(18)]
    plot_loss_history(dummy_loss_history)


if __name__ == "__main__":
    main()