from matplotlib import pyplot as plt
import numpy as np


def create_timing_graph(timing_array):
    x = np.arange(1, timing_array.shape[0]+1)
    plt.plot(x, timing_array)
    plt.title('Elapsed Time')
    plt.xlabel('# of Frame')
    plt.ylabel('Time, msec')
    plt.grid(True)
    plt.xticks(x)
    plt.savefig('timing_report.png')


def save_results(results, dest):
    with open(dest, 'w') as file:
        [print(f"{row[0]}, {row[1]}", file=file) for row in results]
