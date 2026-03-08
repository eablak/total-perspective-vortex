import mne
import matplotlib.pyplot as plt
import sys
import os
import matplotlib

def filtered(raw):

    filtered = raw.copy()
    filtered.filter(8, 30)
    
    filtered.compute_psd().plot()
    plt.show()


def visualize(raw):

    matplotlib.use('TkAgg')

    raw.plot()
    raw.compute_psd().plot()
    plt.show()


if __name__ == "__main__":

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = base_path + "/dataset/S047R01.edf"

    raw = mne.io.read_raw_edf(dataset_path, preload=True)

    visualize(raw)
    filtered_raw = filtered(raw)



