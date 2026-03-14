import mne
import matplotlib.pyplot as plt
import matplotlib
import argparse
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from scipy.fft import fft, fftfreq, fftshift
import numpy as np
import sys



def data_infos(data):
    
    print("\n" ,"*" * 40, "\n")
    print(f"Data Type: {type(data)}")
    print(f"Data Infos: {data.info}\n")
    print(f"Frequency: {data.info['sfreq']}")
    print(f"Channels: {data.info['nchan']}\n")
    print(f"Start and end time in seconds: {data.times[[0, -1]]}\n")
    print(f"n_channels, n_timepoints {data.get_data().shape}\n")
    print(f"Channel names: {data.ch_names}\n")
    print("\n", "*" * 40, "\n\n")


def epoched_data_infos(data, labels):
    
    print("\n" ,"*" * 40, "\n")
    print(f"Data Type: {type(data)}")
    print(f"Data Shape: {data.shape}\n") # trial × channel × time

    print(f"Labels Type: {type(labels)}")
    print(f"Labels Shape: {labels.shape}\n")

    print(f"Unique class codes: {np.unique(labels)}")
    print(f"Class balance: {np.bincount(labels)}") # label values e.g. 28 left hand 32 right hand
    print("\n" ,"*" * 40, "\n\n")


def load_data_mne(file, run):

    path = eegbci.load_data(file, run, path="../../dataset/")

    raw = read_raw_edf(path[0], preload=True)
    eegbci.standardize(raw)

    return raw


def visualize(raw):

    matplotlib.use('TkAgg')

    raw.plot()
    raw.compute_psd().plot()
    plt.show()


def filtered(raw, flag):

    filtered = raw.copy()
    filtered.filter(8, 30)
    
    if flag == 1:
        filtered.compute_psd().plot()
        plt.show()

    return filtered


def event_epoch(raw):

    events, event_id = mne.events_from_annotations(raw)
    task_events = {'T1': event_id['T1'], 'T2': event_id['T2']}

    # try with tmax=2&4 also
    epochs = mne.Epochs(raw, events, task_events, tmin=0, tmax=2, baseline=None, preload=True)
    
    data = epochs.get_data()
    labels = epochs.events[:, -1]

    return data, labels



if __name__ == "__main__":

    if (len(sys.argv) != 3):
        sys.exit("Use it with subject and run args (e.g python preprocessing.py 4 14) ")

    subject = int(sys.argv[1])
    run = int(sys.argv[2])

    raw = load_data_mne(subject, run)
    data_infos(raw)

    visualize(raw)
    filtered_raw = filtered(raw, True)

    data, labels = event_epoch(filtered_raw)
    epoched_data_infos(data, labels)
