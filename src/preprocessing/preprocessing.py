import mne
import matplotlib.pyplot as plt
import matplotlib
import argparse
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from scipy.fft import fft, fftfreq, fftshift
import numpy as np
import sys
import os



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


def load_data_mne(subjects, runs, folder_path="../dataset/"):

    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    path = eegbci.load_data(subjects, runs, path=folder_path)

    raws = [read_raw_edf(subject, preload=True) for subject in path]
    for raw in raws:
        eegbci.standardize(raw)

    data = concatenate_raws(raws)

    return data


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

    epochs = mne.Epochs(raw, events, task_events, tmin=0, tmax=4, baseline=None, preload=True)
    
    data = epochs.get_data()
    labels = epochs.events[:, -1]

    return data, labels



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', nargs='+', type=int, help="Folder Navigation")
    parser.add_argument('-r', nargs='+', type=int, help="Experimental Runs")

    args = parser.parse_args()

    subjects = args.f
    runs = args.r

    raw = load_data_mne(subjects, runs)
    data_infos(raw)

    visualize(raw)
    filtered_raw = filtered(raw, True)

    data, labels = event_epoch(filtered_raw)
    epoched_data_infos(data, labels)
