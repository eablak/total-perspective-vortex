import mne
import matplotlib.pyplot as plt
import matplotlib
import argparse
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from scipy.fft import fft, fftfreq, fftshift
import numpy as np

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
    input()



def load_data_mne(files, runs):

    path = eegbci.load_data(files, runs, path="../dataset/")

    raws = [read_raw_edf(file, preload=True) for file in path]
    for raw in raws:
        eegbci.standardize(raw)

    data = concatenate_raws(raws)

    return data


def visualize(raw):

    matplotlib.use('TkAgg')

    # raw.plot()
    # raw.compute_psd().plot()
    # plt.show()


def filtered(raw):

    filtered = raw.copy()
    filtered.filter(8, 30)
    
    # filtered.compute_psd().plot()
    # plt.show()

    return filtered


def event_epoch(raw):

    events, event_id = mne.events_from_annotations(raw)
    task_events = {'T1': event_id['T1'], 'T2': event_id['T2']}

    # try with tmax=2&4 also
    epochs = mne.Epochs(raw, events, task_events, tmin=0, tmax=2, baseline=None, preload=True)
    
    data = epochs.get_data()  # X (epochs, channels, time)
    labels = epochs.events[:, -1] # y (epochs)

    return data, labels

