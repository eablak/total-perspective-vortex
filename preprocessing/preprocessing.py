import mne
import matplotlib.pyplot as plt
import matplotlib
import argparse
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws 


def load_data_mne(files, runs):

    path = eegbci.load_data(files, runs, path="../dataset/")

    raws = [read_raw_edf(file, preload=True) for file in path]
    for raw in raws:
        eegbci.standardize(raw)

    data = concatenate_raws(raws)

    return data


def visualize(raw):

    matplotlib.use('TkAgg')

    raw.plot()
    raw.compute_psd().plot()
    plt.show()


def filtered(raw):

    filtered = raw.copy()
    filtered.filter(8, 30)
    
    filtered.compute_psd().plot()
    plt.show()
    
    return filtered


def event_epoch(raw):

    events, event_id = mne.events_from_annotations(raw)
    task_events = {'T1': event_id['T1'], 'T2': event_id['T2']}

    # try with tmax=2&4 also
    epochs = mne.Epochs(raw, events, task_events, tmin=0, tmax=2, baseline=None, preload=True)
    
    data = epochs.get_data()  # X (epochs, channels, time)
    labels = epochs.events[:, -1] # y (epochs)

    return data, labels


def feature_extraction(data):
    pass




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', nargs='+', type=int, help="Folder Navigation")
    parser.add_argument('-r', nargs='+', type=int, help="Experimental Runs")

    args = parser.parse_args()
    data = load_data_mne(args.f, args.r)

    visualize(data)
    filtered_data = filtered(data)

    data, labels = event_epoch(filtered_data)
    feature_extraction(data)
