import argparse
from preprocessing.preprocessing import load_data_mne, visualize, filtered, event_epoch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

from sklearn.model_selection import train_test_split



def handle_preprocessing(parser):
    
    args = parser.parse_args()
    data = load_data_mne(args.f, args.r)

    visualize(data)
    filtered_data = filtered(data)

    data, labels = event_epoch(filtered_data)

    return data, labels


def model(X_raw, y):

    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, X_raw, y, cv=cv, n_jobs=1)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                            class_balance))

    # plot CSP patterns estimated on full data for visualization

    # csp.plot_patterns(data.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y, test_size=0.20, random_state=None, stratify=y
    )
    clf.fit(X_train, y_train)
    val_score = clf.score(X_val, y_val)
    print(f"Validation Accuracy: {val_score:.4f}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', nargs='+', type=int, help="Folder Navigation")
    parser.add_argument('-r', nargs='+', type=int, help="Experimental Runs")

    data, labels = handle_preprocessing(parser)
    model(data, labels)