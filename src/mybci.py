import argparse
import sys
from preprocessing.preprocessing import load_data_mne, visualize, filtered, data_infos, event_epoch, epoched_data_infos
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold, StratifiedKFold

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

from sklearn.model_selection import train_test_split
import joblib
import pickle
import time
import os
# from ft_csp.ft_csp import CSP



def handle_preprocessing(subject, task):
    
    raw = load_data_mne(subject, task, "dataset")
    filtered_raw = filtered(raw, False)
    data, labels = event_epoch(filtered_raw)
    
    return data, labels


def save_model_datasets(clf, X_test, y_test, subject, task):
    
    if not os.path.exists("model_results"):
        os.makedirs("model_results")

    joblib.dump(clf, f"model_results/saved_model{subject}_{task}.pkl")

    with open(f'model_results/X_test{subject}_{task}.pkl', 'wb') as file:
        pickle.dump(X_test, file)

    with open(f'model_results/y_test{subject}_{task}.pkl', 'wb') as file:
        pickle.dump(y_test, file)


def train(subject, task):

    X_raw, y = handle_preprocessing(subject, task)
    # print(f"Total epochs: {X_raw.shape[0]}")
    print(X_raw.shape)
    input()


    sf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lda = LinearDiscriminantAnalysis()
    # csp = CSP()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    clf = Pipeline([('CSP', csp), ('LDA', lda)])

    X_main, X_test, y_main, y_test = train_test_split(X_raw, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=(0.15 / 0.85), random_state=42)

    cross_val_results = cross_val_score(clf, X_train, y_train, cv=sf, n_jobs=1)

    clf.fit(X_train, y_train)
    val_score = clf.score(X_val, y_val)
    
    print(f"Validation Accuracy: {val_score:.4f}")
    print(f"{cross_val_results}\ncross_val_score: {cross_val_results.mean()}")

    save_model_datasets(clf, X_test, y_test, subject, task)


def predict(subject, task):

    try:
        
        loaded_model = joblib.load(f"model_results/saved_model{subject}_{task}.pkl")

        with open(f'model_results/X_test{subject}_{task}.pkl', 'rb') as f:
            X_test = pickle.load(f)
        
        with open(f'model_results/y_test{subject}_{task}.pkl', 'rb') as f:
            y_test = pickle.load(f)

        scores = []
        print(f"epoch nb:\t[prediction] [truth] equal?")
        for n in range(X_test.shape[0]):
            pred = loaded_model.predict(X_test[n:n+1, :, :])
            print(f"epoch {n:02d}:\t   {pred}\t\t{y_test[n: n+1]} {'True' if pred[0]==y_test[n] else 'False'}")
            scores.append(pred[0] == y_test[n])
            # time.sleep(2)

        print(f"Mean acc= {np.mean(scores)}")

    except OSError as e:
        print("Train model first! ",e)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', nargs='+', type=int, help="Folder Navigation")
    parser.add_argument('-r', nargs='+', type=int, help="Experimental Runs")
    parser.add_argument('-p', nargs='+', type=str, help="Process")

    args = parser.parse_args()

    subjects = args.f
    runs = args.r
    process = args.p[0]

    if process == "train":
        train(subjects, runs)
    elif process == "predict":
        predict(subjects, runs)
