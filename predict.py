import pickle
import os
import sys
import joblib
import argparse
from train import handle_preprocessing
import mne
import numpy as np
import time



def load_model():
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = base_path + "/saved_model.pkl"

    cl = joblib.load(model_path)
    return cl


def predict(loaded_model, parser):

    epochs, y = handle_preprocessing(parser)
    
    scores = []
    print(f"epoch nb:\t[prediction] [truth] equeal?")
    for n in range(epochs.shape[0]):
        pred = loaded_model.predict(epochs[n:n+1, :, :])
        print(f"epoch {n:02d}:\t   {pred}\t\t{y[n: n+1]} {'True' if pred==y[n: n+1] else 'False'}")
        scores.append(pred[0] == y[n])
        time.sleep(2)

    print(f"Mean acc= {np.mean(scores)}")
    


if __name__ == "__main__":
    
    loaded_model = load_model()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', nargs='+', type=int, help="Folder Navigation")
    parser.add_argument('-r', nargs='+', type=int, help="Experimental Runs")

    predict(loaded_model, parser)