# Checks the robustness for a single case, for the minimum radius
import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
from scipy.stats import norm, binom_test
import numpy as np
import pandas as pd

from utils import convert_to_relu

parser = argparse.ArgumentParser(description='Check the coordinates of a certified example')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("results", type = str, help = "path to ceritfy output we want to check")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=10000)
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--hidden_size", type=int, default=444, help="hidden size of mlp")
parser.add_argument('--nonlinear', default=0, type=int,
                    help="is the first hidden layer linear or non-linear")
parser.add_argument('--ID', default=0, type=int,
                    help="which example to check")
parser.add_argument('--noise_std_lst', nargs = '+', type = float, default=[], help='noise for each layer')
args = parser.parse_args()

if __name__ == "__main__":
    
    # load the output of certify we are checking
    df = pd.read_csv(args.results, delimiter="\t")
    
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset, noise_std = args.noise_std_lst, hidden_size = args.hidden_size, nonlinear = args.nonlinear)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("coord\tlabel\tpredict\trobust\ttime", file=f, flush=True)

    # iterate through the single example
    dataset = get_dataset(args.dataset, args.split)
    
    ID = args.ID
    rad = (df.loc[df['idx'] == ID]['radius'].values[0])
    
    (x, label) = dataset[ID]
    x = x.cuda()
    
    for i in range(len(x)):
        before_time = time()
        y = x.clone()
        y[i] = x[i] + rad
        # make the prediction
        prediction = smoothed_classifier.predict(y, args.N0, args.alpha, args.batch)
        correct = int(prediction == label)
        after_time = time()
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{}\t{}".format(i, label, prediction, correct, time_elapsed), file=f, flush=True)
    f.close()
