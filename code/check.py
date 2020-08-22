""" This script loads a base classifier and then runs PREDICT on perturbations of a single example from the dataset to check the returned radius of robustness for that single example
"""
import argparse
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
from architectures import get_architecture
import datetime

import pandas as pd

parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("results", type = str, help = "path to ceritfy output we want to check")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--noise_std_lst', nargs = '+', type = float, default=[], help='noise for each layer')
args = parser.parse_args()

def check_robust(classifier, x, rad, label):
    """
    Checks manually whether the prediction of a classifier on an input x is robust within a given radius
    """
    robust = 1
    before_time = time()
    for i in range(len(x)):

        y = x.clone()
        y[i] = x[i] + rad

        # make the prediction
        prediction = classifier.predict(y, args.N, args.alpha, args.batch)

        correct = int(prediction == label)
        robust = robust * correct
    after_time = time()
    time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
    return robust, time_elapsed
    
    

if __name__ == "__main__":
    # load the output of certify we are checking
    df = pd.read_csv(args.results, delimiter="\t")
    
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset, noise_std = args.noise_std_lst)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smoothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tradius\trobust\ttime", file=f, flush=True)

    # iterate through the output of certify we are checking
    dataset = get_dataset(args.dataset, args.split)
    
    for _, row in df.iterrows():
        if row['correct']:
            index = row['idx']
            rad = row['radius']
            (x, label) = dataset[index]
            x = x.cuda()
            rob, time_elapsed = check_robust(smoothed_classifier, x, rad, label)
            # log the prediction and whether it was correct
            print("{}\t{}\t{:.3}\t{}\t{}".format(index, label, rad, rob, time_elapsed), file=f, flush=True)
    f.close()