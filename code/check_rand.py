""" This script loads a base classifier and then runs PREDICT on perturbations  to check the returned radius of robustness for that single example
Ideally you would implement a constrained adversarial attack instead...
"""
import argparse
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
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=10000, help="number of samples to use when certifying")
parser.add_argument("--numb", type=int, default=1000, help="number of points to check in each ellipsoid of robustness")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--hidden_size", type=int, default=444, help="hidden size of mlp")
parser.add_argument('--nonlinear', default=0, type=int,
                    help="is the first hidden layer linear or non-linear")
parser.add_argument('--noise_std_lst', nargs = '+', type = float, default=[], help='noise for each layer')
args = parser.parse_args()

def check_robust(classifier, x, rad, label, numb):
    """
    Checks manually whether the prediction of a classifier on an input x is robust within a given radius
    param numb: number of samples within the radius of robustness to check
    """
    robust = 1
    before_time = time()
    dim = len(x)
    #you could probably do this in larger batch sizes...
    for i in range(numb):
        #sample uniformly from radius of sphere
        y = x.clone()
        pert_dir = torch.randn_like(y)
        pert_len = torch.norm(pert_dir, p=2)
        u = torch.rand(1)
        r = torch.pow(u,1/dim)
        r = r.cuda()
        pert = r * pert_dir / pert_len
        y = y + pert
        
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
    base_classifier = get_architecture(checkpoint["arch"], args.dataset, noise_std = args.noise_std_lst, hidden_size = args.hidden_size, nonlinear = args.nonlinear)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smoothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tradius\trobust\ttime", file=f, flush=True)

    # iterate through the output of certify we are checking
    dataset = get_dataset(args.dataset, args.split)
    
    for i, row in df.iterrows():
        if row['correct']:
            index = row['idx']
            rad = row['radius']
            (x, label) = dataset[index]
            x = x.cuda()
            rob, time_elapsed = check_robust(smoothed_classifier, x, rad, label, args.numb)
            # log the prediction and whether it was correct
            print("{}\t{}\t{:.3}\t{}\t{}".format(index, label, rad, rob, time_elapsed), file=f, flush=True)
    f.close()