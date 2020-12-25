""" 
The purpose of this script is a sanity check for the radius of robustness for smoothing over the last hidden layer of a neural net
It simply check different random perturbation to the final hidden layer.
Such a perturbation is already built into the mnist_ll architecture
There should be 100% robustness in the region of robustness, else there is an error
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
parser.add_argument("results", type = str, help = "path to certify output we want to check")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=10000, help="number of samples to use when certifying")
parser.add_argument("--numb", type=int, default=1000, help="number of points to check in region of robustness")
parser.add_argument("--alpha", type=float, default=0.0001, help="failure probability")
parser.add_argument('--nonlinear', default=0, type=int,
                    help="are the layers linear or nonlinear")
parser.add_argument('--noise_std_lst', nargs = '+', type = float, default=[], help='noise for each layer')
args = parser.parse_args()

if __name__ == "__main__":
    # load the output of certify we are checking
    df = pd.read_csv(args.results, delimiter="\t")
    
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tpredict\tlabel\tradius\trcount\tacount\ttcount\ttime", file=f, flush=True)

    # iterate through the output of certify we are checking
    dataset = get_dataset(args.dataset, args.split)
    
    for i, row in df.iterrows():
        if row['predict'] != -1: #no radius of robustness for abstention
            before_time = time()
            index = row['idx']
            pred = row['predict']
            rad = row['radius'] #theoretical radius of robustness
            rcount = 0
            acount = 0
            for j in range(args.numb): 
                pert_dir = torch.rand(20) #hard code final hidden layer length of 20
                pert_len = torch.norm(pert_dir, p=2)
                u = rad * torch.rand(1)
                r = torch.pow(u,1/20) #hard code final hidden layer length of 20
                pert = r * pert_dir / pert_len
                pert = pert.cuda()
                #perturb the smoothed classifier
                base_classifier = get_architecture(checkpoint["arch"], args.dataset, noise_std = args.noise_std_lst, nonlinear = args.nonlinear, pert = pert)
                base_classifier.load_state_dict(checkpoint['state_dict'])
                smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)
                (x, label) = dataset[index]
                x = x.cuda()
                prediction = smoothed_classifier.predict(x, args.N, args.alpha, args.batch)
                if prediction == pred:
                    rcount = rcount + 1
                elif prediction == -1:
                    acount = acount + 1
            tcount = acount + rcount
            after_time = time()
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            # log the prediction and whether it was correct
            print("{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{}".format(index, pred, label, rad, rcount, acount, tcount, time_elapsed), file=f, flush=True)
    f.close()