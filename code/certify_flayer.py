# evaluate a smoothed classifier and give radius of robustness ON THE FINAL HIDDEN LAYER
import argparse
import os
# import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
import numpy as np

from utils import convert_to_relu

from scipy.stats import gmean

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--nonlinear', default=0, type=int,
                    help="are non-linearities applied?")
parser.add_argument('--noise_std_lst', nargs = '+', type = float, default=[], help='noise for each layer')
parser.set_defaults(feature=False)
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset, noise_std = args.noise_std_lst, nonlinear = args.nonlinear)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\tpABar\tAt\tAd\ttime", file=f, flush=True)
    
    #compute specturm of 2nd hidden layer accumulated noise: linear approx
    param_lst = [param.cpu().detach().numpy() for param in base_classifier.parameters()]
    W1 = param_lst[0] #weight matrix from input to first hidden layer
    print(np.shape(W1))
    W2 = param_lst[1] #weight matrix from input to second hidden layer
    print(np.shape(W2))
    inter = (args.noise_std_lst[0] ** 2) * W2 @ W1 @ np.transpose(W1) @ np.transpose(W2) + (args.noise_std_lst[1] ** 2) * W2 @ np.transpose(W2)
    u,d,v = np.linalg.svd(inter)
    eig_vals = d + (args.noise_std_lst[2] ** 2)
    radii = np.sqrt(eig_vals)
    At = np.mean(eig_vals)
    Ad = gmean(radii)
        
    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius, pABar = smoothed_classifier.certify_flayer(x, args.N0, args.N, args.alpha, args.batch, args.noise_std_lst[2])
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}\t{:.2}\t{:.2}\t{}".format(
            i, label, prediction, radius, correct, pABar, At, Ad, time_elapsed), file=f, flush=True)

    f.close()
