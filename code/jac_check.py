# Checks the robustness returned for multilayer GNI
import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
import numpy as np
import pandas as pd

from utils import convert_to_relu

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("results", type = str, help = "path to certify output we want to check")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--N0", type=int, default=10000)
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--hidden_size", type=int, default=444, help="hidden size of mlp")
parser.add_argument('--nonlinear', default=0, type=int,
                    help="is the first hidden layer linear or non-linear")
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
    print("idx\tlabel\tpredict\trobust\ttime", file=f, flush=True)
    
    #compute spectrum of pullback noise
    tst_lst = [param.cpu().detach().numpy() for param in base_classifier.parameters()]
    A = tst_lst[0]
    hid_len, inp_len = np.shape(A)
    if not args.nonlinear:
        R = np.transpose(A) @ np.linalg.inv(A @ np.transpose(A))
        a,b,c = np.linalg.svd(R)
        eig_vals = np.zeros(inp_len)
        eig_vals[:hid_len] = (args.noise_std_lst[1] ** 2) * (b ** 2)
        eig_vals = eig_vals + (args.sigma ** 2)
        radii = np.sqrt(eig_vals)
        eig_vecs = a
    else:
        vf = np.vectorize(convert_to_relu)

    # iterate through the output of certify we are checking
    dataset = get_dataset(args.dataset, args.split)
    
    for i, row in df.iterrows():
        if row['correct']:
            before_time = time()
            index = row['idx']
            (x, label) = dataset[index]
            x = x.cuda()
            if args.nonlinear:
                #manually find Jacobian
                x.requires_grad_(True)
                mask = (base_classifier(x)[1] > 0).cpu().numpy()
                mask = vf(mask)
                J = A * mask[:, np.newaxis]
                R = np.transpose(J) @ np.linalg.inv(J @ np.transpose(J))
                a,b,c = np.linalg.svd(R)
                eig_vals = np.zeros(inp_len)
                eig_vals[:hid_len] = (args.noise_std_lst[1] ** 2) * (b ** 2)
                eig_vals = eig_vals + (args.sigma ** 2)
                radii = np.sqrt(eig_vals)
                eig_vecs = a
            pABar = row['pABar']
            pABar  = pABar - 0.00005 #to try to deal with rounding error
            idx = row['idx']
            # certify the prediction of g around x
            prediction, robust = smoothed_classifier.jac_certify_check(x, label, args.N0, args.alpha, args.batch, pABar, radii, eig_vecs)
            after_time = time()
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            print("{}\t{}\t{}\t{}\t{}".format(
                idx, label, prediction, robust, time_elapsed), file=f, flush=True)
    f.close()
