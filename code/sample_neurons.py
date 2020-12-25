# sample the neurons from the classification layer of an MLP
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

parser = argparse.ArgumentParser(description='Evaluate a smoothed classifier')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=1000, help = "numer of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--hidden_size", type=int, default=444, help="hidden size of mlp")
parser.add_argument('--nonlinear', default=0, type=int,
                    help="is the first hidden layer linear or non-linear")
parser.add_argument('--long', default=0, type=int,
                    help="do we use a four hidden layer MLP")
parser.add_argument('--noise_std_lst', nargs = '+', type = float, default=[], help='noise for each layer')
parser.add_argument('--layered_GNI', dest='layered_GNI', action='store_true')
parser.add_argument('--no_layered_GNI', dest='layered_GNI', action='store_false')
parser.set_defaults(feature=False)
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset, noise_std = args.noise_std_lst, hidden_size = args.hidden_size, nonlinear = args.nonlinear, long = args.long)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    
    
    NC = get_num_classes(args.dataset)
    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, NC, args.sigma)

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
        Bt = eig_vals
#         Bt = np.mean(eig_vals)
        Bd = gmean(radii)
    else:
        vf = np.vectorize(convert_to_relu)
    
    # prepare output file
    col_numbs = args.N
    f = open(args.outfile, 'w')
    label_string = ""
    for i in range(col_numbs - 1):
        label_string = label_string + str(i) + "\t"
    label_string = label_string + str(col_numbs - 1)
    print(label_string, file=f, flush=True)
        
    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only evaluate every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # evaluate the prediction of g around x
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
            Bt = eig_vals
            #Bt = np.mean(eig_vals)
            Bd = gmean(radii)
        prediction = smoothed_classifier.predict(x, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)
        
        counts = smoothed_classifier.see_sample_noise(x, args.N, args.batch, label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        #print(output_string.format(Bt), file=f, flush=True)
        print(*counts, sep="\t", file=f, flush = True)

    f.close()
