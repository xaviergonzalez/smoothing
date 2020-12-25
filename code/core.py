import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint

from scipy.stats import gmean

from time import time

import math

from utils import flex_tuple, convert_to_relu
from find_jacobian import find_jacobian


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius, lower bound on class probability)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0, pABar
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius, pABar
        
    def certify_flayer(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, sigma) -> (int, float, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius IN THE LAST HIDDEN LAYER
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :param sigma: noise applied IN THE LAST HIDDEN LAYER
        :return: (predicted class, certified radius, lower bound on class probability)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0, nA
        else:
            radius = sigma * norm.ppf(pABar)
            return cAHat, radius, nA
        
    def jac_certify_check(self, x: torch.tensor, label: int, n: int, alpha: float, batch_size: int, pABar: float, radii, eig_vecs) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        Examines whether some of the points on the boundary of the ellipsoid do in fact return the same prediction

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :param noise_std_lst: lst of noise to apply to each layer (put 0 for data layer)
        :param nonlinear: boolean to indicate whether our classifier has a nonlinear first layer
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        x = x.detach().cpu().numpy()
        if pABar < 0.5:
            return Smooth.ABSTAIN, Smooth.ABSTAIN
        else:
            radii = radii * norm.ppf(pABar)
            robust = 1
            for i in range(len(x)):
                y = np.copy(x)
                y = x + radii[i] * eig_vecs[:, i]
                y = y.astype(np.float32)
                y = torch.from_numpy(y)
                y = y.cuda()
                # make the prediction
                prediction = self.predict(y, n, alpha, batch_size)
                correct = int(prediction == label)
                robust = robust * correct
            return label, robust

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    #need to think about how to rework to be compatible with MLP...
    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

#                 batch = x.repeat((this_batch_size, 1, 1, 1))
                batch = x.repeat((this_batch_size, 1))
#                 print(np.shape(batch))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
#                 print(noise)
                predictions = flex_tuple(self.base_classifier(batch + noise)).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts
        
            #need to think about how to rework to be compatible with MLP...
    def see_sample_noise(self, x: torch.tensor, num: int, batch_size, label) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :param label: label of the input
        :return: an ndarray[float] of the different logit outputs for the correct label
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

#                 batch = x.repeat((this_batch_size, 1, 1, 1))
                batch = x.repeat((this_batch_size, 1))
#                 print(np.shape(batch))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
#                 print(noise)
                predictions = flex_tuple(self.base_classifier(batch + noise))
                predictions = predictions.cpu().numpy()
                predictions = predictions[:,label]
                predictions = np.around(predictions,3)
            return predictions

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
