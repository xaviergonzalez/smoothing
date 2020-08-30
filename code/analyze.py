import numpy as np
import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import *
import pandas as pd
import seaborn as sns
import math

sns.set()


class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()


class HighProbAccuracy(Accuracy):
    def __init__(self, data_file_path: str, alpha: float, rho: float):
        self.data_file_path = data_file_path
        self.alpha = alpha
        self.rho = rho

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (mean - self.alpha - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
                - math.log(1 / self.rho) / (3 * num_examples))


class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("radius", fontsize=16)
    plt.ylabel("certified accuracy", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def smallplot_certified_accuracy(outfile: str, title: str, max_radius: float,
                                 methods: List[Line], radius_step: float = 0.01, xticks=0.5) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for method in methods:
        plt.plot(radii, method.quantity.at_radii(radii), method.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.xlabel("radius", fontsize=22)
    plt.ylabel("certified accuracy", fontsize=22)
    plt.tick_params(labelsize=20)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(xticks))
    plt.legend([method.legend for method in methods], loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.close()


def latex_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                   methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')

    for radius in radii:
        f.write("& $r = {:.3}$".format(radius))
    f.write("\\\\\n")

    f.write("\midrule\n")

    for i, method in enumerate(methods):
        f.write(method.legend)
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = r" & \textbf{" + "{:.2f}".format(accuracies[i, j]) + "}"
            else:
                txt = " & {:.2f}".format(accuracies[i, j])
            f.write(txt)
        f.write("\\\\\n")
    f.close()


def markdown_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                      methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')
    f.write("|  | ")
    for radius in radii:
        f.write("r = {:.3} |".format(radius))
    f.write("\n")

    f.write("| --- | ")
    for i in range(len(radii)):
        f.write(" --- |")
    f.write("\n")

    for i, method in enumerate(methods):
        f.write("<b> {} </b>| ".format(method.legend))
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = "{:.2f}<b>*</b> |".format(accuracies[i, j])
            else:
                txt = "{:.2f} |".format(accuracies[i, j])
            f.write(txt)
        f.write("\n")
    f.close()


if __name__ == "__main__":
    #adding more hidden layer noise for fixed data noise
    input_noise = ["0.12", "0.25"]
    hlayer_noise = ["0", "0.12", "0.25", "0.5"]
    HSIZE = 44
    h_size = str(HSIZE)
    for i in input_noise:
        file_mean = "mnist_results/linear/hlayer" + h_size + "/mean/train-" + i + "-"
        plot_certified_accuracy(
            "TST/plots/mnist_" + i + "_same" + "_hlayer" + h_size, "Volume of robustness ellipsoid", 2, [
                Line(ApproximateAccuracy(file_mean + s + "/test-" + i + "-" + s), "hidden test noise: $\sigma_h$ =" + s) for s in hlayer_noise
            ])
        file_min = "mnist_results/linear/hlayer" + h_size + "/min/train-" + i + "-"
        plot_certified_accuracy(
            "TST/plots/MIN_mnist_" + i + "_same" + "_hlayer" + h_size , "Minimum radius of robustness", 1.5, [
                Line(ApproximateAccuracy(file_min + s + "/test-" + i + "-" + s), "hidden test noise: $\sigma_h$ =" + s) for s in hlayer_noise
            ])
#         # how much does performance actually change when we adjust model training
#         file_mean = "mnist_results/linear/hlayer444/mean/train-" + i + "-"
#         plot_certified_accuracy(
#             "TST/plots/mnist_" + i + "_tdif", "Volume of robustness ellipsoid", 2, [
#                 Line(ApproximateAccuracy(file_mean + s + "/test-" + i + "-" + i), "hidden train noise: $\sigma_h$ =" + s) for s in hlayer_noise
#             ])
#         file_min = "mnist_results/linear/hlayer444/min/train-" + i + "-"
#         plot_certified_accuracy(
#             "TST/plots/MIN_mnist_" + i + "_tdif", "Minimum radius of robustness", 1.5, [
#                 Line(ApproximateAccuracy(file_min + s + "/test-" + i + "-" + i), "hidden train noise: $\sigma_h$ =" + s) for s in hlayer_noise
#             ])
#     # performance for same pullback noise
#     file = "same_pb/mnist/linear/hlayer444/0.25/train-0.25-0.25/"
#     file2 = "same_pb/mnist/linear/hlayer444/0.25/train-"
#     noise_arr = ["test-0.12-0.27",  "test-0.25-0.25",  "test-0.5-0.21",  "test-1-0.07"]
#     noise_arr2 = ["0.12-0.27",  "0.25-0.25",  "0.5-0.21",  "1-0.07"]
#     latex_table_certified_accuracy(
#         "TST/latex/mnist_spb", 0, 0.5, 0.1, [
#             Line(ApproximateAccuracy(file + "mean/" + a), a) for a in noise_arr
#         ])
#     plot_certified_accuracy(
#         "TST/plots/mnist_spb", "MNIST, same det of pb noise, volume of ellipsoid of robustness, tr 0.25 0.25$", 1.5, [
#             Line(ApproximateAccuracy(file + "mean/" + a), a) for a in noise_arr
#         ])
#     latex_table_certified_accuracy(
#         "TST/latex/MIN_mnist_spb", 0, 0.5, 0.1, [
#             Line(ApproximateAccuracy(file + "min/" + a), a) for a in noise_arr
#         ])
#     plot_certified_accuracy(
#         "TST/plots/MIN_mnist_spb", "MNIST, same det of pb noise, minimum radius of robustness, tr 0.25 0.25$", 1.5, [
#            Line(ApproximateAccuracy(file + "min/" + a), a) for a in noise_arr
#         ])
    # performance when tested against model trained in same way
#     latex_table_certified_accuracy(
#         "TST/latex/mnist_spb_same", 0, 0.5, 0.1, [
#             Line(ApproximateAccuracy(file2 + a2 + "/mean/test-" + a2), a2) for a2 in noise_arr2
#         ])
#     plot_certified_accuracy(
#         "TST/plots/mnist_spb_same", "Vol", 1.5, [
#             Line(ApproximateAccuracy(file2 + a2 + "/mean/test-" + a2), a2) for a2 in noise_arr2
#         ])
#     latex_table_certified_accuracy(
#         "TST/latex/MIN_mnist_spb_same", 0, 0.5, 0.1, [
#             Line(ApproximateAccuracy(file2 + a2 + "/min/test-" + a2), a2) for a2 in noise_arr2
#         ])
#     plot_certified_accuracy(
#         "TST/plots/MIN_mnist_spb_same", "Min rad", 1.5, [
#            Line(ApproximateAccuracy(file2 + a2 + "/min/test-" + a2), a2) for a2 in noise_arr2
#         ])
#     #high probability (same shape)
#     latex_table_certified_accuracy(
#         "TST/latex/mnist_spb_hp", 0, 0.5, 0.1, [
#             Line(HighProbAccuracy(file + "mean/" + a, 0.001, 0.001), a) for a in noise_arr
#         ])
#     plot_certified_accuracy(
#         "TST/plots/mnist_spb_hp", "MNIST, same det of pb noise, volume of ellipsoid of robustness, tr 0.25 0.25$", 1.5, [
#             Line(HighProbAccuracy(file + "mean/" + a, 0.001, 0.001), a) for a in noise_arr
#         ])
#     latex_table_certified_accuracy(
#         "TST/latex/MIN_mnist_spb_hp", 0, 0.5, 0.1, [
#             Line(HighProbAccuracy(file + "min/" + a, 0.001, 0.001), a) for a in noise_arr
#         ])
#     plot_certified_accuracy(
#         "TST/plots/MIN_mnist_spb_hp", "MNIST, same det of pb noise, minimum radius of robustness, tr 0.25 0.25$", 1.5, [
#            Line(HighProbAccuracy(file + "min/" + a, 0.001, 0.001), a) for a in noise_arr
#         ])
#     # comparing approximate and high probability accuracy
#     plot_certified_accuracy(
#         "TST/plots/approx_vs_high_prob", "MNIST, same det of pb noise, minimum radius of robustness, tr 0.25 0.25$", 1.5, [
#            Line(ApproximateAccuracy(file + "mean/" + "test-0.12-0.27"), "approx"),
#            Line(HighProbAccuracy(file + "mean/" + "test-0.12-0.27", 0.001, 0.001), "high prob"),
#         ])
#     latex_table_certified_accuracy(
#         "TST/latex/mnist_012", 0, 0.5, 0.1, [
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_012_0"), "no hidden layer noise"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_012_012"), "hidden layer noise: $\sigma = 0.12$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_012_025"), "hidden layer noise: $\sigma = 0.25$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_012_050"), "hidden layer noise: $\sigma = 0.50$"),
#         ])
#     plot_certified_accuracy(
#         "TST/plots/mnist_012", "MNIST, data noised with $\sigma = 0.12$", 1.5, [
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_012_0"), "no hidden layer noise"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_012_012"), "hidden layer noise: $\sigma = 0.12$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_012_025"), "hidden layer noise: $\sigma = 0.25$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_012_050"), "hidden layer noise: $\sigma = 0.50$"),
#         ])
#     latex_table_certified_accuracy(
#         "TST/latex/mnist_025", 0, 0.5, 0.1, [
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_025_0"), "no hidden layer noise"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_025_012"), "hidden layer noise: $\sigma = 0.12$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_025_025"), "hidden layer noise: $\sigma = 0.25$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_025_05"), "hidden layer noise: $\sigma = 0.50$"),
#         ])
#     plot_certified_accuracy(
#         "TST/plots/mnist_025", "MNIST, data noised with $\sigma = 0.25$", 1.5, [
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_025_0"), "no hidden layer noise"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_025_012"), "hidden layer noise: $\sigma = 0.12$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_025_025"), "hidden layer noise: $\sigma = 0.25$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_025_05"), "hidden layer noise: $\sigma = 0.50$"),
#         ])
#     latex_table_certified_accuracy(
#         "TST/latex/MIN_mnist_025", 0, 0.5, 0.1, [
#             Line(ApproximateAccuracy("MIN_certification_noise_layer_025_0"), "no hidden layer noise"),
#             Line(ApproximateAccuracy("MIN_certification_noise_layer_025_012"), "hidden layer noise: $\sigma = 0.12$"),
#             Line(ApproximateAccuracy("MIN_certification_noise_layer_025_025"), "hidden layer noise: $\sigma = 0.25$"),
#             Line(ApproximateAccuracy("MIN_certification_noise_layer_025_05"), "hidden layer noise: $\sigma = 0.50$"),
#         ])
#     plot_certified_accuracy(
#         "TST/plots/MIN_mnist_025", "MNIST, data noised with $\sigma = 0.25$", 1.5, [
#             Line(ApproximateAccuracy("MIN_certification_noise_layer_025_0"), "no hidden layer noise"),
#             Line(ApproximateAccuracy("MIN_certification_noise_layer_025_012"), "hidden layer noise: $\sigma = 0.12$"),
#             Line(ApproximateAccuracy("MIN_certification_noise_layer_025_025"), "hidden layer noise: $\sigma = 0.25$"),
#             Line(ApproximateAccuracy("MIN_certification_noise_layer_025_05"), "hidden layer noise: $\sigma = 0.50$"),
#         ])
#     latex_table_certified_accuracy(
#         "TST/latex/mnist_050", 0, 0.5, 0.1, [
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_05_0"), "no hidden layer noise"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_05_012"), "hidden layer noise: $\sigma = 0.12$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_05_025"), "hidden layer noise: $\sigma = 0.25$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_05_05"), "hidden layer noise: $\sigma = 0.50$"),
#         ])
#     plot_certified_accuracy(
#         "TST/plots/mnist_050", "MNIST, data noised with $\sigma = 0.5$", 1.5, [
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_05_0"), "no hidden layer noise"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_05_012"), "hidden layer noise: $\sigma = 0.12$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_05_025"), "hidden layer noise: $\sigma = 0.25$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_05_05"), "hidden layer noise: $\sigma = 0.50$"),
#         ])
#     latex_table_certified_accuracy(
#         "TST/latex/comparing_noise", 0, 0.5, 0.1, [
#             Line(ApproximateAccuracy("certification_baseline"), "data: $\sigma = 0.25$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_0_025"), "hidden layer noise: $\sigma = 0.25$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_0_075"), "hidden layer noise: $\sigma = 0.75$"),
#         ])
#     plot_certified_accuracy(
#         "TST/plots/comparing noise", "Comparing the effect of noise on different layers$", 1.5, [
#             Line(ApproximateAccuracy("certification_baseline"), "data: $\sigma = 0.25$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_0_025"), "hidden layer noise: $\sigma = 0.25$"),
#             Line(ApproximateAccuracy("MEAN_certification_noise_layer_0_075"), "hidden layer noise: $\sigma = 0.75$"),
#         ])
#     latex_table_certified_accuracy(
#         "analysis/latex/vary_noise_cifar10", 0.25, 1.5, 0.25, [
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
#         ])
#     markdown_table_certified_accuracy(
#         "analysis/markdown/vary_noise_cifar10", 0.25, 1.5, 0.25, [
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "&sigma; = 0.12"),
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "&sigma; = 0.25"),
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "&sigma; = 0.50"),
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "&sigma; = 1.00"),
#         ])
#     latex_table_certified_accuracy(
#         "analysis/latex/vary_noise_imagenet", 0.5, 3.0, 0.5, [
#             Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
#             Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
#             Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
#         ])
#     markdown_table_certified_accuracy(
#         "analysis/markdown/vary_noise_imagenet", 0.5, 3.0, 0.5, [
#             Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "&sigma; = 0.25"),
#             Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "&sigma; = 0.50"),
#             Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "&sigma; = 1.00"),
#         ])
#     plot_certified_accuracy(
#         "analysis/plots/vary_noise_cifar10", "CIFAR-10, vary $\sigma$", 1.5, [
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
#         ])
#     plot_certified_accuracy(
#         "analysis/plots/vary_train_noise_cifar_050", "CIFAR-10, vary train noise, $\sigma=0.5$", 1.5, [
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/sigma_0.50"), "train $\sigma = 0.25$"),
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "train $\sigma = 0.50$"),
#             Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/sigma_0.50"), "train $\sigma = 1.00$"),
#         ])
#     plot_certified_accuracy(
#         "analysis/plots/vary_train_noise_imagenet_050", "ImageNet, vary train noise, $\sigma=0.5$", 1.5, [
#             Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.50"), "train $\sigma = 0.25$"),
#             Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "train $\sigma = 0.50$"),
#             Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_1.00/test/sigma_0.50"), "train $\sigma = 1.00$"),
#         ])
#     plot_certified_accuracy(
#         "analysis/plots/vary_noise_imagenet", "ImageNet, vary $\sigma$", 4, [
#             Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
#             Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
#             Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
#         ])
#     plot_certified_accuracy(
#         "analysis/plots/high_prob", "Approximate vs. High-Probability", 2.0, [
#             Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "Approximate"),
#             Line(HighProbAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50", 0.001, 0.001), "High-Prob"),
#         ])
