import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from archs.mnist_mlp import mlp as mlp_mnist
from datasets import get_normalize_layer
from torch.nn.functional import interpolate

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110", "mnist_mlp"]

def get_architecture(arch: str, dataset: str, noise_std = [], hidden_size = 444, nonlinear=0) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :param noise_std: list of the noise applied to the different alyers
    :param hidden_size: size of the first hidden layer
    :param nonlinear: indicator for whether first hidden layer has a nonlinearity
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    elif arch == "mnist_mlp":
        model = mlp_mnist(input_size = 444, hidden_size = hidden_size,  second_layer = 20, num_classes=10, noise_std = noise_std, nonlinear = nonlinear).cuda()
    if arch != "mnist_mlp":
        normalize_layer = get_normalize_layer(dataset)
        return torch.nn.Sequential(normalize_layer, model)
    else:
        return model
