"""
prdc 
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
import numpy as np
import sklearn.metrics

from torch import nn
from torchvision.models import vgg16

from fid import preprocess_images, load_images, to_cuda

__all__ = ['compute_prdc']


class PartialVGG16(nn.Module):
    def __init__(self, transform_input=True):
        super().__init__()
        self.vgg16_net = vgg16(pretrained=False)
        self.vgg16_net.classifier = self.vgg16_net.classifier[:-1]
    
    def forward(self, x):
        
        assert x.shape[1:] == (3, 224, 224), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        x = x * 2 -1 # Normalize to [-1, 1]

        features_var=self.vgg16_net(x)

        return features_var


def get_features(images, batch_size):
    """
    Calculates activations for last pool layer for all iamges
    --
        Images: torch.array shape: (N, 3, 299, 299), dtype: torch.float32
        batch size: batch size used for inception network
    --
    Returns: np array shape: (N, 2048), dtype: np.float32
    """
    assert images.shape[1:] == (3, 224, 224), "Expected input shape to be: (N,3,299,299)" +\
                                              ", but got {}".format(images.shape)

    num_images = images.shape[0]
    vgg16 = PartialVGG16()
    vgg16 = to_cuda(vgg16)
    vgg16.eval()
    n_batches = int(np.ceil(num_images  / batch_size))
    vgg16_features = np.zeros((num_images, 4096), dtype=np.float32)
    for batch_idx in range(n_batches):
        start_idx = batch_size * batch_idx
        end_idx = batch_size * (batch_idx + 1)

        ims = images[start_idx:end_idx]
        ims = to_cuda(ims)
        activations = vgg16(ims)
        activations = activations.detach().cpu().numpy()
        assert activations.shape == (ims.shape[0], 4096), "Expexted output shape to be: {}, but was: {}".format((ims.shape[0], 4096), activations.shape)
        vgg16_features[start_idx:end_idx, :] = activations
    return vgg16_features


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii



def compute_prdc(images1, images2, nearest_k, batch_size):
    """
    Computes precision, recall, density, and coverage given two manifolds.
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    images1 = preprocess_images(images1, 224)
    images2 = preprocess_images(images2, 224)
    
    real_features = get_features(images1, batch_size)
    fake_features = get_features(images2, batch_size)

    print('Num real: {} Num fake: {}'
          .format(real_features.shape[0], fake_features.shape[0]))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)



if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--p1", "--path1", dest="path1", 
                      help="Path to directory containing the real images")
    parser.add_option("--p2", "--path2", dest="path2", 
                      help="Path to directory containing the generated images")
    parser.add_option("--multiprocessing", dest="use_multiprocessing",
                      help="Toggle use of multiprocessing for image pre-processing. Defaults to use all cores",
                      default=False,
                      action="store_true")
    parser.add_option("-b", "--batch-size", dest="batch_size",
                      help="Set batch size to use for InceptionV3 network",
                      type=int)
                      
    parser.add_option("-k", "--nearest-k-size", dest="Nearest Neighbour size",
                      help="Set neighbour size to use for KNN model",
                      type=int)
    
    options, _ = parser.parse_args()
    assert options.path1 is not None, "--path1 is an required option"
    assert options.path2 is not None, "--path2 is an required option"
    assert options.batch_size is not None, "--batch_size is an required option"
    images1 = load_images(options.path1)
    images2 = load_images(options.path2)
    fid_value = compute_prdc(images1, images2, options.nearest_k_size, options.batch_size)
    print(fid_value)