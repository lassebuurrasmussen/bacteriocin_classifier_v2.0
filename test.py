"""Asserts that setting up weights from the previous TensorFlow implementation produces the same
results"""
import joblib
import numpy as np
from set_classifier_weights import main as set_classifier_weights
import torch


def assert_dimensions(x):
    elmo_dimensions = 1024
    max_residues = 359

    channel_dimension = x.shape.index(elmo_dimensions)
    residue_dimension = x.shape.index(max_residues)
    batch_dimension = [i_ for i_ in range(3) if i_ not in [channel_dimension, residue_dimension]][0]

    return (0, 1, 2) == (batch_dimension, channel_dimension, residue_dimension)


if __name__ == '__main__':
    net = set_classifier_weights()
    params = list(net.parameters())
    total_params = sum([np.prod(t.shape) for t in params])

    X = torch.tensor(joblib.load("template_X.dump"))

    if not assert_dimensions(X):
        X = torch.tensor(X.numpy().swapaxes(1, 2))
    assert assert_dimensions(X)

    outpt = net(X.clone())
    tf_outpt = joblib.load("template_predictions.dump")
    assert np.allclose(tf_outpt, outpt.detach().numpy())
