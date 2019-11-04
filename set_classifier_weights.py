from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from BacteriocinClassifier import BacteriocinClassifier


def main():
    net: BacteriocinClassifier = BacteriocinClassifier()
    net.eval()

    # Set weights of other model
    weights = joblib.load("./weights/tf_weights.pickle")

    parameter_dict = dict(net.named_parameters())
    for i, (name, params) in enumerate(parameter_dict.items()):
        w: np.ndarray = weights[i].swapaxes(0, -1)

        assert tuple(params.shape) == w.shape

        atts = name.split('.')
        att_obj = net
        for att in atts[:-1]:
            att_obj = getattr(att_obj, att)

        # noinspection PyArgumentList
        setattr(att_obj, atts[-1], nn.Parameter(torch.tensor(w)))

    parameter_dict = dict(net.named_parameters())
    for i, (name, params) in enumerate(parameter_dict.items()):
        w: np.ndarray = weights[i].swapaxes(0, -1)
        assert (params.detach().numpy() == w).all()

    return net


if __name__ == '__main__':
    path = Path("./weights/bacteriocin_classifier_params.dump")

    bac_clf: BacteriocinClassifier = main()

    do_override = input(f"Override existing weights at {path}?") if path.exists() else 'y'
    if do_override == 'y':
        print('Saving model parameters')
        torch.save(bac_clf.state_dict(), path)
    else:
        print("aborting")
