from pathlib import Path

import torch
from pandas import DataFrame

from BacteriocinClassifier import BacteriocinClassifier
from ELMo import encode_input_fasta


def main(cuda_device=None):
    if cuda_device is None:
        cuda_device = 0 if torch.cuda.is_available() else -1

    inpt = encode_input_fasta(input_fasta=Path("./test_data/sample_fasta.faa"),
                              cuda_device=cuda_device)

    net: BacteriocinClassifier = BacteriocinClassifier()
    net.load_state_dict(torch.load("./weights/bacteriocin_classifier_params.dump"))
    net.eval()

    outpt = net(inpt)

    return DataFrame(outpt.detach().numpy())


if __name__ == "__main__":
    csv_output = main()
