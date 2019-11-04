from pathlib import Path

import torch
from pandas import DataFrame

from BacteriocinClassifier import BacteriocinClassifier
from ELMo import encode_input_fasta

CUDA_DEVICE = -1
inpt = encode_input_fasta(input_fasta=Path("./test_data/sample_fasta.faa"), cuda_device=CUDA_DEVICE)
inpt = torch.tensor(inpt.swapaxes(1, 2))

net: BacteriocinClassifier = BacteriocinClassifier()
net.load_state_dict(torch.load("./weights/bacteriocin_classifier_params.dump"))
net.eval()

outpt = net(inpt)

DataFrame(outpt.detach().numpy())
