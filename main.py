import argparse
from pathlib import Path

import torch
from pandas import DataFrame

from BacteriocinClassifier import BacteriocinClassifier
from ELMo import encode_input_fasta


def main(cuda_device, fasta_input_file, csv_output_file):
    if cuda_device is None:
        cuda_device = 0 if torch.cuda.is_available() else -1
        print(f'No cuda device specified. Using cuda device {cuda_device}')

    if fasta_input_file is None:
        fasta_input_file = "./test_data/sample_fasta.faa"
        print(f"No input fasta file specified. Using {fasta_input_file}")

    if csv_output_file is None:
        csv_output_file = "./results.csv"
        print(f"No output path specified. Using {csv_output_file}")

    inpt = encode_input_fasta(input_fasta=Path(fasta_input_file),
                              cuda_device=cuda_device)

    net: BacteriocinClassifier = BacteriocinClassifier()
    net.load_state_dict(torch.load("./weights/bacteriocin_classifier_params.dump"))
    net.eval()

    outpt = net(inpt)

    # Save results
    (DataFrame(outpt.detach().numpy(), columns=['not_bacteriocin_score', 'bacteriocin_score'])
     .to_csv(csv_output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda_device", help='Specify which cuda device to use', type=int)
    parser.add_argument("-f", "--fasta_input_file",
                        help='Input fasta file with sequences to classify', type=str)
    parser.add_argument("-o", "--csv_output_file", help='Where to output csv file with results',
                        type=str)
    parser.add_argument("--mode", help='not used')
    parser.add_argument("--port", help='not used')
    args = parser.parse_args()

    main(cuda_device=args.cuda_device, fasta_input_file=args.fasta_input_file,
         csv_output_file=args.csv_output_file)
