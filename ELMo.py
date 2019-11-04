import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from allennlp.commands.elmo import ElmoEmbedder


def load_elmo_model(cuda_device, model_dir=Path('elmo_model_uniref50/')):
    weights = model_dir / 'weights.hdf5'
    options = model_dir / 'options.json'
    return ElmoEmbedder(options, weights, cuda_device=cuda_device)


def fasta2csv(path):
    fasta_dir = {'ID': [], 'Sequence': [], 'Name': []}
    for record in SeqIO.parse(path, 'fasta'):
        fasta_dir['ID'].append(record.id)
        fasta_dir['Sequence'].append(str(record.seq))
        fasta_dir['Name'].append(record.description)
    return pd.DataFrame(fasta_dir)


def pad_n_flatten_embeddings(embedding, max_length, elmo_dimension):
    flattener = (lambda e: e.sum(0).flatten())

    # Flatten embeddings
    embedding_flat = [flattener(e) for e in embedding]

    # Pad embeddings
    pad_n = max_length * elmo_dimension
    embedding_flat_padded = [np.pad(array=e, pad_width=[0, pad_n - len(e)],
                                    mode='constant', constant_values=0)
                             for e in embedding_flat]
    embedding_flat_padded = np.stack(embedding_flat_padded)

    # Reshape into (batch size, amino acid sequence length, # elmo dimensions)
    return embedding_flat_padded.reshape([-1, max_length, elmo_dimension])


def encode_input_fasta(input_fasta, cuda_device, max_length=359, elmo_dimension=1024):
    # Convert input fasta to csv
    df = fasta2csv(input_fasta)

    # Assert sequence lengths
    seq_lengths = df['Sequence'].str.len().values
    assert all(seq_lengths <= max_length), f"String length exceeding {max_length}"

    print("Embedding sequences..")
    start_time = time.time()
    residue_list = [list(s) for s in df['Sequence']]
    seqvec = load_elmo_model(cuda_device=cuda_device)
    embedding = list(seqvec.embed_sentences(residue_list))
    embedding = pad_n_flatten_embeddings(embedding=embedding, max_length=max_length,
                                         elmo_dimension=elmo_dimension)
    end_time = time.time()
    print(f"Took {end_time - start_time} seconds")

    # Channel dimension is the second one for PyTorch convolution
    return torch.tensor(embedding.swapaxes(1, 2))


if __name__ == '__main__':
    CUDA_DEVICE = -1
    encode_input_fasta(input_fasta=Path("./test_data/sample_fasta.faa"), cuda_device=CUDA_DEVICE)
