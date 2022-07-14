import random
from pathlib import Path

import numpy as np
import torch
from annoy import AnnoyIndex


def bimap_embeddings(embeddings):
    int_to_word = {i: word for i, word in enumerate(embeddings.keys())}
    word_to_int = {word: i for i, word in int_to_word.items()}
    return int_to_word, word_to_int


def load_glove_embeddings(n_dim=100, path=None, dtype="float32"):
    mapping = {}
    if path is None:
        path = f"/Users/kevin/Code/embeddings/data/glove.6B/glove.6B.{n_dim}d.txt"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            """
            Embeddings could not be found. Please download them from

            https://nlp.stanford.edu/data/glove.6B.zip"""
        )
    with open(path) as filehandle:
        for line in filehandle:
            values = line.split()
            mapping[values[0]] = np.asarray(values[1:], dtype=dtype)
    embeddings = np.array(list(mapping.values()))
    int_to_word, word_to_int = bimap_embeddings(mapping)
    return embeddings, int_to_word, word_to_int


def load_self_trained_word2vec_embeddings(path=None, dtype="float32"):
    mapping = []
    if path is None:
        path = "/Users/kevin/Code/embeddings/data/self-trained-word2vec/embeddings.csv"
    path = Path(path)
    with open(path) as filehandle:
        for line in filehandle:
            mapping.append(np.asarray(line.split(","), dtype=dtype))
    embeddings = np.array(mapping)
    vocabulary = torch.load(
        "/Users/kevin/Code/embeddings/data/self-trained-word2vec/vocabulary.pth"
    )
    word_to_int = vocabulary.vocab.get_stoi()
    int_to_word = vocabulary.vocab.get_itos()
    return embeddings, int_to_word, word_to_int


def load_embeddings(embedding_type, *args, **kwargs):
    if embedding_type == "self-trained":
        return load_self_trained_word2vec_embeddings(*args, **kwargs)
    elif embedding_type == "glove":
        return load_glove_embeddings(*args, **kwargs)
    raise ValueError(f"Unexpected embedding type: {embedding_type}.")


def build_index(embeddings, metric="euclidean", n_trees=10):
    n_dim = embeddings.shape[1]
    index = AnnoyIndex(n_dim, metric)
    for i in range(embeddings.shape[0]):
        index.add_item(i, embeddings[i])
    index.build(n_trees)
    return index


def random_word(int_to_word):
    vocabulary_size = len(int_to_word)
    id = random.randint(0, vocabulary_size - 1)
    return id, int_to_word[id]


def retrieve_nn(id: int, index: AnnoyIndex, n_neighbors=10):
    return index.get_nns_by_item(id, n_neighbors, include_distances=True)
