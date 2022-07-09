import random
from pathlib import Path

import numpy as np
from annoy import AnnoyIndex


def bimap_embeddings(embeddings):
    int_to_word = {i: word for i, word in enumerate(embeddings.keys())}
    word_to_int = {word: i for i, word in int_to_word.items()}
    return int_to_word, word_to_int


def create_embeddings(
    path="/Users/kevin/Code/embeddings/data/glove.6B/glove.6B.100d.txt",
    dtype="float32",
):
    mapping = {}
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
