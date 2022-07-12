import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.metrics.pairwise import paired_cosine_distances

import glove
import helper

ex = Experiment()

ex.observers.append(
    MongoObserver.create(
        url="mongodb://sample:password@localhost:27017/?authMechanism=SCRAM-SHA-1",
        db_name="db",
    )
)


def word_pair_distance(word_pairs, embeddings):
    # TODO: Consider using a generator.
    lemma_vectors = np.array([embeddings[id] for id in word_pairs["lemma_id"]])
    synonym_vectors = np.array([embeddings[id] for id in word_pairs["pair_id"]])
    return paired_cosine_distances(lemma_vectors, synonym_vectors)


def total_loss(antonym_distances, synonym_distances):
    return -antonym_distances.sum() + synonym_distances.sum()


@ex.config
def my_config():
    n_dim = 100
    dtype = "float16"


def plot_antonym_synonym_histograms(
    antonym_distances,
    synonym_distances,
    filename="histogram.png",
):
    fig, ax = plt.subplots()
    bins = np.linspace(0, 2, 100)

    ax.hist(antonym_distances, bins, alpha=0.5, label="antonyms")
    ax.hist(synonym_distances, bins, alpha=0.5, label="synonyms")
    ax.legend()
    fig.savefig(filename)
    return filename


def plot_sample_embeddings(
    embeddings,
    word_to_int,
    words=["king", "queen", "man", "woman", "fire", "water", "jealous", "deprived"],
    filename="embedding_sample.png",
):
    # Context:
    # https://stackoverflow.com/questions/3529666/matplotlib-matshow-labels
    ids = [word_to_int[word] for word in words]
    embedding_selection = embeddings[ids, :]
    fig, ax = plt.subplots()

    cax = ax.matshow(embedding_selection, aspect="auto")
    ax.set_yticklabels([""] + words)
    fig.colorbar(cax)

    fig.savefig(filename)
    return filename


@ex.capture
@ex.automain
def eval(n_dim, dtype, vocabulary=None):
    embeddings, int_to_word, word_to_int = glove.create_embeddings()
    if vocabulary is None:
        vocabulary = set(word_to_int.keys())

    synonyms = helper.process_word_pairs(
        helper.load_synonyms(),
        is_synonym=True,
        word_to_int=word_to_int,
        vocabulary=vocabulary,
    )

    antonyms = helper.process_word_pairs(
        helper.load_antonyms(),
        is_synonym=False,
        word_to_int=word_to_int,
        vocabulary=vocabulary,
    )

    a_distances = word_pair_distance(antonyms, embeddings)
    s_distances = word_pair_distance(synonyms, embeddings)
    loss = total_loss(a_distances, s_distances)
    ex.log_scalar("loss", loss)
    ex.add_artifact(plot_antonym_synonym_histograms(a_distances, s_distances))
    ex.add_artifact(plot_sample_embeddings(embeddings, word_to_int))
