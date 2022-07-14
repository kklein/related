import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def load_synonyms(path="/Users/kevin/Code/embeddings/data/synonyms/synonyms.csv"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            """
            Synonyms could not be found. Please download them from
            https://www.kaggle.com/datasets/duketemon/wordnet-synonyms?resource=download
            """
        )

    df = pd.read_csv(path)
    # TODO: Consider creating several rows with extra synonyms.
    df = df.assign(synonym=df["synonyms"].str.split(";").str[0])
    df = df.drop(columns="synonyms")
    return df


def load_antonyms(path="/Users/kevin/Code/embeddings/data/antonyms/antonyms.csv"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            """
            Antonyms could not be found. Please download them from
            https://www.kaggle.com/datasets/duketemon/antonyms-wordnet
            """
        )

    df = pd.read_csv(path)
    df = df.rename(columns={"antonyms": "antonym"})
    return df


def process_word_pairs(word_pairs, is_synonym, word_to_int, vocabulary, n_samples=2500):
    if is_synonym:
        prefix = "synonym"
    else:
        prefix = "antonym"
    word_pairs = word_pairs.assign(
        lemma_present=word_pairs["lemma"].isin(vocabulary),
        counterpart_present=word_pairs[prefix].isin(vocabulary),
    )
    word_pairs = word_pairs.assign(
        both_present=word_pairs["lemma_present"] & word_pairs["counterpart_present"]
    )

    word_pairs = word_pairs[word_pairs["both_present"]]
    word_pairs = word_pairs.assign(
        lemma_id=word_pairs.apply(lambda x: (word_to_int[x["lemma"]]), axis=1),
        pair_id=word_pairs.apply(lambda x: word_to_int[x[prefix]], axis=1),
    )
    word_pairs = word_pairs.reset_index(drop=True)
    if (n_samples_available := word_pairs.shape[0]) < n_samples:
        warnings.warn(
            f"Could not sample {n_samples} samples, only {n_samples_available} available."
        )
    else:
        n_samples = min(n_samples, word_pairs.shape[0])
        indeces = np.random.choice(word_pairs.shape[0], n_samples, replace=False)
        indeces.sort()
        word_pairs = word_pairs.iloc[indeces]
    return word_pairs
