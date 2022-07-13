from functools import partial

import numpy as np
import torch
import torchtext
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment(name="word2vec")

ex.observers.append(
    MongoObserver.create(
        url="mongodb://sample:password@localhost:27017/?authMechanism=SCRAM-SHA-1",
        db_name="db",
    )
)


@ex.config
def my_config():
    n_dim = 30
    window_size = 6
    max_sequence_length = 1000
    min_word_frequency = 50
    n_negative_samples = 5
    learning_rate = 0.0003
    n_epochs = 10


def load_tokenizer():
    return torchtext.data.get_tokenizer("basic_english")


def load_vocabulary(tokenizer, min_word_frequency):
    train_iter = torchtext.datasets.WikiText2(split="train")
    vocab = torchtext.vocab.build_vocab_from_iterator(
        map(tokenizer, train_iter),
        specials=[" < unk>"],
        min_freq=min_word_frequency,
    )
    vocab.set_default_index(vocab[" < unk>"])
    return vocab


class SkipGramEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, n_negative_samples, n_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=n_dim,
        )
        self.vocab_size = vocab_size
        self.noise_distribution = torch.ones(vocab_size)
        self.n_dim = n_dim
        self.n_negative_samples = n_negative_samples

    def forward(self, contexts, centers):
        n_center_words = centers.shape[0]
        # Double_window_size expresses how many words are in left and right window of center word, combined.
        double_window_size = contexts.shape[1]

        center_embeddings = self.embedding(centers)
        context_embeddings = self.embedding(contexts)

        # We sample random words as to simulate negative sampling.
        # For every (center word, context word) tuple, we have n_negative_samples words to go along
        # with that same center word.
        noise_words = torch.multinomial(
            self.noise_distribution,
            n_center_words * self.n_negative_samples * double_window_size,
            replacement=True,
        )
        noise_embeddings = self.embedding(noise_words)
        noise_embeddings = noise_embeddings.view(
            n_center_words, double_window_size * self.n_negative_samples, self.n_dim
        )
        return context_embeddings, center_embeddings, noise_embeddings


class SkipGramNCE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, context_embeddings, center_embeddings, noise_embeddings):
        n_center_words = center_embeddings.shape[0]
        n_dim = center_embeddings.shape[1]

        # Transpose shape for matrix multiplication.
        center_embeddings = center_embeddings.view(n_center_words, n_dim, 1)
        out_loss = torch.bmm(context_embeddings, center_embeddings).sigmoid().log()

        noise_loss = (
            torch.bmm(noise_embeddings.neg(), center_embeddings).sigmoid().log()
        )
        noise_multiplier = noise_embeddings.shape[1] / context_embeddings.shape[1]

        # Since batch sizes vary, it is preferable to use means over sums.
        # In order to weight negative sampling, we use a multiplier.
        return -(out_loss.mean() + noise_multiplier * noise_loss.mean())


def collate_skipgram(
    paragraph, vocabulary, tokenizer, window_size, max_sequence_length=None
):
    contexts, centers = [], []
    text_token_ids = vocabulary(tokenizer(paragraph))

    if len(text_token_ids) < window_size * 2 + 1:
        return torch.tensor([]), torch.tensor([])
        # continue

    if max_sequence_length:
        text_token_ids = text_token_ids[:max_sequence_length]

    for idx in range(len(text_token_ids) - window_size * 2):
        token_id_sequence = text_token_ids[idx : (idx + window_size * 2 + 1)]
        center = token_id_sequence.pop(window_size)
        contexts.append(token_id_sequence)
        centers.append(center)

    contexts = torch.tensor(contexts, dtype=torch.long)
    centers = torch.tensor(centers, dtype=torch.long)
    return contexts, centers


@ex.capture
@ex.automain
def train(
    n_dim,
    window_size,
    max_sequence_length,
    min_word_frequency,
    n_negative_samples,
    learning_rate,
    n_epochs,
):
    train_iter = torchtext.datasets.WikiText2(split="train")
    tokenizer = load_tokenizer()
    vocabulary = load_vocabulary(
        tokenizer=tokenizer, min_word_frequency=min_word_frequency
    )
    dataloader = torch.utils.data.DataLoader(
        train_iter,
        # The atomic unit of the dataset is paragraphs - which already represent more than one
        # datapoint for our use case.
        batch_size=None,
        shuffle=True,
        collate_fn=partial(
            collate_skipgram,
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            window_size=window_size,
        ),
    )

    model = SkipGramEmbedding(
        vocab_size=len(vocabulary), n_dim=n_dim, n_negative_samples=n_negative_samples
    )
    nce = SkipGramNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_skips = 0
    n_batches = 0
    for epoch_index in range(n_epochs):
        for _, batch_data in enumerate(dataloader, 1):
            contexts = batch_data[0]
            centers = batch_data[1]

            if contexts.shape[0] == 0:
                n_skips += 1
                continue

            outputs = model(contexts, centers)
            loss = nce(*outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if n_batches % 1000 == 0:
                ex.log_scalar("loss", loss.item(), n_batches)
                for name, param in model.named_parameters():
                    ex.log_scalar(
                        f"{name}_gradient", param.grad.norm().item(), n_batches
                    )

            n_batches += 1
    print(f"Skipped {n_skips} times.")
    print(f"Number of batches with updates: {n_batches}")
    np.savetxt("embeddings.csv", model.embedding.weight.detach().numpy(), delimiter=",")
    torch.save(vocabulary, "vocabulary.pth")
