import random

import colorama

import glove


def round(index, int_to_word):
    word_id, word = glove.random_word(int_to_word)
    negative_id, negative_word = glove.random_word(int_to_word)
    candidates, _ = glove.retrieve_nn(word_id, index)
    positive_id = candidates[1]
    positive_word = int_to_word[positive_id]

    positive_position = random.randint(1, 2)

    print(f"What's closest to {word}?")
    if positive_position == 1:
        print(f"1: {positive_word}")
        print(f"2: {negative_word}")
    else:
        print(f"1: {negative_word}")
        print(f"2: {positive_word}")

    response = int(input())

    if response == positive_position:
        print(colorama.Fore.GREEN + "Success!")
        return True
    print(colorama.Fore.RED + "That was wrong.")


def main():
    colorama.init(autoreset=True)
    embeddings, int_to_word, word_to_int = glove.create_embeddings()
    index = glove.build_index(embeddings, metric="angular", n_trees=30)
    successes = 0
    rounds = 0
    while True:
        if round(index, int_to_word):
            successes += 1
        rounds += 1
        print(f"{successes}/{rounds}")
        print("-" * 20)


if __name__ == "__main__":
    main()
