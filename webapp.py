# Inspired by
# https://realpython.com/python-web-applications/#convert-a-script-into-a-web-application

import random
from functools import cache

from flask import Flask, request

import glove

app = Flask(__name__)

n_successes = 0
n_trials = 0


@cache
def get_objects():
    embeddings, int_to_word, word_to_int = glove.create_embeddings()
    index = glove.build_index(embeddings, metric="angular", n_trees=30)
    return embeddings, int_to_word, word_to_int, index


@app.route("/")
def index():
    global n_successes
    global n_trials
    last_result = request.args.get("result", "")
    app.logger.info(f"Last result: {last_result}")

    if last_result != "":
        n_trials += 1
        if last_result == "success":
            n_successes += 1

    embeddings, int_to_word, word_to_int, index = get_objects()
    word_id, word = glove.random_word(int_to_word)
    negative_id, negative_word = glove.random_word(int_to_word)
    app.logger.info(type(index))
    candidates, _ = glove.retrieve_nn(word_id, index)
    positive_id = candidates[1]
    positive_word = int_to_word[positive_id]
    app.logger.info(f"Current word: {word}")
    app.logger.info(f"Positive candidate: {positive_word}")
    app.logger.info(f"Negative candidate: {negative_word}")

    positive_position = random.randint(1, 2)
    if positive_position == 1:
        return f"""
            Score: {n_successes}/{n_trials} </br>
            What's closer to {word}? </br>
            <form action="" method="get">
                <button name="result" type="submit" value="success">{positive_word}</button>
            </form>
            <form action="" method="get">
                <button name="result" type="submit" value="failure">{negative_word}</button>
            </form>
        """
    return f"""
        Score: {n_successes}/{n_trials} </br>
        What's closer to {word}? </br>
        <form action="" method="get">
            <button name="result" type="submit" value="failure">{negative_word}</button>
        </form>
        <form action="" method="get">
            <button name="result" type="submit" value="success">{positive_word}</button>
        </form>
    """


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
