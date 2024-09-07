import numpy as np
from collections import Counter
from players.guesser import Guesser
import players.random_dialect_guesser
import matplotlib.pyplot as plt


class MetaGuesser:
    """
    Each player votes for their top guess, and the guess with the
    most votes wins, without considering the certainty levels
    (AKA simple committe)
    """
    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
        self.players = [players.random_dialect_guesser.AIGuesser(brown_ic, glove_vecs, word_vectors, 0) for i in range(5)]

    def set_board(self, words):
        for player in self.players:
            player.set_board(words)

    def set_clue(self, clue, num):
        for player in self.players:
            player.set_clue(clue, num)

    def keep_guessing(self):
        return all(player.keep_guessing() for player in self.players)

    def get_answer(self):
        answers = [player.get_answer() for player in self.players] # assume this function (certainty, guesses)
        answer_counts = Counter(answers)


        final_answer, final_count = answer_counts.most_common(1)[0]

        print(f'Meta-player final guess: {final_answer}')
        return final_answer