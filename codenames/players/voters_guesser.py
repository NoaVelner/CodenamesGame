import numpy as np
from collections import Counter
from players.guesser import Guesser
import players.random_dialect_guesser
import matplotlib.pyplot as plt



class MetaGuesser:
    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
        self.players = [players.random_dialect_guesser.AIGuesser(brown_ic, glove_vecs, word_vectors, -2) for i in range(5)]
        self.unique_guess_counts =[]

    def set_board(self, words):
        for player in self.players:
            player.set_board(words)

    def set_clue(self, clue, num):
        for player in self.players:
            player.set_clue(clue, num)

    def keep_guessing(self):
        # Keep guessing if all players agree that they should.
        # TODO: Change to "any" and avoid None guesses if needed
        return all(player.keep_guessing() for player in self.players)

    def get_answer(self):
        """ Simple weights version: the first is the most important, and so on"""
        all_guesses = []
        for player in self.players:
            guesses = player.get_answer(3)  # Assume this method returns [(guess, certainty), ...]
            all_guesses.append(guesses)

        answer_counts = Counter()
        for guesses in all_guesses:
            for i in range(3):
                answer_counts[guesses[i][1]]+=2- guesses[i][0]
        for (el, count) in answer_counts.items():
            print(f"ELEMENT: {el}, COUNT: {count}")
        final_answer = answer_counts.most_common(1)[0][0]
        return final_answer

    def plot_unique_guess_counts(self):
        # Plot the number of unique guesses in each iteration
        plt.plot(self.unique_guess_counts, marker='o')
        plt.title('Number of Unique Guesses per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Number of Unique Guesses')
        plt.show()
