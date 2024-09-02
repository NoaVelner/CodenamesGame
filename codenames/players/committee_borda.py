import numpy as np
from collections import Counter
from players.guesser import Guesser
import players.random_dialect_guesser
import matplotlib.pyplot as plt


class MetaGuesser:
    """
    Each player ranks their guesses between 1-number of guesses,
    and the guesses receive points based on their rank.
    The guess with the highest total points across all players is the
    most acceptable candidate."""
    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
        self.players = [players.random_dialect_guesser.AIGuesser(brown_ic, glove_vecs, word_vectors, -2) for i in
                        range(5)]
        self.unique_guess_counts = []
        self.certainty_of_chosen_guess = []

    def set_board(self, words):
        for player in self.players:
            player.set_board(words)

    def set_clue(self, clue, num):
        for player in self.players:
            player.set_clue(clue, num)

    def keep_guessing(self):
        # TODO: Change to "any" and avoid None guesses if needed
        return all(player.keep_guessing() for player in self.players)

    def get_answer(self):
        """ Simple weights version: the first is the most important, and so on"""
        all_guesses = []
        for player in self.players:
            guesses = player.get_answer(3)  # Assume this method returns [(certainty, guess), ...]
            all_guesses.append(guesses)

        answer_counts = Counter()
        for guesses in all_guesses:
            for i in range(3):
                answer_counts[guesses[i][1]] += 3 - i

        for (el, count) in answer_counts.items():
            print(f"ELEMENT: {el}, COUNT: {count}")

        unique_count = len(answer_counts)
        self.unique_guess_counts.append(unique_count)
        final_answer, final_count = answer_counts.most_common(1)[0]

        total_score = sum(answer_counts.values())

        certainty_percentage = np.round((final_count / total_score),3)
        self.certainty_of_chosen_guess.append(certainty_percentage)
        print(f'Meta-player final guess: {final_answer}')
        return final_answer

    def get_certainty(self):
        return self.certainty_of_chosen_guess

    def plot_unique_guess_counts(self):
        plt.plot(self.unique_guess_counts, marker='o')
        plt.title('Number of Unique Guesses per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Number of Unique Guesses')
        plt.show()

    def plot_certainty_of_chosen_guess(self):
        plt.plot(self.certainty_of_chosen_guess, marker='o')
        plt.title('Certainty of Chosen Guess per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Certainty')
        plt.show()


    # def get_answer(self):
    #     """ normalization"""
    #     all_guesses = []
    #     for player in self.players:
    #         guesses = player.get_answer(3)  # Assume this method returns [(certainty, guess), ...]
    #         all_guesses.append(guesses)
    #
    #     answer_counts = Counter()
    #     for guesses in all_guesses:
    #         for i in range(3):
    #             answer_counts[guesses[i][1]] += 2 - guesses[i][0]
    #
    #     for (el, count) in answer_counts.items():
    #         print(f"ELEMENT: {el}, COUNT: {count}")
    #
    #     self.unique_guess_counts = len(answer_counts.items())
    #     final_answer = answer_counts.most_common(1)[0][0]
    #     return final_answer