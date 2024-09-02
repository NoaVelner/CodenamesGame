import numpy as np
from collections import Counter
from players.guesser import Guesser
import players.random_dialect_guesser
import matplotlib.pyplot as plt


class MetaGuesser:
    """
    This committee works as follows: each guesser returns 3 guesses along with their certainty levels.
    Each guesser has 1 point to distribute among their guesses, and they will do so using normalization.
    """

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
        """
        Nornalize the guesses as the closest the word is, the higher score it gets.
        Every player has 1 point to distribute among their guesses
        """
        all_guesses = []
        for player in self.players:
            guesses = player.get_answer(3)  # Assume this method returns [(certainty, guess), ...]
            all_guesses.append(guesses)

        answer_counts = Counter()

        for guesses in all_guesses:
            weights = [2 - guess[0] for guess in guesses]
            total_weight = sum(weights)
            normalized_weights = [weight / total_weight for weight in weights]

            for i in range(3):
                answer_counts[guesses[i][1]] += normalized_weights[i]

        for (el, count) in answer_counts.items():
            print(f"ELEMENT: {el}, COUNT: {count}")

        final_answer, final_count = answer_counts.most_common(1)[0]

        unique_count = len(answer_counts)
        self.unique_guess_counts.append(unique_count)
        total_score = sum(answer_counts.values())

        certainty_percentage = np.round((final_count / total_score), 3)
        self.certainty_of_chosen_guess.append(certainty_percentage)
        return final_answer

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

    def get_certainty(self):
        return self.certainty_of_chosen_guess


