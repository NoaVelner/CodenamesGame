import numpy as np
from collections import Counter
from players.guesser import Guesser
import players.random_dialect_guesser
import matplotlib.pyplot as plt


class MetaGuesser:
    """
    Each guesser suggest a bid for guesse, and this class selects the
    player with the highest bid, and returns the guess made by that player.
    Each player has limited budget.
    """

    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None, num_players=5, budget=10):
        self.players = [players.random_dialect_guesser.AIGuesser(brown_ic, glove_vecs, word_vectors, budget)
                        for _ in range(num_players)]

    def set_board(self, words):
        for player in self.players:
            player.set_board(words)

    def set_clue(self, clue, num):
        for player in self.players:
            player.set_clue(clue, num)

    def keep_guessing(self):
        return all(player.keep_guessing() for player in self.players)

    def get_answer(self):
        """
        Determines the best guess among multiple players based on their suggested bids and guesses.
        This method collects bids and guesses from all players, selects the player with the highest bid,
        deducts the bid amount from the chosen player's budget, and returns the guess made by that player.
        """
        answers = [player.suggest_bid_and_guess() for player in self.players]
        choosen_guesser = 0

        for i in range(len(self.players)):
            print(answers[i])
            if answers[i][0] > answers[choosen_guesser][0]:
                choosen_guesser = i

        self.players[choosen_guesser].update_budget(-answers[choosen_guesser][0])

        final_answer = answers[choosen_guesser][1]
        print(f'Sealed Bid Meta-player Final Guess: {final_answer}, '
              f'The highest bid is {np.round(answers[choosen_guesser][0],5)}')

        return final_answer
