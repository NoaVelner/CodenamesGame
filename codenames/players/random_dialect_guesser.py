import numpy as np
import scipy.spatial.distance

from players.guesser import Guesser


class AIGuesser(Guesser):
    """"AIGuesser with dialect matrix"""

    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None, budget=0):
        super().__init__()
        self.brown_ic = brown_ic
        self.glove_vecs = glove_vecs
        self.word_vectors = word_vectors
        self.budget = budget
        self.num = 0
        self.generate_dialect()

    def generate_dialect(self):
        size = 300
        p_1 = 0.09
        p_2 = 0.02
        random_values = np.random.binomial(1, p_1, size=(size,))
        secondary_diagonal = np.random.binomial(1, p_2, size=(size - 1,))
        self.matrix = np.diag(random_values)
        self.matrix[np.arange(size - 1), np.arange(1, size)] = secondary_diagonal

    def set_board(self, words):
        self.words = words

    def set_clue(self, clue, num):
        self.clue = clue
        self.num = num
        li = [clue, num]
        return li

    def keep_guessing(self):
        return self.num > 0

    def get_answer(self, num_of_guess=1):
        sorted_words = self.compute_distance(self.clue, self.words)
        next_guess_distance, next_guess_word = sorted_words[0]

        if num_of_guess > 1:
            return sorted_words[:num_of_guess]

        print(sorted_words[0][1], next_guess_distance)
        self.num -= 1
        return sorted_words[0][1]

    def compute_distance(self, clue, board):
        w2v = []

        flag = True
        for word in board:
            try:
                if word[0] == '*':
                    continue

                transformed_clue_vec = np.dot(self.word_vectors[clue], self.matrix)
                transformed_word_vec = np.dot(self.word_vectors[word.lower()], self.matrix)
                if flag:
                    flag = False

                distance = scipy.spatial.distance.cosine(transformed_clue_vec, transformed_word_vec)
                w2v.append((distance, word))

            except KeyError:
                continue

        w2v = list(sorted(w2v))
        return w2v

    def update_budget(self, amount):
        self.budget += amount

    def suggest_bid_and_guess(self):
        """
        Suggests a bid amount and the next guess based on certainty and budget.
        The distance in range [0,2].
        """
        guesses_list = self.get_answer(2)
        certainty, guess = guesses_list[0]
        return (((2 - certainty) / 2) * self.budget * 0.8, guess)
