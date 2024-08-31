import numpy as np
import scipy.spatial.distance

from players.guesser import Guesser

class AIGuesser(Guesser):

    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None, param=0.1, distance_threshold=0.7):
        super().__init__()
        self.brown_ic = brown_ic
        self.glove_vecs = glove_vecs
        self.word_vectors = word_vectors
        self.num = 0
        size=300
        random_values =  np.random.binomial(1, 0.05, size=(300,))
        secondary_diagonal =  np.random.binomial(1, 0.01, size=(299,))
        theard_diagonal =  np.random.binomial(1, 0.1, size=(300,))
        self.matrix = np.diag(random_values)
        self.matrix[np.arange(size - 1), np.arange(1, size)] = secondary_diagonal
        self.distance_threshold = distance_threshold

    def set_board(self, words):
        self.words = words

    def set_clue(self, clue, num):
        self.clue = clue
        self.num = num
        li = [clue, num]
        return li

    def keep_guessing(self):
        return self.num > 0

    def get_answer(self):
        sorted_words = self.compute_distance(self.clue, self.words)
        next_guess_distance, next_guess_word = sorted_words[0]

        print(sorted_words[0][1], next_guess_distance)
        self.num -= 1
        return sorted_words[0][1]

    def compute_distance(self, clue, board):
        w2v = []

        flag=True
        for word in board:
            try:
                if word[0] == '*':
                    continue

                transformed_clue_vec = np.dot(self.word_vectors[clue], self.matrix)
                transformed_word_vec = np.dot(self.word_vectors[word.lower()], self.matrix)
                if flag:
                    flag=False

                distance = scipy.spatial.distance.cosine(transformed_clue_vec, transformed_word_vec)
                w2v.append((distance, word))

            except KeyError:
                continue

        w2v = list(sorted(w2v))
        return w2v