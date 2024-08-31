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
        return self.words[0]

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