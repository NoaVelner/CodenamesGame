import numpy as np
import scipy.spatial.distance

from players.guesser import Guesser

#
# class AIGuesser(Guesser):
#
#     def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
#         super().__init__()
#         self.brown_ic = brown_ic
#         self.glove_vecs = glove_vecs
#         self.word_vectors = word_vectors
#         self.num = 0
#         self.dialect = np.random(0.95, 1, size=(50, 50))
#
#     def set_board(self, words):
#         self.words = words
#
#     def set_clue(self, clue, num):
#         self.clue = clue
#         self.num = num
#         print("The clue is:", clue, num)
#         li = [clue, num]
#         return li
#
#     def keep_guessing(self):
#         return self.num > 0
#
#     def get_answer(self):
#         sorted_words = self.compute_distance(self.clue, self.words)
#         self.num -= 1
#         return sorted_words[0][1]
#
#     def compute_distance(self, clue, board):
#         w2v = []
#         all_vectors = (self.word_vectors, self.glove_vecs,)
#
#         for word in board:
#             try:
#                 if word[0] == '*':
#                     continue
#                 weighted_clue = self.dialect*clue
#                 w2v.append((scipy.spatial.distance.cosine(self.concatenate(clue, all_vectors),
#                                                           self.concatenate(word.lower(), all_vectors)), word))
#             except KeyError:
#                 continue
#
#         w2v = list(sorted(w2v))
#         return w2v
#
#     def combine(self, words, wordvecs):
#         factor = 1.0 / float(len(words))
#         new_word = self.concatenate(words[0], wordvecs) * factor
#         for word in words[1:]:
#             new_word += self.concatenate(word, wordvecs) * factor
#         return new_word
#
#     def concatenate(self, word, wordvecs):
#         concatenated = wordvecs[0][word]
#         for vec in wordvecs[1:]:
#             concatenated = np.hstack((concatenated, vec[word]))
#         return concatenated
#
#     import scipy.spatial.distance
#     import numpy as np
#
#     from players.guesser import Guesser

class AIGuesser(Guesser):

    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
        super().__init__()
        self.brown_ic = brown_ic
        self.glove_vecs = glove_vecs
        self.word_vectors = word_vectors
        self.num = 0
        param = 0.1
        self.matrix = np.random.uniform(1-param, 1+param, (300, 300))
        print(self.matrix)

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
        print(sorted_words[0][1])
        # print(f'guesses: {sorted_words}')
        self.num -= 1
        return sorted_words[0][1]

    def compute_distance(self, clue, board):
        w2v = []

        for word in board:
            try:
                if word[0] == '*':
                    continue

                transformed_clue_vec = np.dot(self.word_vectors[clue], self.matrix)
                transformed_word_vec = np.dot(self.word_vectors[word.lower()], self.matrix)

                distance = scipy.spatial.distance.cosine(transformed_clue_vec, transformed_word_vec)
                w2v.append((distance, word))

            except KeyError:
                continue

        w2v = list(sorted(w2v))
        return w2v