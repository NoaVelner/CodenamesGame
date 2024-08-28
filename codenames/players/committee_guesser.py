import numpy as np
from collections import Counter
from players.guesser import Guesser
import players.random_dialect_guesser



class MetaGuesser:
    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
        # noise = np.random.normal(0, 0.1, 5)
        # print(noise)
        self.players = [players.random_dialect_guesser.AIGuesser(brown_ic, glove_vecs, word_vectors, -2) for i in range(5)]

    def set_board(self, words):
        for player in self.players:
            player.set_board(words)

    def set_clue(self, clue, num):
        for player in self.players:
            player.set_clue(clue, num)

    def keep_guessing(self):
        # Only keep guessing if all players agree that they should
        return all(player.keep_guessing() for player in self.players)

    def get_answer(self):
        answers = [player.get_answer() for player in self.players]
        answer_counts = Counter(answers)

        final_answer = answer_counts.most_common(1)[0][0]
        print(f'Meta-player final guess: {final_answer}')
        return final_answer
