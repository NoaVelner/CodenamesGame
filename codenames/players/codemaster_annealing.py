import numpy as np
from simanneal import Annealer
import random
import json

from players.codemaster import Codemaster


class MyProblem(Annealer):
    def __init__(self, state, pre_processed_ds, lm, good_words, bad_words):
        self.state = state
        self.pre_processed_ds = pre_processed_ds  # a dictionary where each word has it's top 20 words
        self.lm = lm  # a dictionary where each word is a vector
        self.good_words = good_words
        self.bad_words = bad_words
        super(MyProblem, self).__init__(state)


    def move(self):

        # legal_words = self.pre_processed_ds.legal_words[self.state.lower()] - self.good_words - self.bad_words
        legal_words = [item for item in self.pre_processed_ds[self.state.lower()] if item.upper() not in self.good_words and item.upper() not in self.bad_words]
        weights = [20 - i for i in range(len(legal_words))]

        # Choose a word randomly with weights
        chosen_word = random.choices(legal_words, weights=weights, k=1)
        new_state = chosen_word[0]
        self.state = new_state

    def energy(self):
        score = 0
        last_good_word_index = -1
        available_words = []
        for word in self.good_words:
            available_words.append(word)
        for word in self.bad_words:
            available_words.append(word)

        available_words.sort(key=lambda x: np.linalg.norm(self.lm[self.state.lower()] - self.lm[x.lower()]))

        # Define the objective function (energy) to minimize
        # For example, calculate how close the state is to good words and far from bad words
        for w in available_words:
            if w in self.good_words:

                score += 100
                last_good_word_index += 1

            else:
                break
        next_good_word_index = None
        for i in range(last_good_word_index+1, len(available_words)):
            if available_words[i] in self.good_words:
                next_good_word_index = i
                break

        if next_good_word_index is not None:
            score += np.linalg.norm(self.lm[available_words[last_good_word_index]] - self.lm[available_words[next_good_word_index]])

        return -score




class AICodemaster(Codemaster):

    def __init__(self, brown_ic = None, glove_vecs =None, word_vectors = None):
    # def __init__(self, pre_processed_ds = None, lm = None):
        self.lm = glove_vecs
        super().__init__()
        self.pre_processed_ds = {}
        with open('closest_combined_words_within_dataset.json', 'r') as f:
            self.pre_processed_ds = json.load(f)
        # print(self.lm)


            # a dictionary where each word has it's top 10 words
        # self.lm = lm # a dictionary where
    def set_game_state(self, words_in_play, map_in_play):
        self.words = words_in_play
        self.maps = map_in_play


    def get_clue(self):
        red_words = []
        bad_words = []

        # Creates Red-Labeled Word arrays, and everything else arrays
        for i in range(25):
            if self.words[i][0] == '*':
                continue
            elif self.maps[i] == "Assassin" or self.maps[i] == "Blue" or self.maps[i] == "Civilian":
                bad_words.append(self.words[i].lower())
            else:
                red_words.append(self.words[i].lower())

        initial_state = random.choice([w for w in self.words if w[0]!='*'])
        # Start with a word and a number of guesses
        # problem = MyProblem(initial_state,)
        problem = MyProblem(initial_state, self.pre_processed_ds, self.lm, red_words, bad_words)
        problem.steps = 50000  # Number of iterations
        problem.Tmax = 1.0  # Initial temperature
        problem.Tmin = 0.01  # Final temperature
        # problem.schedule = 'geometric'  # Cooling schedule
        problem.schedule = 'exponential'  # Cooling schedule
        best_state, best_energy = problem.anneal()

        return best_state.lower(), 1

