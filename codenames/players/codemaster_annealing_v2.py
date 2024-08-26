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
        legal_words = [item for item in self.pre_processed_ds[self.state[0].lower()] if item.upper() not in self.good_words and item.upper() not in self.bad_words]
        # weights = [len(legal_words) - i for i in range(len(legal_words))]
        # Choose a word randomly with weights
        chosen_word = random.choices(legal_words, k=1)
        num = random.randint(1, len(self.good_words))
        new_state = chosen_word[0]
        self.state = (new_state, num)

    def energy(self):
        score = 0
        cur_word = self.state[0] #TOOD MAKE IT CHOOSE A WORD FROM THE 7K and not from the 25
        num = self.state[1]
        available_words = []
        for word in self.good_words:
            available_words.append(word.lower())
        for word in self.bad_words:
            available_words.append(word.lower())
        available_words.sort(key=lambda x: np.linalg.norm(self.lm[cur_word.lower()] - self.lm[x.lower()]))
        best_words = available_words[:num]

        for i, w in enumerate(best_words): #TODO BIGGER PUNISHMENT IF IT'S RELATED TO ASSASSIN.
            if w in self.bad_words:
                score += 10 / (i+1)

        best_words_avg = np.mean([self.lm[w] for w in best_words], axis=0)
        dist = np.linalg.norm(self.lm[cur_word.lower()] - best_words_avg)
        score += dist/num
        return score
        # return dist / num




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
        random_red_word = random.choice(red_words)
        initial_state = (random.choice([w for w in self.pre_processed_ds[random_red_word.lower()]]), 3) #TODO CHANGE NUMBER
        # Start with a word and a number of guesses
        # problem = MyProblem(initial_state,)
        problem = MyProblem(initial_state, self.pre_processed_ds, self.lm, red_words, bad_words)
        problem.steps = 50000  # Number of iterations
        problem.Tmax = 1.0  # Initial temperature
        problem.Tmin = 0.01  # Final temperature
        # problem.schedule = 'geometric'  # Cooling schedule
        problem.schedule = 'exponential'  # Cooling schedule
        best_state, best_energy = problem.anneal()

        return best_state[0].lower(), best_state[1]

