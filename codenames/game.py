import random
import time
import json
import enum
import os
import shutil
import sys

import colorama
import gensim.models.keyedvectors as word2vec
import numpy as np
from nltk.corpus import wordnet_ic

class GameCondition(enum.Enum):
    """Enumeration that represents the different states of the game"""
    HIT_RED = 0
    HIT_BLUE = 1
    HIT_ASSASSIN = 2
    LOSS = 3
    WIN = 4
    CONTINUE = 5



class Game:
    """Class that setups up game details and calls Guesser/Codemaster pair to play the game
    """

    def __init__(self, codemasters, guessers, two_teams,
                 seed="time", do_print=True, do_log=True, game_name="default",
                 cm_kwargs={}, g_kwargs={}):
        """ Setup Game details

        Args:
            codemaster (:class:`Codemaster`):
                Codemaster (spymaster in Codenames' rules) class that provides a clue.
            guesser (:class:`Guesser`):
                Guesser (field operative in Codenames' rules) class that guesses based on clue.
            seed (int or str, optional): 
                Value used to init random, "time" for time.time(). 
                Defaults to "time".
            do_print (bool, optional): 
                Whether to keep on sys.stdout or turn off. 
                Defaults to True.
            do_log (bool, optional): 
                Whether to append to log file or not. 
                Defaults to True.
            game_name (str, optional): 
                game name used in log file. Defaults to "default".
            cm_kwargs (dict, optional): 
                kwargs passed to Codemaster.
            g_kwargs (dict, optional): 
                kwargs passed to Guesser.
        """

        self.game_start_time = time.time()
        colorama.init()

        self.do_print = do_print
        if not self.do_print:
            self._save_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        self.codemasters = [codemaster(**cm_kwargs) for codemaster in codemasters]
        self.guessers = [guesser(**g_kwargs) for guesser in guessers]

        self.two_teams = two_teams

        self.num_cards = [7,7]
        self.current_team = 0
        self.choose_starting_team()
        self.num_cards[self.current_team] += 1

        self.cm_kwargs = cm_kwargs
        self.g_kwargs = g_kwargs
        self.do_log = do_log
        self.game_name = game_name
        self.colors = ["Red, Blue"]


        # set seed so that board/keygrid can be reloaded later
        if seed == 'time':
            self.seed = time.time()
            random.seed(self.seed)
        else:
            self.seed = seed
            random.seed(int(seed))

        print("seed:", self.seed)

        # load board words
        with open("game_wordpool.txt", "r") as f:
            temp = f.read().splitlines()
            assert len(temp) == len(set(temp)), "game_wordpool.txt should not have duplicates"
            random.shuffle(temp)
            self.words_on_board = temp[:25]

        # set grid key for codemaster (spymaster)
        self.key_grid = ["Red"] * self.num_cards[0] + ["Blue"] * self.num_cards[1] + ["Civilian"] * 9 + ["Assassin"]
        random.shuffle(self.key_grid)

    def choose_starting_team(self):
        if self.two_teams:
            self.current_team = 0 if random.choice(['Red', 'Blue']) == 'Red' else 1

    def __del__(self):
        """reset stdout if using the do_print==False option"""
        if not self.do_print:
            sys.stdout.close()
            sys.stdout = self._save_stdout

    @staticmethod
    def load_glove_vecs(glove_file_path, weights_file_path=None):
        """Load stanford nlp glove vectors
        Original source that matches the function: https://nlp.stanford.edu/data/glove.6B.zip
        """
        weight_matrix = []
        if weights_file_path is not None:
            with open(weights_file_path, encoding="utf-8") as infile:
                for line in infile:
                    line = line.rstrip().split(' ')
                    weight_matrix.append(np.array([float(n) for n in line]))

        with open(glove_file_path, encoding="utf-8") as infile:
            glove_vecs = {}
            for line in infile:
                line = line.rstrip().split(' ')
                if weights_file_path is not None:
                    glove_vecs[line[0]] = np.dot(np.array([float(n) for n in line[1:]]), weight_matrix)
                else:
                    glove_vecs[line[0]] = np.array([float(n) for n in line[1:]])
            return glove_vecs

    @staticmethod
    def load_wordnet(wordnet_file):
        """Function that loads wordnet from nltk.corpus"""
        return wordnet_ic.ic(wordnet_file)

    @staticmethod
    def load_w2v(w2v_file_path):
        """Function to initalize gensim w2v object from Google News w2v Vectors
        Vectors Source: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
        """
        return word2vec.KeyedVectors.load_word2vec_format(w2v_file_path, binary=True, unicode_errors='ignore')

    def _display_board_codemaster(self):
        """prints out board with color-paired words, only for codemaster, color && stylistic"""
        print(str.center("___________________________BOARD___________________________\n", 60))
        counter = 0
        for i in range(len(self.words_on_board)):
            if counter >= 1 and i % 5 == 0:
                print("\n")
            if self.key_grid[i] == 'Red':
                print(str.center(colorama.Fore.RESET + self.words_on_board[i], 15), " ", end='')
                # print(str.center(colorama.Fore.RED + self.words_on_board[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Blue':
                print(str.center(colorama.Fore.RESET + self.words_on_board[i], 15), " ", end='')
                # print(str.center(colorama.Fore.BLUE + self.words_on_board[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Civilian':
                print(str.center(colorama.Fore.RESET + self.words_on_board[i], 15), " ", end='')
                counter += 1
            else:
                print(str.center(colorama.Fore.RESET + self.words_on_board[i], 15), " ", end='')
                # print(str.center(colorama.Fore.MAGENTA + self.words_on_board[i], 15), " ", end='')
                counter += 1
        print(str.center(colorama.Fore.RESET +
                         "\n___________________________________________________________", 60))
        print("\n")

    def _display_board(self):
        """prints the list of words in a board like fashion (5x5)"""
        print(colorama.Style.RESET_ALL)
        print(str.center("___________________________BOARD___________________________", 60))
        for i in range(len(self.words_on_board)):
            if i % 5 == 0:
                print("\n")
            print(str.center(self.words_on_board[i], 10), " ", end='')

        print(str.center("\n___________________________________________________________", 60))
        print("\n")

    def _display_key_grid(self):
        """ Print the key grid to stdout  """
        print("\n")
        print(str.center(colorama.Fore.RESET +
                         "____________________________KEY____________________________\n", 55))
        counter = 0
        for i in range(len(self.key_grid)):
            if counter >= 1 and i % 5 == 0:
                print("\n")
            if self.key_grid[i] == 'Red':
                print(str.center(colorama.Fore.RESET + self.key_grid[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Blue':
                print(str.center(colorama.Fore.RESET + self.key_grid[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Civilian':
                print(str.center(colorama.Fore.RESET + self.key_grid[i], 15), " ", end='')
                counter += 1
            else:
                print(str.center(colorama.Fore.RESET + self.key_grid[i], 15), " ", end='')
                counter += 1
        print(str.center(colorama.Fore.RESET +
                         "\n___________________________________________________________", 55))
        print("\n")

    def get_words_on_board(self):
        """Return the list of words that represent the board state"""
        return self.words_on_board

    def get_key_grid(self):
        """Return the codemaster's key"""
        return self.key_grid

    def _accept_guess(self, guess_index):
        """Function that takes in an int index called guess to compare with the key grid
        CodeMaster will always win with Red and lose if Blue =/= 7 or Assassin == 1
        """
        if self.key_grid[guess_index] == "Red":
            self.words_on_board[guess_index] = "*Red*"
            if self.words_on_board.count("*Red*") >= self.num_cards[0]:
                return GameCondition.WIN
            elif self.current_team == 1 and self.two_teams:
                self.current_team = 1- self.current_team
                return GameCondition.CONTINUE
            return GameCondition.HIT_RED

        elif self.key_grid[guess_index] == "Blue":
            self.words_on_board[guess_index] = "*Blue*"
            if self.words_on_board.count("*Blue*") >= self.num_cards[1]:
                print("Blue Won!")
                return GameCondition.LOSS
            elif self.current_team == 0 and self.two_teams:
                self.current_team = 1 - self.current_team
                return GameCondition.CONTINUE
            return GameCondition.HIT_BLUE

        elif self.key_grid[guess_index] == "Assassin":
            self.words_on_board[guess_index] = "*Assassin*"
            return GameCondition.HIT_ASSASSIN

        else:
            self.words_on_board[guess_index] = "*Civilian*"
            return GameCondition.CONTINUE

    def write_results(self, num_of_turns, certainties):
        """Logging function
        writes in both the original and a more detailed new style
        """
        print(certainties)
        red_result = 0
        blue_result = 0
        civ_result = 0
        assa_result = 0

        for i in range(len(self.words_on_board)):
            if self.words_on_board[i] == "*Red*":
                red_result += 1
            elif self.words_on_board[i] == "*Blue*":
                blue_result += 1
            elif self.words_on_board[i] == "*Civilian*":
                civ_result += 1
            elif self.words_on_board[i] == "*Assassin*":
                assa_result += 1
        total = red_result + blue_result + civ_result + assa_result

        if not os.path.exists("results"):
            os.mkdir("results")

        with open("results/bot_results.txt", "a") as f:
            f.write(
                f'TOTAL:{num_of_turns} B:{blue_result} C:{civ_result} A:{assa_result}'
                f' R:{red_result} CM:{type(self.codemasters[0]).__name__} '
                f'GUESSER:{type(self.guessers[0]).__name__} SEED:{self.seed}\n'
            )
        with open("results/certainties.txt", "a") as f:
            f.write(f"{certainties}\n")

        with open("results/bot_results_new_style.txt", "a") as f:
            results = {"game_name": self.game_name,
                       "total_turns": num_of_turns,
                       "R": red_result, "B": blue_result, "C": civ_result, "A": assa_result,
                       "codemaster": type(self.codemasters[0]).__name__,
                       "guesser": type(self.guessers[0]).__name__,
                       "seed": self.seed,
                       "time_s": (self.game_end_time - self.game_start_time),
                       "cm_kwargs": {k: v if isinstance(v, float) or isinstance(v, int) or isinstance(v, str) else None
                                     for k, v in self.cm_kwargs.items()},
                       "g_kwargs": {k: v if isinstance(v, float) or isinstance(v, int) or isinstance(v, str) else None
                                    for k, v in self.g_kwargs.items()},
                       }
            f.write(json.dumps(results))
            f.write('\n')


    @staticmethod
    def clear_results():
        """Delete results folder"""
        if os.path.exists("results") and os.path.isdir("results"):
            shutil.rmtree("results")

    def run(self):
        """Function that runs the codenames game between codemaster and guesser"""
        game_condition = GameCondition.CONTINUE
        game_counter = 0
        opponent_colors = ["Blue", "Red"]
        while game_condition != GameCondition.LOSS and game_condition != GameCondition.WIN and game_condition != GameCondition.HIT_ASSASSIN:
            # board setup and display
            print('\n' * 2)
            words_in_play = self.get_words_on_board()
            current_key_grid = self.get_key_grid()
            self.codemasters[self.current_team].set_game_state(words_in_play, current_key_grid)
            # self._display_key_grid()
            self._display_board_codemaster()

            # codemaster gives clue & number here
            clue, clue_num = self.codemasters[self.current_team].get_clue(opponent_colors[self.current_team])
            game_counter += 1 if self.current_team == 0 else 0
            keep_guessing = True
            guess_num = 0
            clue_num = int(clue_num)

            print('\n' * 2)
            self.guessers[self.current_team].set_clue(clue, clue_num)

            game_condition = GameCondition.HIT_RED if self.current_team == 0 else GameCondition.HIT_BLUE
            turn_condition = game_condition

            while guess_num < clue_num and keep_guessing and game_condition == turn_condition:
                print(f"Round {game_counter}, Red Turn: " if self.current_team==0 else "Blue Turn: ")
                self.guessers[self.current_team].set_board(words_in_play)
                guess_answer = self.guessers[self.current_team].get_answer()

                # if no comparisons were made/found than retry input from codemaster
                if guess_answer is None or guess_answer == "no comparisons":
                    break
                guess_answer_index = words_in_play.index(guess_answer.upper().strip())
                game_condition = self._accept_guess(guess_answer_index)

                if (game_condition == GameCondition.HIT_RED and self.current_team == 0) or \
                        (game_condition == GameCondition.HIT_BLUE and self.current_team == 1):
                    print('\n' * 2)
                    self._display_board_codemaster()
                    guess_num += 1
                    print("Keep Guessing? the clue is ", clue, clue_num)
                    keep_guessing = self.guessers[self.current_team].keep_guessing()

                # if guesser selected a civilian or a blue-paired word
                elif game_condition == GameCondition.CONTINUE:
                    break

                elif game_condition == GameCondition.HIT_ASSASSIN:
                    self.game_end_time = time.time()
                    game_counter = 25
                    self._display_board_codemaster()
                    if self.do_log:
                        print()
                        print()
                        self.write_results(game_counter, self.guessers[0].get_certainty())
                        self.write_results(game_counter, self.guessers[0])
                    print("You Lost")
                    print("Game Counter:", game_counter)
                    # self.guessers[self.current_team].plot_unique_guess_counts()
                    # self.guessers[self.current_team].plot_certainty_of_chosen_guess()

                elif game_condition == GameCondition.LOSS:
                    self.game_end_time = time.time()
                    self._display_board_codemaster()
                    if self.do_log:
                        self.write_results(game_counter, self.guessers[0].get_certainty())
                        self.write_results(game_counter, self.guessers[0])
                    print(f"You Lost, Blue Won")
                    print("Game Counter:", game_counter)
                    # self.guessers[self.current_team].plot_unique_guess_counts()
                    # self.guessers[self.current_team].plot_certainty_of_chosen_guess()


                elif game_condition == GameCondition.WIN:
                    self.game_end_time = time.time()
                    self._display_board_codemaster()
                    if self.do_log:
                        self.write_results(game_counter, self.guessers[0])
                        # self.write_results(game_counter, self.guessers[0].get_certainty())
                    print("You Won")
                    print("Game Counter:", game_counter)
                    # self.guessers[self.current_team].plot_unique_guess_counts()
                    # self.guessers[self.current_team].plot_certainty_of_chosen_guess()

            if self.two_teams:
                self.current_team = 1 - self.current_team