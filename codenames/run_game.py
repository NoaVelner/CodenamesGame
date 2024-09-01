import sys
import importlib
import argparse
import time
import os

import numpy as np

from game import Game
from players.guesser import *
from players.codemaster import *

class GameRun:
    """Class that builds and runs a Game based on command line arguments"""

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Run the Codenames AI competition game.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--codemaster1", help="import string of form A.B.C.MyClass or 'human'", default='human')
        parser.add_argument("--guesser1", help="import string of form A.B.C.MyClass or 'human'", default='human')
        parser.add_argument("--codemaster2", help="import string of form A.B.C.MyClass or 'human'", default = None)
        parser.add_argument("--guesser2", help="import string of form A.B.C.MyClass or 'human'", default = None)
        parser.add_argument("--seed", help="Random seed value for board state -- integer or 'time'", default='time')

        parser.add_argument("--w2v", help="Path to w2v file or None", default=None)
        parser.add_argument("--w2v_weights", help="path to w2v weights file of None", default=None)
        parser.add_argument("--glove", help="Path to glove file or None", default=None)
        parser.add_argument("--glove_weights", help="path to glove weights file of None", default=None)
        parser.add_argument("--wordnet", help="Name of wordnet file or None, most like ic-brown.dat", default=None)
        parser.add_argument("--glove_cm", help="Path to glove file or None", default=None)
        parser.add_argument("--glove_guesser", help="Path to glove file or None", default=None)

        parser.add_argument("--two_teams", help="Creates a game with 2 competing teams", action='store_true', default=False)
        parser.add_argument("--no_log", help="Supress logging", action='store_true', default=False)
        parser.add_argument("--no_print", help="Supress printing", action='store_true', default=False)
        parser.add_argument("--game_name", help="Name of game in log", default="default")

        parser.add_argument("--num_games", help="Number of games to run", default=1, type=int)

        args = parser.parse_args()

        self.do_log = not args.no_log
        self.do_print = not args.no_print
        self.have_AI_player = False
        self.save_stdout = sys.stdout
        if not self.do_print:
            sys.stdout = open(os.devnull, 'w')
        self.set_game_name(args.game_name)

        self.g_kwargs = {}
        self.cm_kwargs = {}

        self.codemasters = []
        self.guessers = []


        #check for input correctness for 2 teams game
        self.two_teams = args.two_teams
        if self.two_teams and (args.codemaster2 is None or args.guesser2 is None):
            print("Error: two_teams is true but there's not info about the players")
            exit()

        # load codemaster class
        if args.codemaster1 == "human":
            self.codemasters.append(HumanCodemaster)
            print('human codemaster')
        else:
            self.codemasters.append(self.import_string_to_class(args.codemaster1))
            print('loaded codemaster class')
            self.have_AI_player = True

        if args.codemaster2 is not None:
            if args.codemaster2 == 'human':
                self.codemasters.append(HumanCodemaster)
                print('human codemaster2')
            else:
                self.codemasters.append(self.import_string_to_class(args.codemaster2))
                print('loaded codemaster2 class')
                self.have_AI_player = True

        # load guesser class
        if args.guesser1 == "human":
            self.guessers.append(HumanGuesser)
            print('human guesser')
        else:
            self.guessers.append(self.import_string_to_class(args.guesser1))
            print('loaded guesser class')
            self.have_AI_player = True

        if args.guesser2 is not None:
            if args.guesser1 == "human":
                self.guessers.append(HumanGuesser)
                print('human guesser2')
            else:
                self.guessers.append(self.import_string_to_class(args.guesser2))
                print('loaded guesser2 class')
                self.have_AI_player = True

        # if the game is going to have an ai, load up word vectors
        if self.have_AI_player:
            if args.wordnet is not None:
                brown_ic = Game.load_wordnet(args.wordnet)
                self.g_kwargs["brown_ic"] = brown_ic
                self.cm_kwargs["brown_ic"] = brown_ic
                print('loaded wordnet')

            if args.glove is not None:
                glove_vectors = Game.load_glove_vecs(args.glove, args.glove_weights)
                self.g_kwargs["glove_vecs"] = glove_vectors
                self.cm_kwargs["glove_vecs"] = glove_vectors
                print('loaded glove vectors')

            if args.w2v is not None:
                w2v_vectors = Game.load_w2v(args.w2v)
                self.g_kwargs["word_vectors"] = w2v_vectors
                self.cm_kwargs["word_vectors"] = w2v_vectors
                print('loaded word vectors')

            if args.glove_cm is not None:
                glove_vectors = Game.load_glove_vecs(args.glove_cm)
                self.cm_kwargs["glove_vecs"] = glove_vectors
                print('loaded glove vectors')

            if args.glove_guesser is not None:
                glove_vectors = Game.load_glove_vecs(args.glove_guesser)
                self.g_kwargs["glove_vecs"] = glove_vectors
                print('loaded glove vectors')

        # set seed so that board/keygrid can be reloaded later
        if args.seed == 'time':
            self.seed = time.time()
        else:
            self.seed = int(args.seed)

        self.num_games = args.num_games

    def set_game_name(self, game_name):
        self.game_name = game_name

    def update_seed(self):
        self.seed = time.time()


    def __del__(self):
        """reset stdout if using the do_print==False option"""
        if not self.do_print:
            sys.stdout.close()
            sys.stdout = self.save_stdout

    def import_string_to_class(self, import_string):
        """Parse an import string and return the class"""
        parts = import_string.split('.')
        module_name = '.'.join(parts[:len(parts) - 1])
        class_name = parts[-1]

        module = importlib.import_module(module_name)
        my_class = getattr(module, class_name)

        return my_class


def print_progress_bar(game_setup=None, iteration=0, total=1, length=50):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '#' * filled_length + '-' * (length - filled_length)
    if game_setup:
        game_setup.save_stdout.write(f'\rProgress: |{bar}| {percent}% Complete')
        game_setup.save_stdout.flush()
    else:
        sys.stdout.write(f'\rProgress: |{bar}| {percent}% Complete')

    if iteration == total:
        print()

if __name__ == "__main__":
    game_setup = GameRun()
    if game_setup.num_games > 1:
        game_setup.set_game_name(f"{game_setup.game_name}_{0}")
        print_progress_bar(game_setup, 0, game_setup.num_games)

    for i in range(game_setup.num_games):
        if game_setup.num_games > 1:
            game_setup.set_game_name(f"{game_setup.game_name[:-(len(str(i+1))+1)]}_{i+1}")
        game = Game(game_setup.codemasters,
                    game_setup.guessers,
                    seed=game_setup.seed,
                    do_print=game_setup.do_print,
                    do_log=game_setup.do_log,
                    two_teams= game_setup.two_teams,
                    game_name=game_setup.game_name,
                    cm_kwargs=game_setup.cm_kwargs,
                    g_kwargs=game_setup.g_kwargs)

        game.run()
        if game_setup.num_games > 1:
            print_progress_bar(game_setup, i + 1, game_setup.num_games)
            game_setup.update_seed()
