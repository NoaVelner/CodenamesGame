import sys
import importlib
import argparse
import time
import os

from game import Game
from players.guesser import *
from players.codemaster import *

class GameRun:
    """Class that builds and runs a Game based on command line arguments"""

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Run the Codenames AI competition game.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("codemaster", help="import string of form A.B.C.MyClass or 'human'")
        parser.add_argument("guesser", help="import string of form A.B.C.MyClass or 'human'")
        parser.add_argument("--seed", help="Random seed value for board state -- integer or 'time'", default='time')

        parser.add_argument("--w2v", help="Path to w2v file or None", default=None)
        parser.add_argument("--glove", help="Path to glove file or None", default=None)
        parser.add_argument("--wordnet", help="Name of wordnet file or None, most like ic-brown.dat", default=None)
        parser.add_argument("--glove_cm", help="Path to glove file or None", default=None)
        parser.add_argument("--glove_guesser", help="Path to glove file or None", default=None)

        parser.add_argument("--no_log", help="Supress logging", action='store_true', default=False)
        parser.add_argument("--no_print", help="Supress printing", action='store_true', default=False)
        parser.add_argument("--game_name", help="Name of game in log", default="default")

        parser.add_argument("--num_games", help="Number of games to run", default=1, type=int)

        args = parser.parse_args()

        self.do_log = not args.no_log
        self.do_print = not args.no_print
        self.save_stdout = sys.stdout
        if not self.do_print:
            sys.stdout = open(os.devnull, 'w')
        self.set_game_name(args.game_name)

        self.g_kwargs = {}
        self.cm_kwargs = {}

        # load codemaster class
        if args.codemaster == "human":
            self.codemaster = HumanCodemaster
            print('human codemaster')
        else:
            self.codemaster = self.import_string_to_class(args.codemaster)
            print('loaded codemaster class')

        # load guesser class
        if args.guesser == "human":
            self.guesser = HumanGuesser
            print('human guesser')
        else:
            self.guesser = self.import_string_to_class(args.guesser)
            print('loaded guesser class')

        # if the game is going to have an ai, load up word vectors
        if sys.argv[1] != "human" or sys.argv[2] != "human":
            if args.wordnet is not None:
                brown_ic = Game.load_wordnet(args.wordnet)
                self.g_kwargs["brown_ic"] = brown_ic
                self.cm_kwargs["brown_ic"] = brown_ic
                print('loaded wordnet')

            if args.glove is not None:
                glove_vectors = Game.load_glove_vecs(args.glove)
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
        game = Game(game_setup.codemaster,
                    game_setup.guesser,
                    seed=game_setup.seed,
                    do_print=game_setup.do_print,
                    do_log=game_setup.do_log,
                    game_name=game_setup.game_name,
                    cm_kwargs=game_setup.cm_kwargs,
                    g_kwargs=game_setup.g_kwargs)

        game.run()
        if game_setup.num_games > 1:
            print_progress_bar(game_setup, i + 1, game_setup.num_games)
            game_setup.update_seed()
