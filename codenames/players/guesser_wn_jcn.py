import random
from operator import itemgetter

from nltk.corpus import wordnet

from players.guesser import Guesser


class AIGuesser(Guesser):

    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
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
        print("The clue is:", clue, num)
        li = [clue, num]
        return li

    def keep_guessing(self):
        return self.num > 0

    # def get_answer(self):
    #     sorted_results = self.wordnet_synset(self.clue, self.words)
    #     if not sorted_results:
    #         # Fallback: random choice excluding already guessed words
    #         remaining_words = [word for word in self.words if not word.startswith('*')]
    #         choice = random.choice(remaining_words) if remaining_words else None
    #         print(f"No strong match found. Random guess: '{choice}'")
    #         return choice
    #     # Select the best guess
    #     best_guess = sorted_results[0]['word']
    #     self.num -= 1
    #     print(f"Guesses (top 3): {[res['word'] for res in sorted_results[:3]]}")
    #     return best_guess

    def get_answer(self):
        sorted_results = self.wordnet_synset(self.clue, self.words)
        if not sorted_results:
            choice = "*"
            while choice[0] is '*':
                choice = random.choice(self.words)
            return choice
        print(f'guesses: {sorted_results}')
        self.num -= 1
        return sorted_results[0][5]


    def wordnet_synset(self, clue, board):
        jcn_results = []
        count = 0
        for i in board:
            for clue_list in wordnet.synsets(clue):
                jcn_clue = 0
                for board_list in wordnet.synsets(i):
                    try:
                        # only if the two compared words have the same part of speech
                        jcn = clue_list.jcn_similarity(board_list, self.brown_ic)
                    except :
                        continue
                    if jcn:
                        jcn_results.append(("jcn: ", jcn, count, clue_list, board_list, i))
                        if jcn > jcn_clue:
                            jcn_clue = jcn

        # if results list is empty
        if not jcn_results:
            return []

        jcn_results = list(reversed(sorted(jcn_results, key=itemgetter(1))))
        return jcn_results[:3]

    # def wordnet_synset(self, clue, board):
    #     """
    #     Computes similarity scores between the clue and board words using WordNet and JCN similarity.
    #     Implements heuristic search and pruning for efficiency.
    #     """
    #     results = []
    #     clue_synsets = wordnet.synsets(clue, pos=wordnet.NOUN)  # Heuristic: consider nouns only
    #     if not clue_synsets:
    #         print(f"No synsets found for clue '{clue}'.")
    #         return []
    #
    #     # Heuristic: prioritize most common synsets
    #     clue_synsets = sorted(clue_synsets, key=lambda s: s.lexname(), reverse=True)[:3]
    #
    #     for board_word in board:
    #         board_synsets = wordnet.synsets(board_word, pos=wordnet.NOUN)
    #         if not board_synsets:
    #             continue
    #         # Prune: consider top 3 synsets per board word
    #         board_synsets = board_synsets[:3]
    #
    #         max_similarity = 0
    #         best_synset_pair = (None, None)
    #
    #         for clue_synset in clue_synsets:
    #             for board_synset in board_synsets:
    #                 try:
    #                     similarity = clue_synset.jcn_similarity(board_synset, self.brown_ic)
    #                     if similarity and similarity > max_similarity:
    #                         max_similarity = similarity
    #                         best_synset_pair = (clue_synset, board_synset)
    #                         # Early stopping if similarity is very high
    #                         if similarity > 0.9:
    #                             break
    #                 except (WordNetError, ZeroDivisionError):
    #                     continue
    #             if max_similarity > 0.9:
    #                 break
    #
    #         if max_similarity > 0:
    #             results.append({
    #                 'word': board_word,
    #                 'similarity': max_similarity,
    #                 'clue_synset': best_synset_pair[0],
    #                 'board_synset': best_synset_pair[1]
    #             })
    #
    #     # Final pruning: set similarity threshold
    #     similarity_threshold = 0.07  # Adjust based on experimentation
    #     filtered_results = [res for res in results if res['similarity'] >= similarity_threshold]
    #
    #     if not filtered_results:
    #         print("No results above similarity threshold.")
    #         return []
    #
    #     # Sort results by similarity in descending order
    #     sorted_results = sorted(filtered_results, key=lambda x: x['similarity'], reverse=True)
    #
    #     return sorted_results
