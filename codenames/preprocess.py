import json
import sys

from scipy.spatial.distance import cosine

import gensim.downloader as api

from codenames.run_game import print_progress_bar

model_name = 'word2vec-google-news-300'

# Load pre-trained GloVe vectors (e.g., 50-dimensional vectors)
model = api.load(model_name)

# Load your dataset of 7,500 words
with open('combine_words.txt', 'r') as file:
    word_dataset = [line.strip().lower() for line in file.readlines()]

print(word_dataset)

with open('players/cm_wordlist.txt', 'r') as file:
    master_dataset = [line.strip().lower() for line in file.readlines()]


def get_closest_words_within_dataset(word, word_list, model, max_words=20):
    """
    Get the 20 closest words to 'word' from 'word_list' using the 'model'.
    Only considers words within 'word_list'.
    """
    if word not in model:
        return []

    distances = []
    for other_word in word_list:
        if other_word == word or other_word not in model:
            continue
        distance = cosine(model[word], model[other_word])
        distances.append((other_word, distance))

    # Sort by distance and select the top max_words closest words
    closest_words = sorted(distances, key=lambda x: x[1])[:max_words]
    return [word for word, _ in closest_words]


closest_words_dataset = {}

length = len(word_dataset)
precentage = 0

for i, word in enumerate(word_dataset):
    if i*100/length > precentage:
        print_progress_bar(iteration=i, total=length)
        precentage = precentage + 1
    closest_words_dataset[word] = get_closest_words_within_dataset(word, master_dataset, model, max_words=20)


with open(f'closest_combined_words_within_{model_name}.json', 'w') as f:
    json.dump(closest_words_dataset, f)

print(closest_words_dataset)