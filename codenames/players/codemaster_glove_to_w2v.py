import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

from players.codemaster_annealing_v4 import AICodemaster

def cosine_distance_loss(y_true, y_pred):
    # Calculate the cosine similarity and convert it to a distance
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    return 1 - tf.reduce_sum(y_true * y_pred, axis=-1)


def learn_vector_relationship(dict1, dict2, key_list, epochs=64, batch_size=32):
    # Extract input (X) and output (Y) data based on the key list
    X = np.array([dict1[key.lower()] for key in key_list])
    Y = np.array([dict2[key.lower()] for key in key_list])

    # Define a simple feedforward neural network
    model = models.Sequential()
    model.add(layers.Input(shape=(X.shape[1],)))  # Input layer with number of units matching X's dimensionality
    model.add(layers.Dense(300, activation='relu'))  # Hidden layer with 128 units
    model.add(layers.Dense(Y.shape[1]))  # Output layer with number of units matching Y's dimensionality

    # Compile the model
    model.compile(optimizer='adam', loss=cosine_distance_loss)

    # Train the model with validation
    model.fit(X, Y, epochs=epochs, batch_size=batch_size)

    return model

class Translator(AICodemaster):

    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
        wordlist = []
        with open('combine_words.txt') as infile:
            for line in infile:
                wordlist.append(line.rstrip().lower())

        model = learn_vector_relationship(glove_vecs, word_vectors, wordlist)

        # Prepare the input matrix for all words in the wordlist
        input_vectors = np.array([glove_vecs[word.lower()] for word in wordlist])

        # Predict all vectors at once
        predicted_vectors = model.predict(input_vectors)

        # Create the translated_vecs dictionary
        translated_vecs = {word.lower(): predicted_vectors[i] for i, word in enumerate(wordlist)}

        super().__init__(brown_ic, glove_vecs, translated_vecs)

    def set_game_state(self, words, maps):
        super().set_game_state(words, maps)

    def get_clue(self, bad_color="Blue", good_color = "Red"):
        return super().get_clue(bad_color, good_color)

