import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from codenames.game import Game

# Custom loss function using cosine distance
def cosine_distance_loss(y_true, y_pred):
    # Calculate the cosine similarity and convert it to a distance
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    return 1 - tf.reduce_sum(y_true * y_pred, axis=-1)

def learn_vector_relationship(dict1, dict2, key_list, epochs=1024, batch_size=32, validation_split=0.2):
    # Extract input (X) and output (Y) data based on the key list
    X = np.array([dict1[key.lower()] for key in key_list])
    Y = np.array([dict2[key.lower()] for key in key_list])

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_split)

    # Define a simple feedforward neural network
    model = models.Sequential()
    model.add(layers.Input(shape=(X.shape[1],)))  # Input layer with number of units matching X's dimensionality
    model.add(layers.Dense(300, activation='relu'))  # Hidden layer with 128 units
    model.add(layers.Dense(Y.shape[1]))  # Output layer with number of units matching Y's dimensionality

    # Compile the model
    model.compile(optimizer='adam', loss=cosine_distance_loss)

    # Train the model with validation
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                        epochs=epochs, batch_size=batch_size)

    # Plot the training and validation loss over epochs
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs learning w2v from glove 300')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig("learning_loss_from_big_glove_to_w2v.png")
    plt.show()

    # Return the trained model
    return model

if __name__ == "__main__":
    # Example usage:
    glove_vectors = Game.load_glove_vecs("players/glove.6B.300d.txt")
    w2v_vectors = Game.load_w2v("players/GoogleNews-vectors-negative300.bin")
    wordlist = []
    with open('combine_words.txt') as infile:
        for line in infile:
            wordlist.append(line.rstrip().lower())
    model = learn_vector_relationship(glove_vectors, w2v_vectors, wordlist)

    # Prepare the input matrix for all words in the wordlist
    input_vectors = np.array([glove_vectors[word.lower()] for word in wordlist])

    # Predict all vectors at once
    predicted_vectors = model.predict(input_vectors)

    # Create the translated_vecs dictionary
    translated_vecs = {word.lower(): predicted_vectors[i] for i, word in enumerate(wordlist)}
    model.save('glove300_to_w2v_model.keras')

