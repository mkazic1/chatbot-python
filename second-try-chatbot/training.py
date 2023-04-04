import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

# Part 1 - Preparations for training - load and prepare data from data.json file
# Object created to reduce words to its stems/lemmas in order to not lose on the performance
lemmatizer = WordNetLemmatizer()

# Load the data for training
data = json.loads(open("data.json").read())

words = []
classes = []
documents = []
ignore_characters = ["?", "!", ".", ","]

# Extract data for training - Getting words from patterns
for sample in data["data"]:
    for pattern in sample["patterns"]:
        # Split text into a list of individual words with tokenize and form a list of words
        word_list = nltk.word_tokenize(pattern)
        # Add it to words, documents and classes list with coresponding additional fields
        words.extend(word_list)
        documents.append((word_list, sample["tag"]))
        if sample["tag"] not in classes:
            classes.append(sample["tag"])

# Lemmatize words, eliminate duplicates and sort words
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_characters]
words = sorted(set(words))

# Saving data with pickle library so they can be used later for training the chatbot
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))
pickle.dump(documents, open("documents.pkl", "wb"))

# Part 2 - Training data - Machine learning part and neural networks
# Resulting list
training = []
# Will be used for creating a one-hot encoded output vector for each training example
output_empty = [0] * len(classes)

# Process of converting words into numerical values
for document in documents:
    # For every document making a list/bag of words that it contains
    bag = []
    word_list = document[0]
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]
    for word in words:
        bag.append(1) if word in word_list else bag.append(0)
    # Copying list
    output_row = list(output_empty)
    # Setting the index corresponding to the class of the current example to 1
    output_row[classes.index(document[1])] = 1
    # Setting pairs of input vectors (bag) and corresponding output vectors (one-hot encoded)
    training.append([bag, output_row])

# Shuffle the data
random.shuffle(training)
training = np.array(training)

# Part 3 - Building the Neural network
# List of inputs
train_x = list(training[:, 0])
# List of outputs
train_y = list(training[:, 1])

# Creating the Neural network
model = Sequential()
model.add(Dense(120, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compiling the network
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Training the network
trained_model = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save("chatbot.h5", trained_model)

# Completed running
print("Done")
