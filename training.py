import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

# Part 1 - Preparations for training - load and prepare data from data.json file

# To reduce words to its stems/lemmas in order to not lose on the performance
lemmatizer = WordNetLemmatizer()

# Load the data from statics file for training
data = json.loads(open("data.json").read())

words = []
classes = []
documents = []
ignore_characters = ["?", "!", ".", ","]

# Extract data for training - Getting words from patterns
for sample in data["data"]:
    for pattern in sample["patterns"]:
        # Split sentences into a list of individual words with tokenize
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, sample["tag"]))
        if sample["tag"] not in classes:
            classes.append(sample["tag"])

# Lemmatize words to eliminate duplicates and sort words
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_characters]
words = sorted(set(words))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))
pickle.dump(documents, open("documents.pkl", "wb"))

# Part 2 - Training data - Machine learning part and neural networks

# Neural network needs numerical values --> words into numerical values
training = []
output_empty = [0] * len(classes)

# For every document making a list of bag that it contains
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    # Copying list
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle the data
random.shuffle(training)

# Part 3 - Building the Neural network
train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(120, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
