import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.python.keras.models import load_model

# Part 1
# To reduce words to its stems/lemmas in order to not lose on the performance
lemmatizer = WordNetLemmatizer()

# Load the data for training
data = json.loads(open("data.json").read())

# Load words, classes and model data used during training
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot.h5")

# Getting the indivual words and lemmatize
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Making a one-hot encoded vector containing presence(1) or absence(0) of each word from the sentence in the overall words of the model
def bag_of_words(sentence):
    sentece_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for sentence_word in sentece_words:
        for i, word in enumerate(words):
            if word == sentence_word:
                bag[i] = 1
    return np.array(bag)

# Getting the list of predicted intents and their probabilities
def predict_class(sentence):
    # Passing the one-hot encoded vector through pretrained neural network to predict the class/field of the sentence
    bag = bag_of_words(sentence)
    model_probabilities = model.predict(np.array([bag]))[0]

    # Bottom level condition for probabilities
    ERROR_THRESHOLD = 0.2
    # Results are all the probabilities that are greater that the bottom level, sorted in descending order
    results = [[i,  r] for i, r in enumerate(model_probabilities) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    # Return list of possible classes and their probabilities, the highest first
    return_list = []
    for r in results:
        return_list.append({"data": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(data_list, data_json):
    # Get the class with the highest probability
    tag = data_list[0]['data']
    data = data_json['data']
    for i in data:
        # Check if tags are matching
        if i['tag'] == tag:
            # Result is picked random from the responses object
            result = random.choice(i['responses'])
            break
    return result

print("Chatbot is ready!")

while True:
    message = input("You: ")
    predict = predict_class(message)
    res = get_response(predict, data)
    print(res)
