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

# Load words, classes and model data that were saved through training
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot.h5")

# Clean up sentence - get the indivual words and lemmatize
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentece_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for sentence_word in sentece_words:
        for i, word in enumerate(words):
            if word == sentence_word:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bag = bag_of_words(sentence)
    res = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,  r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(data_list, data_json):
    # print(data_list)
    # print(data_json)
    tag = data_list[0]['intent']
    data = data_json['data']
    for i in data:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Chatbot is ready!")

while True:
    message = input("You: ")
    predict = predict_class(message)
    res = get_response(predict, data)
    print(res)
