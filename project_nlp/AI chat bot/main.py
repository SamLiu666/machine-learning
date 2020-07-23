import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
# import tflearn
import tensorflow as tf
import random
import json

with open("intents.json") as f:
    data = json.load(f)

words = []
labels = []
# docs = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])


words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

# one-hot
for x, doc in enumerate(docs_x):

    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
print(training.shape)
output = numpy.array(output)

############## model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(46,)),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(6)
])

model.summary()

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

try:
    model.load("chatbot.h5")
except:
    history = model.fit(training,
                        output,
                        epochs=23,
                        batch_size=512,
                        verbose=1)
    model.save("chatbot.h5")
def bag_of_words(s, words):
    bag = []
    print(len(bag))

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag.append(1)
            else:
                bag.append(0)
    bag = numpy.array([bag])  # keep the dimention
    # print(bag.shape)
    return bag

def chat():
    s = words
    print("stat talking with the bot: ")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict(bag_of_words(inp, s))
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        # print(results)
        # print(tag)
        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
        print(random.choice(responses))
# s = bag_of_words('hello', words)
# print(s.shape)
chat()