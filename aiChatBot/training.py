# Training file for training the chatbot
# Another file will be created for USING the chatbot

import json
import random
import pickle
import numpy as np

import nltk
# Lemmatizer will reduce a word to its stem, generalizing our provided ex's
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Set up lemmatizer constructor from natural language tool kit
lemmatizer = WordNetLemmatizer()

# Load provided json file
intents = json.loads(open('intents.json').read())

# Create empty lists for the words, labels, and documents (combinations)
words = []
labels = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Now we iterate over the intents
# imagine the intents as a python dictionary with subvalues such as hello, bye
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize means to get a text and split it up into individual words
        # ex) "hi there you" ==> "hi" "there" "you"
        word_list = nltk.word_tokenize(pattern)
        # Instead of appending, we want to take the content and appending it
        # to the list, as opposed to taking the list and appending it to
        # the list
        words.extend(word_list)
        #print(words)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
            #print(labels)
#print(training_data)

# Takes out ignored letters then sorts it into a set
# A set will NOT have duplicates, thus we eliminate them automatically
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters] #edited
words = sorted(list(set(words))) #edited
#print(words)
labels = sorted(list(set(labels))) #edited

#print(labels)

# Write to a file for later use
# wb = writing binaries
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(labels, open('labels.pkl', 'wb'))

# ML part: So far, our problem is we only have words, classes, chars.
# But these are not numerical values, which is what neural networks use
# So we need to represent these words as numerical values. So we use
# bag_of_words which sets the individual word values to either 0 or 1
# depending if it occurs in that particular pattern

# All the documents will be in the training list
training_set = []
# Template of 0's with length of # of classes
output_empty = [0] * len(labels)
#print(output_empty)

# for each document, create an empty bag of words
# Here, we process our document data to go into our training list set
for doc in documents:
    bag = []
    word_patterns = doc[0]
    #print(word_patterns)
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    #print(word_patterns)
    for word in words:
        # If the word is found in the pattern, append it to the bag or words
        bag.append(1) if word in word_patterns else bag.append(0)

    # copy template
    output_row = list(output_empty)
    # know the class at index 1
    output_row[labels.index(doc[1])] = 1
    training_set.append([bag, output_row])

# randomize set
random.shuffle(training_set)
training_set = np.array(training_set)

# Features and labels used to train neural network
# train_x will be list of everything in the 0th dimension, y will be 1st dimension
train_x = list(training_set[:, 0])
train_y = list(training_set[:, 1])

# NOW, we can start to build the neural network model
# Neural Networks always must have an input layer, hidden layer, and single out
# put layer
model = Sequential()
# Adding a regular densely-connected NN layer, with the amt of neurons
# Activation is the type of function to use
#
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# Dropout prevents overfitting:
# It randomly drops neurons from the neural network during training in each iteration.
# This way, the network learns to use all of its inputs not always present during
# training
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# Softmax scales results in output layer into percentages of result into 1
model.add(Dense(len(train_y[0]), activation='softmax'))

# SGD Gradient descent (with momentum) optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# epochs = number of times to feed data into neural network
# verbose = 1 for a medium amt of information
hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("Done")
print(documents)