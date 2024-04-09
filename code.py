import json
import random
import nltk
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def bag_of_words(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    bag = [0]*len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)


def pickle_maker():
    words = []
    classes = []
    documents = []
    ignore_letters = ['!', '?', ',', '.']
    with open("./intents.json") as file:
        intents = json.load(file)

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word
            word = nltk.word_tokenize(pattern)
            words.extend(word)
            # add documents in the corpus
            documents.append((word, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # lemmaztize and lower each word and remove duplicates
    words = [lemmatizer.lemmatize(w.lower())
             for w in words if w not in ignore_letters]
    words = sorted(list(set(words)))

    # sort classes
    classes = sorted(list(set(classes)))
    # documents = combination between patterns and intents

    # create our training data
    training = []
    # create an empty array for our output
    output_empty = [0] * len(classes)
    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # lemmatize each word - create base word, in attempt to represent related words
        pattern_words = [lemmatizer.lemmatize(
            word.lower()) for word in pattern_words]
        # create our bag of words array with 1, if word match found in current pattern
        for word in words:
            bag.append(1) if word in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])
    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)
    # create train and test lists. X - patterns, Y - intents
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    with open('./data.pkl', 'wb') as file:
        pickle.dump((words, classes, train_x,
                     train_y), file)


pickle_maker()