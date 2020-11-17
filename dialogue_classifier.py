import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import csv
import numpy as np
import time


def get_raw_training_data(filename):
    """ """
    training_data = []

    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        line = 0
        for row in reader:
            actor_dict = {}
            name = row[0].lower()
            sentence = row[1].lower()
            actor_dict["person"] = name
            actor_dict["sentence"] = sentence
            print(name + ": " + sentence)
            
            training_data.append(actor_dict)
            
            line += 1
            
    return training_data


def organize_raw_training_data(raw_training_data, stemmer):
""" """
    words = []
    classes = []
    documents = []

    for element in raw_training_data:
        name = element["person"]
        sentence = element["sentence"]
        tokens = nltk.word_tokenize(sentence)
        words.extend(tokens)
        
        if name not in classes:
            classes.append(name)
        
        actor_tup = (tokens, name)
        documents.append(actor_tup)

    preprocessed_words = preprocess_words(words, stemmer)

    return preprocessed_words, classes, documents


def preprocess_words(words, stemmer):
    """ Stems each word in the words list and returns a 
    list of these word stems without duplicates."""
    word_stems = set()
    preprocessed_words = []

    for word in words:
        stemmed_word = stemmer.stem(word)
        if stemmed_word != "?" or "!" or ".":
            word_stems.add(stemmed_word)
    
    for stem in word_stems:
        preprocessed_words.append(stem)
    
    return preprocessed_words


def create_training_data(words, classes, documents, stemmer):
    training_data = []
    output = []

    for line in documents:
        name = line[0]
        sentence = line[1]
        elements = []
        bag = []
        for elem in sentence.split():
            stem = stemmer.stem(elem)
            elements.append(stem)
        
        for word in words:
            if word in elements:
                bag.append(1)
            else:
                bag.append(0)

        class_list = []
        for actor_name in classes:
            if name == actor_name:
                class_list.append(1)
            else:
                class_list.append(0)
        
        training_data.append(bag)
        output.append(class_list)

    return training_data, output
    

def sigmoid(z):
    """Returns the basic sigmoid formula for z"""
    denominator = 1 + np.exp(-z)
    output = 1/denominator
    return output


def sigmoid_output_to_derivative(output):
    """Convert the sigmoid function's output to its derivative."""
    return output * (1-output)


def main():
    stemmer = LancasterStemmer()

    #raw_training_data = get_raw_training_data('dialogue_data.csv')
    raw_training_data = get_raw_training_data('test.csv')
    words, classes, documents = organize_raw_training_data(raw_training_data, stemmer)
    training_data, output = create_training_data(words, classes, documents, stemmer)

if __name__ == "__main__":
    main()  