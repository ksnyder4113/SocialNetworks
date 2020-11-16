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
    actor_dict = {}
    #should open a CSV file and extract its data into a dictionary structure
    #person --> name of actor who is speaking
    #sentence --> sentence that person has said
    #make sure all entries are changed to lowercase before being added into the dictionary

    return training_data


def preprocess_words(words, stemmer):
    """ Stems each word in the words list and returns a list of these word stems without duplicates."""

    word_stems = ()
    preprocessed_words = []

    for word in words:
        stemmed_word = stemmer.stem(word)
        word_stems.add(stemmed_word)

    for stem in word_stems:
        new_words_list.append(stem)
    
    return pre_processed_words

    



def main():
    stemmer = LancasterStemmer()

    raw_training_data = get_raw_training_data('dialogue_data.csv')


if __name__ == ""__main__"":
    main()  