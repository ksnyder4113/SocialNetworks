import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import csv
import numpy as np
import time


def get_raw_training_data(filename):
    """
    Read in our csv of training data, creating a list of dictionaries to keep track of each
    person and sentence 

    @params:
        filename --> name of the csv file we are reading in
    
    @returns:
        training_data --> list of dictionaries, where each dictionary is a name and sentence
    """
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
    """
    @params:
        raw_training_data --> dictionary of names and sentences
        stemmer --> our Lancaster Stemmer object
    """
    words = []
    classes = []
    documents = []

    #loop through dictionaries inside training data list
    for element in raw_training_data:
        name = element["person"]
        sentence = element["sentence"]
        tokens = nltk.word_tokenize(sentence)
        words.extend(tokens)
        
        #if we have not come across this person, add it to classes list
        if name not in classes:
            classes.append(name)
        
        #tuple mapping our tokens to the actor name
        actor_tup = (tokens, name)
        documents.append(actor_tup)

    #preprocess our tokens 
    preprocessed_words = preprocess_words(words, stemmer)

    return preprocessed_words, classes, documents


def preprocess_words(words, stemmer):
    """ Stems each word in the words list and returns a 
    list of these word stems without duplicates."""
    word_stems = set()
    preprocessed_words = []

    #loop through the word tokens
    for word in words:
        stemmed_word = stemmer.stem(word)
        # we don't want punctuation included
        if stemmed_word != "?" or "!" or ".":
            word_stems.add(stemmed_word)
    
    #add our words without repeats to a list
    for stem in word_stems:
        preprocessed_words.append(stem)
    
    #return tokens with no repeats
    return preprocessed_words


def create_training_data(words, classes, documents, stemmer):
    """
    Create our training data based on the unique words, names, and sentences in the document.

    @params:
        words --> unique tokens in the document
        classes --> list of actor names
        documents --> list of tuples including the sentence and actor name
        stemmer --> our Lancaster Stemmer object

    @returns:
        training_data --> list of lists, where each element is a list whose length is the number
        of unique words, and the number of elements is the number of sentences. At each index,
        there is 0 if the word isn't in the sentence and 1 if it is
        output --> list of lists, where each element is a list whose length is the number
        of unique names, and the number of elements is the total number of names. At each index,
        there is 0 if the name isn't mapped to the sentence and 1 if it is

    """
    training_data = []
    output = []
    # loop through our sentences
    for line in documents:
        name = line[0]
        sentence = line[1]
        elements = []
        bag = []
        for elem in sentence.split():
            stem = stemmer.stem(elem)
            elements.append(stem)
        
        # we look through all our unique words; if a word in the sentence is in that list we put 1
        # at the corresponding index and 0 if not
        for word in words:
            if word in elements:
                bag.append(1)
            else:
                bag.append(0)

        # we do the same thing for the name, at the start of the sentence. 1 at the index
        # if that actor is the one preceding the sentence, 0 otherwise
        class_list = []
        for actor_name in classes:
            if name == actor_name:
                class_list.append(1)
            else:
                class_list.append(0)
        # our training data is fed all the vectors of word occurences
        training_data.append(bag)
        # our output is the vectors of name occurences
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


def feedforward(X, synapse_0, synapse_1):
    """Feed forward through layers 0, 1, and 2."""
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    return layer_0, layer_1, layer_2
    
def get_synapses(epochs, X, y, alpha, synapse_0, synapse_1):
    """Update our weights for each epoch."""
    # Initializations.
    last_mean_error = 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    synapse_0_direction_count = np.zeros_like(synapse_0)

    prev_synapse_1_weight_update = np.zeros_like(synapse_1)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    # Make an iterator out of the number of epochs we requested.
    for j in iter(range(epochs+1)):
        layer_0, layer_1, layer_2 = feedforward(X, synapse_0, synapse_1)

        # How much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # If this 10k iteration's error is greater than the last iteration,
            # break out.
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break

        # In what direction is the target value?  How much is the change for layer_2?
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # How much did each l1 value contribute to the l2 error (according to the weights)?
        # (Note: .T means transpose and can be accessed via numpy!)
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # In what direction is the target l1?  How much is the change for layer_1?
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        # Manage updates.
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if j > 0:
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    return synapse_0, synapse_1


def save_synapses(filename, words, classes, synapse_0, synapse_1):
    """Save our weights as a JSON file for later use."""
    now = datetime.datetime.now()

    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print("Saved synapses to:", synapse_file)

def init_synapses(X, hidden_neurons, classes):
    """Initializes our synapses (using random values)."""
    # Ensures we have a "consistent" randomness for convenience.
    np.random.seed(1)

    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    return synapse_0, synapse_1

def train(X, y, words, classes, hidden_neurons=10, alpha=1, epochs=50000):
    """Train using specified parameters."""
    print("Training with {0} neurons and alpha = {1}".format(hidden_neurons, alpha))

    synapse_0, synapse_1 = init_synapses(X, hidden_neurons, classes)

    # For each epoch, update our weights
    synapse_0, synapse_1 = get_synapses(epochs, X, y, alpha, synapse_0, synapse_1)

    # Save our work
    save_synapses("synapses.json", words, classes, synapse_0, synapse_1)

def start_training(words, classes, training_data, output):
    """Initialize training process and keep track of processing time."""
    start_time = time.time()
    X = np.array(training_data)
    y = np.array(output)

    train(X, y, words, classes, hidden_neurons=20, alpha=0.1, epochs=100000)

    elapsed_time = time.time() - start_time
    print("Processing time:", elapsed_time, "seconds")


def main():
    #allows us to find the stem of any word to determine its basic meaning
    stemmer = LancasterStemmer()

    #raw_training_data = get_raw_training_data('dialogue_data.csv')
    raw_training_data = get_raw_training_data('test.csv')
    words, classes, documents = organize_raw_training_data(raw_training_data, stemmer)
    training_data, output = create_training_data(words, classes, documents, stemmer)
    start_training(words, classes, training_data, output)

if __name__ == "__main__":
    main()  