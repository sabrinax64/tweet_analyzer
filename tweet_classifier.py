from __future__ import division

#############################################################################
# Copyright 2011 Jason Baldridge
# 
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#############################################################################

# Imports from external packages included in the same directory
from porter_stemmer import PorterStemmer
import twokenize
import emoticons
from itertools import islice
from operator import itemgetter
from collections import defaultdict

from mlp import LogisticClassifier, compute_evaluation_loss, train 

#############################################################################
# Code to set up some resources for your features

# Create the stemmer for later use.
stemmer = PorterStemmer()

# Read in the stop words
stop_words = set([x.strip() for x in open("data/resources/stopwords.english", encoding="ISO-8859-1").readlines()])

# Obtain the words from Opinion Lexicon
neg_words =  set([x.strip() for x in open("data/resources/negative-words.txt", encoding="ISO-8859-1").readlines() if not(x.startswith(";"))])
neg_words.remove("")

pos_words =  set([x.strip() for x in open("data/resources/positive-words.txt", encoding="ISO-8859-1").readlines() if not(x.startswith(";"))])
pos_words.remove("")


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def tweets_from_files():
    dataset = {}
    for filetype in ["train", "dev", "test"]:
        dataset[filetype] = []
        fi = open("data/debate08/" + filetype + ".txt", "r")
        for line in fi:
            parts = line.strip().split("\t")
            dataset[filetype].append(parts)
    
    return dataset


#############################################################################
# General features
def extract_features(tweet, extended_features):
    features = []
    stems = []
    # FIXME: Change to use tokenization from twokenize.
    tokens = twokenize.tokenize(tweet)
    # FIXME: lower case the tokens
  
    lctokens = [token.lower() for token in tokens]

    # FIXME: Create stems here
    
    stems = [stemmer.stem_token(token) for token in lctokens]
    
    #all:Validation loss: 0.878645122051239 Validation accuracy: 0.6163522012578616
    # Add unigram features. 
    # FIXME: exclude stop words
    stopped_words = [token for token in lctokens if token not in stop_words]

    # FIXME: consider using lower case version and/or stems
    for tok in lctokens:
        features.append("word=" + tok)

    if extended_features:

    
            bigrams = ['_'.join([first, second]) for first, second in window(lctokens, 2)]
            features.extend(bigrams)       
        # FIXME: Add other features -- be creative!
            trigrams = ['_'.join([first, second, third]) for first, second, third in window(lctokens, 3)] 
            features.extend(trigrams)   
            upperct = 0
            final = ""
            for token in tokens:
                if token.istitle():
                    upperct += 1
                    final = "caps=" + str(min(upperct,3))
            features.append(final)
    return features




def feature_counter(dataset, extended_features):
    feature2id = {} # A dictionary converting each feature to a numerical ID
    feature_count = 0 # The number of features that we have

    label2id = {} # A dictionary converting each class label to a numerical ID
    label_count = 0 # The number of labels that we have
    id2label = {} # Inverse of label2id

    # Now loop over our datasets and find all features and class labels
    # that are used there. Each feature will be added to our feature2id
    # dictionary, and each label will be added to our label2id dictionary
    splits = ["train", "dev", "test"]
    for split in splits:
        for example in dataset[split]:
            
            if len(example) != 2:
                continue

            tweet = example[0]
            label = example[1]

            features = extract_features(tweet, extended_features)

            for feature in features:
                if feature not in feature2id:
                    feature2id[feature] = feature_count
                    feature_count += 1

            if label not in label2id:
                label2id[label] = label_count
                id2label[label_count] = label
                label_count += 1

    return feature2id, feature_count, label2id, label_count, id2label




# Creates a dataset from a file listing features.
# The dataset is a list of (input, output) pairs.
# The input should be a PyTorch vector whose length is the number of features 
# that we have. This tensor should include a 1 at the vector positions corresponding
# to the features that appear in this example, with zeroes everywhere else. For example,
# if this example included features 1 and 7 (but no others), the vector would look like:
#      torch.tensor([0,1,0,0,0,0,0,1,0,0,0, ... ,0,0])
# The output should be a PyTorch tensor containing a single integer corresponding
# to the correct label for this example. Thus, if the example belongs to class 0, the
# output should be torch.tensor([0]); if it belongs to class 1, the output 
# should be torch.tensor([1]).
def tensorize_dataset(datasplit, extended_features, n_features, feature2id, n_labels, label2id):
    
    tensor_datasplit = []
    for example in datasplit:

        if len(example) != 2:
            continue

        tweet = example[0]
        label = example[1]

        numerical_features = [0.0 for _ in range(n_features)]
        features = extract_features(tweet, extended_features)

        for feature in features:
            numerical_features[feature2id[feature]] = 1.0

        tensor_features = torch.tensor(numerical_features)

        tensor_labels = torch.tensor([label2id[label]], dtype=torch.long)

        tensor_datasplit.append([tensor_features, tensor_labels])

    return tensor_datasplit








#############################################################################
# Predict sentiment based on the most common sentiment label in the dataset

def majority_class_baseline(datasplit):
    label_counts = defaultdict(int)

    for tweet, label in datasplit:
        label_counts[label] += 1

    majority_class_label = max(label_counts, key=label_counts.get)

    return majority_class_label


#############################################################################
# Predict sentiment based on ratio of positive and negative terms in a tweet
def lexicon_ratio_baseline(tweet):
    tokens = twokenize.tokenize(tweet)
   

    # FIXME: Count the number of positive and negative words in the tweet
    num_positive = 0
    num_negative = 0

    for tok in tokens:
     if tok.lower() in pos_words:
        num_positive += 1 
    for tok in tokens:
        if tok.lower() in neg_words:
            num_negative += 1 


    #########################################################################
    # Don't change anything below this comment
    #########################################################################

    # Let neutral be prefered if nothing is found.
    num_neutral = .2

    # Go with neutral if pos and neg are the same
    if num_positive == num_negative:
        num_neutral += len(tokens)

    # Add a small count to each so we don't get divide-by-zero error
    num_positive += .1
    num_negative += .1

    denominator = num_positive + num_negative + num_neutral

    # Create pseudo-probabilities based on the counts
    predictions = [("positive", num_positive/denominator), 
                   ("negative", num_negative/denominator),
                   ("neutral", num_neutral/denominator)]

    # Sort
    predictions.sort(key=itemgetter(1),reverse=True)

    # Return the top label and its confidence
    return predictions[0]
 
if __name__ == "__main__":
    import argparse
    import random
    import numpy as np
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--extended_features", help="evaluate on the test set", action='store_true')
    parser.add_argument("--method", help="model type", type=str, default="logistic") 
    args = parser.parse_args()


    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)

    dataset = tweets_from_files()

    feature2id, feature_count, label2id, label_count, id2label = feature_counter(dataset, args.extended_features)

    training_set = tensorize_dataset(dataset["train"], args.extended_features, feature_count, feature2id, label_count, label2id) 
    dev_set = tensorize_dataset(dataset["dev"], args.extended_features, feature_count, feature2id, label_count, label2id)
    test_set = tensorize_dataset(dataset["test"], args.extended_features, feature_count, feature2id, label_count, label2id)


    if args.method == "logistic":
        input_feature_count = len(training_set[0][0])
        model = LogisticClassifier(input_feature_count, label_count)

        train(model, training_set, dev_set, dev_set, id2label, lr=0.001, n_epochs=10, batch_size=10, eval_every=100, model_name="twitter")
    elif args.method == "majority":
        majority_class = majority_class_baseline(dataset["train"])
        count_correct = 0
        total = 0

        for example in dataset["dev"]:
            if example[1] == majority_class:
                count_correct += 1
            total += 1
        print("Majority class performance:", count_correct*1.0/total)

    elif args.method == "lexicon_ratio":
        count_correct = 0
        total = 0

        for example in dataset["dev"]:
            tweet = example[0]
            label = example[1]
            prediction = lexicon_ratio_baseline(tweet)
            if prediction[0] == label:
                count_correct += 1
            total += 1

        print("Lexicon ratio performance:", count_correct*1.0/total)
