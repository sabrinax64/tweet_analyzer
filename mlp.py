
import torch
import torch.nn as nn

import argparse

import random
import numpy as np

#####################################################################################
# READ IN PRETRAINED GLOVE WORD EMBEDDINGS
#####################################################################################

fi = open("glove.6B.50d.small.txt", "r")
word2vector = {}

for line in fi:
  line_list = line.strip().split()
  word = line_list[0]

  vector = line_list[1:]
  vector = [float(x) for x in vector]

  tensor = torch.tensor(vector)

  word2vector[word] = tensor



#####################################################################################
# PREPROCESS DATA SO THAT PYTORCH CAN HANDLE IT
#####################################################################################


# Takes in a pair of files, then finds all features and class
# labels used in those files. Creates a dictionary that has features
# as keys with numerical IDs as values. That is, the first feature that
# shows up in the files will get an ID of 0, the second will get an ID of 1,
# etc. This will be important for converting our features into tensors that PyTorch can use.
# Similarly, also creates a dictionary that has labels as keys with 
# numerical IDs as values (i.e., the first label that shows up gets
# an ID of 0, the second gets an ID of 1, etc. - since we are dealing with
# a binary classification task, there should only be 2 unique labels).
def feature_counter(train_filename, validation_filename, test_filename):
    feature2id = {} # A dictionary converting each feature to a numerical ID
    feature_count = 0 # The number of features that we have

    label2id = {} # A dictionary converting each class label to a numerical ID
    label_count = 0 # The number of labels that we have
    id2label = {} # The inverse of label2id

    # Now loop over our files and find all features and class labels
    # that are used there. Each feature will be added to our feature2id
    # dictionary, and each label will be added to our label2id dictionary
    filenames = [train_filename, validation_filename, test_filename]
    for filename in filenames:
        fi = open(filename, "r")
        for line in fi:
            features = line.strip().split(",")[:-1]

            for feature in features:
                if feature not in feature2id:
                    feature2id[feature] = feature_count
                    feature_count += 1


            label = line.strip().split(",")[-1]
            if label not in label2id:
                label2id[label] = label_count
                id2label[label_count] = label
                label_count += 1

        fi.close()

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
def create_dataset_from_feature_file(filename, n_features, feature2id, n_labels, label2id):
    dataset = []
    fi = open(filename, "r")
    for line in fi:
        
        parts = line.strip().split(",")
        
        features = [0.0 for _ in range(n_features)]

        for part in parts[:-1]:
            features[feature2id[part]] = 1.0

        tensor_features = torch.tensor(features)
        
        label = label2id[parts[-1]]
        tensor_labels = torch.tensor([label], dtype=torch.long)

        dataset.append([tensor_features, tensor_labels])

    return dataset


# Creates a dataset from a file. For each example, the input
# features should be the GloVe vectors for the four words,
# concatenated together.
#
# For instance, the first example in out/ppa.basic.train has
# the four words "join", "board", "as", "director". Suppose that
# "join" has the embedding [0.1, -0.2, 1.3], "board" has the embedding
# [-1.3, -7.6, 0.5], "as" has the embedding [2.2, 1.5, 6.0], and "director"
# has the embedding [-1.0, 1.0, 2.5]. Then, the input feature vector for this
# example should be [0.1, -0.2, 1.3, -1.3, -7.6, 0.5, 2.2, 1.5, 6.0, -1.0, 1.0, 2.5]
# (This example is just for illustration; actual GloVe vectors are much larger 
# than this).
#
# For now, this function is copied from create_dataset_from_feature_file. Modify it
# so that it implements the GloVe-based features described here. You should be able
# to do this by modifying only the lines of code between the two comments below.
#
# It may be helpful to take note of the word2vector dictionary created near the
# top of this file.
def create_dataset_glove(filename, n_labels, label2id):
    dataset = []
    fi = open(filename, "r")
    for line in fi:
        parts = line.strip().split(",")

 
        # START OF THE CODE YOU SHOULD CHANGE
        features = [0.0 for _ in range(n_features)]

        # This only goes up to -1 because we should not include the label
        # as an input to the model
        for part in parts[:-1]:
            features[feature2id[part]] = 1.0

        tensor_features = torch.tensor(features)
        # END OF THE CODE YOU SHOULD CHANGE        


        label = label2id[parts[-1]]
        tensor_labels = torch.tensor([label], dtype=torch.long)

        dataset.append([tensor_features, tensor_labels])

    return dataset


#####################################################################################
# DEFINE NEURAL NETWORK ARCHITECTURES
#####################################################################################

class LogisticClassifier(nn.Module):

    def __init__(self, n_input, n_output):
        super(LogisticClassifier, self).__init__()

        self.layer1 = nn.Linear(n_input, n_output)

    def forward(self, inp):

        # Note: To get our predictions, we should apply a softmax
        # to the vector "outp". However, it turns out that, in 
        # PyTorch, the loss function we will be using (which is
        # CrossEntropyLoss) incorporates a softmax. Thus, we do not
        # need to include a softmax inside the model architecture.
        outp = self.layer1(inp)

        return outp


# Currently, the definition of this class has been copied from the
# definition of LogisticClassifier above. Modify it so that it instantiates
# a multi-layer perceptron (MLP) with two layers.
#
# Specifically, to go from inp to outp, you should first apply a linear 
# layer to the input, whose output should be a vector of size hidden_size.
# Then, you should apply a ReLU to that vector. 
# Then, you should apply dropout with a proportion of dropout_p
# Finally, you should apply another linear layer that maps the output of the
# dropout layer to a vector of size n_output.
# You should not include a Softmax or Sigmoid at the end of "forward", for the
# reasons noted in the comment within "forward".
#
# It may be helpful to examine the SentimentClassifier class from the PyTorch
# tutorial posted on Canvas. For implementing dropout, it may be helpful to
# check out the PyTorch documentation for Dropout at the following link:
# https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
class MLP2(nn.Module):

    def __init__(self, n_input, n_output, hidden_size=100, dropout_p=0.0):
        super(MLP2, self).__init__()

        self.layer1 = nn.Linear(n_input, n_output)

    def forward(self, inp):

        # Note: To get our predictions, we should apply a softmax
        # to the vector "outp". However, it turns out that, in 
        # PyTorch, the loss function we will be using (which is
        # CrossEntropyLoss) incorporates a softmax. Thus, we do not
        # need to include a softmax inside the model architecture.
        outp = self.layer1(inp)

        return outp

# Modify this class so that it instantiates a multilayer perceptron (MLP) with
# three layers. This should be similar to MLP2 above, but with one extra layer.
#
# Specifically, the forward pass should include these components, in order:
# - First apply a linear layer to the input that maps it to a vector of
#   size hidden_size.
# - Then apply ReLU to that vector
# - Then apply dropout, with a proportion of dropout_p
# - Then apply another linear layer, mapping this vector (which is of size
#   hidden_size) to another vector of size hidden_size.
# - Then apply a ReLU to that vector
# - Then apply dropout, with a proportion of dropout_p
# - And finally apply another linear layer that maps this vector (which has
#   a size of hidden_size) to the final output, which should be a vector of
#   size n_output
class MLP3(nn.Module):

    def __init__(self, n_input, n_output, hidden_size=100, dropout_p=0.0):
        super(MLP3, self).__init__()

        self.layer1 = nn.Linear(n_input, n_output)

    def forward(self, inp):

        # Note: To get our predictions, we should apply a softmax
        # to the vector "outp". However, it turns out that, in 
        # PyTorch, the loss function we will be using (which is
        # CrossEntropyLoss) incorporates a softmax. Thus, we do not
        # need to include a softmax inside the model architecture.
        outp = self.layer1(inp)

        return outp


#####################################################################################
# FUNCTION FOR EVALUATING ON A DATASET
# Can be applied to the validation set or the test set
#####################################################################################

def compute_evaluation_loss(model, eval_set, loss_function):
    total_loss = 0
    total_correct = 0

    count_examples = 0

    precision_dict = {}
    recall_dict = {}

    for example in eval_set:
        pred = model(example[0]).unsqueeze(0)
        correct = example[1]

        loss = loss_function(pred, correct)
        total_loss += loss
        count_examples += 1

        _, pred_label = torch.topk(pred, 1)
        pred_label = pred_label.item()
        correct_label = correct.item()

        if pred_label not in precision_dict:
            precision_dict[pred_label] = [0,0]
        if correct_label not in recall_dict:
            recall_dict[correct_label] = [0,0]


        if pred_label == correct_label:
            total_correct += 1
            precision_dict[pred_label][0] += 1
            recall_dict[correct_label][0] += 1
        precision_dict[pred_label][1] += 1
        recall_dict[correct_label][1] += 1

    evaluation_loss = total_loss / count_examples
    evaluation_accuracy = total_correct / count_examples

    return evaluation_loss.item(), evaluation_accuracy, precision_dict, recall_dict

#####################################################################################
# FUNCTION FOR TRAINING
#####################################################################################

def train(model, training_set, validation_set, test_set, id2label, eval_on_test=False, lr=0.001, n_epochs=1, batch_size=1, eval_every=1000, model_name="mlp"):

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    training_examples_seen = 0
    best_loss = 1000000

    for epoch in range(n_epochs):

        for example_input, example_output in training_set:

            prediction = model(example_input)
            label = example_output

            # The "unsqueeze" is necessary to get the right
            # sizes for the tensors here
            loss = loss_function(prediction.unsqueeze(0), label)
            loss.backward()

            if training_examples_seen % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            if training_examples_seen % eval_every == 0:
                validation_loss, validation_accuracy, _, _ = compute_evaluation_loss(model, validation_set, loss_function)
                print(epoch, training_examples_seen, "Loss:", validation_loss)

                if validation_loss < best_loss:
                    best_loss = validation_loss
                    torch.save(model.state_dict(), model_name + ".weights")

            training_examples_seen += 1


    # Once we are done training, we evaluate
    model.load_state_dict(torch.load(model_name + ".weights"))
    validation_loss, validation_accuracy, _, _ = compute_evaluation_loss(model, validation_set, loss_function)
    print("\nFINAL VALIDATION METRICS:")
    print("Validation loss:", validation_loss, "Validation accuracy:", validation_accuracy)

    if eval_on_test:
        test_loss, test_accuracy, precision_dict, recall_dict = compute_evaluation_loss(model, test_set, loss_function)
        print("\nTEST METRICS:")
        print("Test loss:", test_loss, "Test accuracy:", test_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", help="training file", type=str, default="out/ppa.basic.train")
    parser.add_argument("--validation", "-v", help="validation file", type=str, default="out/ppa.basic.dev") 
    parser.add_argument("--test", "-e", help="test file", type=str, default="out/ppa.basic.test") 
    parser.add_argument("--eval_on_test", help="evaluate on the test set", action='store_true')
    parser.add_argument("--lr", help="learning rate", type=float, default=0.001) 
    parser.add_argument("--dropout", help="dropout percentage", type=float, default=0.0)
    parser.add_argument("--hidden_size", help="hidden size for the MLP", type=int, default=100)
    parser.add_argument("--n_epochs", help="number of times we loop over the training set", type=int, default=10) 
    parser.add_argument("--batch_size", help="batch size", type=int, default=1) 
    parser.add_argument("--eval_every", help="this is the number n where we evaluate on the validation set after every n examples", type=int, default=5000)  
    parser.add_argument("--glove", help="Use concatenated GloVe embeddings as the input features", action='store_true')
    parser.add_argument("--model", help="type of model (mlp2, mlp3, or logistic)", type=str, default="logistic")
    parser.add_argument("--random_seed", help="random seed", type=int, default=42)
    args = parser.parse_args()


    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)


    feature2id, feature_count, label2id, label_count, id2label = feature_counter(args.train, args.validation, args.test)

    if args.glove:
        training_set = create_dataset_glove(args.train, label_count, label2id)        
        validation_set = create_dataset_glove(args.validation, label_count, label2id)
        test_set = create_dataset_glove(args.test, label_count, label2id)

    else:
        training_set = create_dataset_from_feature_file(args.train, feature_count, feature2id, label_count, label2id)
        validation_set = create_dataset_from_feature_file(args.validation, feature_count, feature2id, label_count, label2id)
        test_set = create_dataset_from_feature_file(args.test, feature_count, feature2id, label_count, label2id)

    input_feature_count = len(training_set[0][0])
    if args.model == "logistic":
        model = LogisticClassifier(input_feature_count, label_count)
    elif args.model == "mlp2":
        model = MLP2(input_feature_count, label_count, hidden_size=args.hidden_size, dropout_p=args.dropout)
    elif args.model == "mlp3":
        model = MLP3(input_feature_count, label_count, hidden_size=args.hidden_size, dropout_p=args.dropout)
    
    train(model, training_set, validation_set, test_set, id2label, eval_on_test=args.eval_on_test, lr=args.lr, n_epochs=args.n_epochs, batch_size=args.batch_size, eval_every=args.eval_every, model_name="mlp")




