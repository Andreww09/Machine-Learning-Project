import os
import pandas as pd
import numpy as np
from collections import Counter

# Functions for loading and preprocessing data
def load_data(folder_name):
    data = []
    labels = []
    for filename in os.listdir(folder_name):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_name, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read().lower()
                words = content.split()
                data.append(words)
                labels.append(1 if "spmsg" in filename else 0)
    return data, labels

def create_bag_of_words(data):
    all_words = [word for text in data for word in text]
    word_freq = Counter(all_words)
    return word_freq

def create_feature_vectors(data, bag_of_words):
    feature_vectors = []
    for text in data:
        features = [1 if word in text else 0 for word in bag_of_words]
        feature_vectors.append(features)
    return feature_vectors

# Functions for ID3 algorithm
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    return np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])

def InfoGain(data, split_attribute_name, target_name="label"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
    return total_entropy - Weighted_Entropy

def ID3(data, originaldata, features, target_attribute_name="label", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature:{}}
        features = [i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, originaldata, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        return tree

def predict(query, tree, default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default

            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result

# Testing the tree's performance
def test_tree(data, tree):
    queries = data.iloc[:,:-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"]) 

    # Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i], tree, 1.0) 
    accuracy = np.sum(predicted["predicted"] == data["label"]) / len(data)
    return accuracy

def leave_one_out_cross_validation(all_parts, k=9):
    accuracies = []

    for i in range(1, k+1):
        validation_part = f'lingspam_public//stop//part{i}'
        train_data, train_labels = [], []

        # Load training data from other parts
        for j in range(1, k+1):
            if j != i:
                folder_name = f'lingspam_public//stop//part{j}'
                data, labels = load_data(folder_name)
                train_data.extend(data)
                train_labels.extend(labels)

        # Prepare training DataFrame
        train_bag_of_words = create_bag_of_words(train_data)
        top_words = [word for word, freq in train_bag_of_words.most_common(50)]
        train_feature_vectors = create_feature_vectors(train_data, top_words)
        train_df = pd.DataFrame(train_feature_vectors, columns=top_words)
        train_df['label'] = train_labels

        # Build the ID3 Tree
        features = train_df.columns[:-1]
        tree = ID3(train_df, train_df, features)

        # Prepare validation data
        validation_data, validation_labels = load_data(validation_part)
        validation_feature_vectors = create_feature_vectors(validation_data, top_words)
        validation_df = pd.DataFrame(validation_feature_vectors, columns=top_words)
        validation_df['label'] = validation_labels

        # Testing on the validation set
        accuracy = test_tree(validation_df, tree)
        accuracies.append(accuracy)
        print(f"Accuracy for part {i} as validation set: {accuracy}")

    # Calculate and return the average accuracy
    average_accuracy = sum(accuracies) / len(accuracies)
    return average_accuracy



# Load and preprocess training data
train_data, train_labels = [], []
for i in range(1, 10):
    folder_name = f'lingspam_public//stop//part{i}'
    data, labels = load_data(folder_name)
    train_data.extend(data)
    train_labels.extend(labels)

# Bag of words and feature vectors for training data
train_bag_of_words = create_bag_of_words(train_data)
top_words = [word for word, freq in train_bag_of_words.most_common(50)]
train_feature_vectors = create_feature_vectors(train_data, top_words)

# Load and preprocess test data
test_data, test_labels = load_data(f'lingspam_public//stop//part10')
test_feature_vectors = create_feature_vectors(test_data, top_words)

# Convert data into DataFrame format for ID3 algorithm
train_df = pd.DataFrame(train_feature_vectors, columns=top_words)
train_df['label'] = train_labels
features = train_df.columns[:-1]  # Exclude the label column
test_df = pd.DataFrame(test_feature_vectors, columns=top_words)
test_df['label'] = test_labels

# Building the ID3 Tree
tree = ID3(train_df, train_df, train_df.columns[:-1])

train_accuracy = test_tree(train_df, tree)
print(f"Training Accuracy: {train_accuracy}")

import pprint
pprint.pprint(tree)

# Testing on the dataset 
accuracy = test_tree(test_df, tree)
print(f"Accuracy: {accuracy}")

# Call the function and print the average accuracy
average_accuracy = leave_one_out_cross_validation(all_parts='lingspam_public//stop', k=9)
print(f"Average Accuracy: {average_accuracy}")
