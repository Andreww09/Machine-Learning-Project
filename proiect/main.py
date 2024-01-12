import os
import string

import numpy as np


class NaiveBayes:
    def __init__(self):
        self.parameters_normal = None
        self.parameters_spam = None
        self.p_spam = None
        self.p_normal = None
        self.class_probs = None
        self.feature_probs = None

    def fit(self, x, y):

        # features will contain a list of the words from every email
        features = []
        # all the words in all the emails
        bag_of_words = []
        for i in range(0, len(x)):
            features.append(x[i].split())
            for word in features[i]:
                bag_of_words.append(word)
        bag_of_words = list(set(bag_of_words))

        # sort the emails to their categories
        normal = [features[i] for i in range(0, len(features)) if y[i] == 0]
        spam = [features[i] for i in range(0, len(features)) if y[i] == 1]
        # P(Spam) and P(Normal)
        self.p_normal = len(normal) / len(features)
        self.p_spam = len(spam) / len(features)

        # N_Spam
        n_words_per_spam = list(map(len, spam))
        n_spam = sum(n_words_per_spam)

        # N_normal
        n_words_per_normal = list(map(len, normal))
        n_normal = sum(n_words_per_normal)
        # Laplace smoothing
        alpha = 1

        # number of times a word appears in normal emails
        normal_count = {unique_word: 0 for unique_word in bag_of_words}
        for line in normal:
            for word in line:
                normal_count[word] += 1

        # number of times a word appear in spam emails
        spam_count = {unique_word: 0 for unique_word in bag_of_words}
        for line in spam:
            for word in line:
                spam_count[word] += 1

        self.parameters_spam = {unique_word: 0 for unique_word in bag_of_words}
        self.parameters_normal = {unique_word: 0 for unique_word in bag_of_words}
        # Calculate parameters
        for word in bag_of_words:
            n_word_given_spam = spam_count[word]
            p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha * len(bag_of_words))
            self.parameters_spam[word] = p_word_given_spam
            n_word_given_normal = normal_count[word]
            p_word_given_normal = (n_word_given_normal + alpha) / (n_normal + alpha * len(bag_of_words))
            self.parameters_normal[word] = p_word_given_normal

    def predict(self, features):

        p_spam = np.log(self.p_spam)
        p_normal = np.log(self.p_normal)
        for word in features:
            if word in self.parameters_spam:
                p_spam += np.log(self.parameters_spam[word])

            if word in self.parameters_normal:
                p_normal += np.log(self.parameters_normal[word])

        if p_normal >= p_spam:
            return 0
        else:
            return 1

    def evaluate(self, x, labels):
        features = []
        for i in range(0, len(x)):
            features.append(x[i].split())
        count = 0
        for i in range(0, len(features)):
            if self.predict(features[i]) == labels[i]:
                count += 1
        return count / len(features)

    def leave_one_out_evaluation(self):

        accuracy = 0
        for i in range(0, 9):
            # Exclude the i-th sample from training
            features, labels = get_features_and_labels(i + 1)

            # Train the model on the remaining samples
            self.fit(features, labels)

            # Evaluate on the excluded sample
            accuracy += self.evaluate(train_features[i], train_labels[i])

        accuracy /= 9
        return accuracy


def load_data(folder_path):
    files = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # replaces punctuation and makes everything lowercase
                for punctuation in string.punctuation:
                    content = content.replace(punctuation, ' ')
                content = content.lower()
                files.append(content)
                labels.append(1 if filename.startswith("spmsg") else 0)
    return files, labels


def get_features_and_labels(exclude=-1):
    '''
    Combines the emails and labels from all folders into one array, potentially excluding 1 folder
    :param exclude: the folder to be excluded
    '''
    features = []
    labels = []

    for i in range(0, 9):
        if i + 1 != exclude:
            features.extend(train_features[i])
            labels.extend(train_labels[i])

    return features, labels


for folder in ['bare', 'lemm', 'lemm_stop', 'stop']:
    # Load data from the first 9 folders for training
    train_features = []
    train_labels = []
    for i in range(1, 10):
        folder_path = f'lingspam_public//{folder}//part{i}'
        files, labels = load_data(folder_path)
        train_features.append(files)
        train_labels.append(labels)
    features, labels = get_features_and_labels()

    test_folder_path = f'lingspam_public//{folder}//part10'
    test_features, test_labels = load_data(test_folder_path)

    naive_bayes_model = NaiveBayes()
    naive_bayes_model.fit(features, labels)
    print(f'{folder}:')
    print(f'Train Accuracy: {naive_bayes_model.evaluate(features, labels)}')
    print(f'Test Accuracy: {naive_bayes_model.evaluate(test_features, test_labels)}')
    print(f'LOOCV: {naive_bayes_model.leave_one_out_evaluation()}')
