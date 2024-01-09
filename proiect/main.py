import os
import string
from collections import Counter


class NaiveBayes:
    def __init__(self):
        self.parameters_normal = None
        self.parameters_spam = None
        self.p_spam = None
        self.p_normal = None
        self.class_probs = None
        self.feature_probs = None

    def fit(self, features, y):
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

        self.parameters_spam = Counter(bag_of_words)
        self.parameters_normal = Counter(bag_of_words)
        # Calculate parameters
        for word in bag_of_words:
            n_word_given_spam = self.parameters_spam[word]
            p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha * len(bag_of_words))
            self.parameters_spam[word] = p_word_given_spam

            n_word_given_normal = self.parameters_normal[word]  # ham_messages already defined
            p_word_given_normal = (n_word_given_normal + alpha) / (n_normal + alpha * len(bag_of_words))
            self.parameters_normal[word] = p_word_given_normal

    def predict(self, features):

        p_spam = self.p_spam
        p_normal = self.p_normal

        for word in features:
            if word in self.parameters_spam:
                p_spam *= self.parameters_spam[word]

            if word in self.parameters_normal:
                p_normal *= self.parameters_normal[word]

        if p_normal >= p_spam:
            return 0
        else:
            return 1

    def evaluate(self, features, labels):
        count = 0
        for i in range(0, len(features)):
            if self.predict(features[i]) == labels[i]:
                count += 1
        return count / len(features)

    def leave_one_out_evaluation(self, features, labels):
        num_samples = len(features)
        correct_predictions = 0
        print(num_samples)
        for i in range(0, num_samples):
            if i %10==0:
                print(i)
            # Exclude the i-th sample from training
            train_features = features[:i] + features[(i + 1):]
            train_labels = labels[:i] + labels[(i + 1):]

            # Train the model on the remaining samples
            self.fit(train_features, train_labels)

            # Evaluate on the excluded sample
            prediction = self.predict(features[i])
            correct_predictions += (prediction == labels[i])

        accuracy = correct_predictions / num_samples
        return accuracy


def load_data(folder_path):
    files = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                for punctuation in string.punctuation:
                    content = content.replace(punctuation, ' ')
                content = content.lower()
                files.append(content)
                labels.append(1 if filename.startswith("spmsg") else 0)
    return files, labels


# Load data from the first 9 folders for training
train_files = []
train_labels = []
for i in range(1, 10):
    folder_path = f'lingspam_public//bare//part{i}'
    files, labels = load_data(folder_path)
    train_files.extend(files)
    train_labels.extend(labels)

test_folder_path = 'lingspam_public//bare//part10'
test_files, test_labels = load_data(test_folder_path)
bag_of_words = []
for i in range(0, len(train_files)):
    train_files[i] = train_files[i].split()
    for word in train_files[i]:
        bag_of_words.append(word)

bag_of_words = list(set(bag_of_words))

naive_bayes_model = NaiveBayes()
naive_bayes_model.fit(train_files, train_labels)
print(naive_bayes_model.evaluate(train_files, train_labels))
# print(naive_bayes_model.leave_one_out_evaluation(train_files, train_labels))
