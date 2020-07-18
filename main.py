import pickle
from collections import Counter
from logging import INFO
from logging import basicConfig
from logging import getLogger
from string import punctuation
from time import time

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from spam_classifier import SpamClassifier


def clean(arg, omit):
    tokens = arg.split()
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in {'\t', '\n'} and token not in omit]
    tokens = [token for token in tokens if not str(token).isdigit() and not str(token).isdecimal()]
    tokens = [token.replace('_', '', ).replace('*', '', ).replace('"', '', ) for token in tokens]
    tokens = [token.replace('(', '', ).replace(')', '', ).replace('[', '', ).replace(']', '', ) for token in tokens]
    tokens = [token[:-1] if any([str(token).endswith(symbol) for symbol in punctuation])
              else token for token in tokens]
    tokens = [token[:-1] if any([str(token).endswith(symbol) for symbol in punctuation])
              else token for token in tokens]
    result = ' '.join(tokens)
    return result


def collect(arg):
    tokens = [token for value in arg.values for token in str(value).split()]
    return Counter(tokens)


if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', level=INFO, )
    logger.info('started.', )

    with open(file='./stopwords.txt', mode='r') as stopwords_fp:
        stopwords = stopwords_fp.readlines()
    logger.info('stop word count: {}'.format(len(stopwords, ), ), )

    train_pkl = './reference/train.pkl'
    logger.info('loading pickled data from {}'.format(train_pkl, ), )
    with open(file=train_pkl, mode='rb', ) as pickle_fp:
        examples = pickle.load(pickle_fp, )

    input_file = './data/AI_ML_Challenge_Training_Data_Set_1_v1.csv'
    logger.info('loading training data from {}'.format(input_file, ), )
    all_columns = ['Clause ID', 'Clause Text', 'Classification', ]
    train_df = pd.read_csv(filepath_or_buffer=input_file, usecols=all_columns, )
    logger.info('headers: {}'.format(list(train_df, ), ), )
    logger.info(train_df['Classification'].value_counts().to_dict(), )
    logger.info('cleaning training data', )
    train_df['clean'] = train_df['Clause Text'].apply(clean, args=(stopwords,))
    positive_df = train_df[train_df['Classification'] == 1]
    negative_df = train_df[train_df['Classification'] == 0]

    count = dict(collect(train_df.clean))
    positive_count = dict(collect(positive_df.clean))

    scores = dict()
    for word in dict(count).keys():
        if word in positive_count.keys():
            word_count = count[word]
            if word_count > 5:
                scores[word] = positive_count[word] / word_count

    flat_scores = sorted([(key, value) for key, value in scores.items()], key=lambda x: x[1], reverse=True, )
    top_flat_scores = [item for item in flat_scores if 1 > item[1] > 0.9]
    for score in top_flat_scores:
        logger.info('{} {:5.4f}'.format(score[0], score[1]))

    which_classifier = 1
    if which_classifier == 0:
        logger.info('building spam classifier')
        methods = ['bow', 'tf-idf']
        run_count = 40
        test_size_ = 0.1
        random_states = list(range(1, run_count + 1))
        for method in methods:
            for random_state_ in random_states:
                X_train, X_test, y_train, y_test = train_test_split(train_df['clean'], train_df['Classification'],
                                                                    random_state=random_state_, test_size=test_size_, )
                train_data_ = pd.DataFrame(data={'message': X_train, 'label': y_train, }, ).reset_index()
                classifier = SpamClassifier(method=method, grams=1, train_data=train_data_, )
                classifier.train()
                y_predicted = classifier.predict(test_data=X_test, )
                y_predicted = [y_predicted[key] for key in sorted(y_predicted.keys())]
                accuracy_format = 'method: {} accuracy: {:5.2f} dummy classifier accuracy: {:5.2f} difference: {:5.2f}'

                accuracy = accuracy_score(y_pred=y_predicted, y_true=y_test, )
                dummy_accuracy = accuracy_score(y_pred=[0 for _ in range(len(y_predicted))], y_true=y_test, )
                difference = 100 * (accuracy - dummy_accuracy)
                logger.info(accuracy_format.format(method, accuracy, dummy_accuracy, difference))
        else:
            run_count = 40
            test_size_ = 0.1
            random_states = list(range(1, run_count + 1))
            for random_state_ in random_states:
                X_train, X_test, y_train, y_test = train_test_split(train_df['clean'], train_df['Classification'],
                                                                    random_state=random_state_, test_size=test_size_, )
                vectorizer = CountVectorizer()
                counts = vectorizer.fit_transform(X_train.values, )
                classifier = MultinomialNB()
                classifier.fit(X=counts, y=y_train.values, )
                y_predicted = classifier.predict(X=X_test, )
                accuracy_format = 'naive Bayes: accuracy: {:5.2f} dummy classifier accuracy: {:5.2f} difference: {:5.2f}'

                accuracy = accuracy_score(y_pred=y_predicted, y_true=y_test, )
                dummy_accuracy = accuracy_score(y_pred=[0 for _ in range(len(y_predicted))], y_true=y_test, )
                difference = 100 * (accuracy - dummy_accuracy)
                logger.info(accuracy_format.format(accuracy, dummy_accuracy, difference, ))

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
