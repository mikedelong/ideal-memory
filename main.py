import pickle
from collections import Counter
from logging import INFO
from logging import basicConfig
from logging import getLogger
from string import punctuation
from time import time

import pandas as pd
from numpy import array
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from spam_classifier import SpamClassifier


def adaboost_count(x_train, y, test, ):
    vectorizer = CountVectorizer(ngram_range=(1, 3), )
    transformed = vectorizer.fit_transform(x_train.values, )
    local_classifier = AdaBoostClassifier(n_estimators=100, )
    local_classifier.fit(X=transformed, y=y.values, )
    result = local_classifier.predict(X=vectorizer.transform(test), )
    return 'ada/count', result


def adaboost_tf_idf(x_train, y, test, ):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), )
    transformed = vectorizer.fit_transform(x_train.values, )
    local_classifier = AdaBoostClassifier(n_estimators=100, )
    local_classifier.fit(X=transformed, y=y.values, )
    result = local_classifier.predict(X=vectorizer.transform(test), )
    return 'ada/tf-idf', result


def bayes_count(x_train, y, test, ):
    vectorizer = CountVectorizer(ngram_range=(1, 3), )
    transformed = vectorizer.fit_transform(x_train.values, )
    result = MultinomialNB().fit(X=transformed, y=y.values, ).predict(X=vectorizer.transform(test, ), )
    return 'Bayes/count', result


def bayes_tf_idf(x_train, y, test, ):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), )
    transformed = vectorizer.fit_transform(x_train.values, )
    local_classifier = MultinomialNB()
    local_classifier.fit(X=transformed, y=y.values, )
    result = local_classifier.predict(X=vectorizer.transform(test, ), )
    return 'Bayes/tf-idf', result


def clean(arg, omit, ):
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


def collect(arg, ):
    tokens = [token for value in arg.values for token in str(value).split()]
    return Counter(tokens)


def decision_tree_count(x_train, y, test, random_state,):
    vectorizer = CountVectorizer(ngram_range=(1, 3), )
    transformed = vectorizer.fit_transform(x_train.values, )
    local_classifier = DecisionTreeClassifier(random_state=random_state)
    local_classifier.fit(X=transformed, y=y.values, )
    result = classifier.predict(X=vectorizer.transform(test), )
    return 'tree/count', result


def logreg_count(x_train, y, test, ):
    # https://towardsdatascience.com/spam-detection-with-logistic-regression-23e3709e522
    vectorizer = CountVectorizer(ngram_range=(1, 3), )
    transformed = vectorizer.fit_transform(x_train.values, )
    local_classifier = LogisticRegression(penalty='l1', solver='liblinear', )
    local_classifier.fit(X=transformed, y=y.values, )
    result = local_classifier.predict(X=vectorizer.transform(test, ), )
    return 'logreg/count', result


def logreg_tf_idf(x_train, y, test, ):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), )
    transformed = vectorizer.fit_transform(x_train.values, )
    local_classifier = LogisticRegression(penalty='l1', solver='liblinear', )
    local_classifier.fit(X=transformed, y=y.values, )
    result = local_classifier.predict(X=vectorizer.transform(test, ), )
    return 'logreg/tf-idf', result


def random_forest_count(x_train, y, test, random_state,):
    vectorizer = CountVectorizer(ngram_range=(1, 3), )
    transformed = vectorizer.fit_transform(x_train.values, )
    local_classifier = RandomForestClassifier(n_estimators=100, random_state=random_state,)
    local_classifier.fit(X=transformed, y=y.values, )
    result = local_classifier.predict(X=vectorizer.transform(test, ), )
    return 'randfor/count', result


def random_forest_tf_idf(x_train, y, test, random_state,):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), )
    transformed = vectorizer.fit_transform(x_train.values, )
    local_classifier = RandomForestClassifier(n_estimators=30, random_state=random_state,)
    local_classifier.fit(X=transformed, y=y.values, )
    result = classifier.predict(X=vectorizer.transform(test, ), )
    return 'randfor/tf-idf', result


def spam_bow(train, test, ):
    local_classifier = SpamClassifier(method='bow', grams=1, train_data=train, )
    local_classifier.train()
    result = local_classifier.predict(test_data=test, )
    result = [result[key] for key in sorted(result.keys())]
    return 'spam/bow', result


def spam_tf_idf(train, test, ):
    local_classifier = SpamClassifier(method='tf-idf', grams=1, train_data=train, )
    local_classifier.train()
    result = local_classifier.predict(test_data=test, )
    result = [result[key] for key in sorted(result.keys())]
    return 'spam/tf-idf', result


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

    # todo: add code to produce a submittable guess
    run_count = 10
    differences = list()
    random_states = list(range(1, run_count + 1))
    test_size_ = 0.1
    score = -200.0
    best_classifier = ''
    model_name = ''
    for which_classifier in range(12):
        for random_state_ in random_states:
            X_train, X_test, y_train, y_test = train_test_split(train_df['clean'], train_df['Classification'],
                                                                random_state=random_state_, test_size=test_size_, )
            train_data_ = pd.DataFrame(data={'message': X_train, 'label': y_train, }, ).reset_index()

            if which_classifier == 0:
                model_name, y_predicted = spam_bow(train_data_, X_test, )
            elif which_classifier == 1:
                model_name, y_predicted = spam_tf_idf(train_data_, X_test, )
            elif which_classifier == 2:
                model_name, y_predicted = bayes_count(X_train, y_train, X_test, )
            elif which_classifier == 3:
                model_name, y_predicted = bayes_tf_idf(X_train, y_train, X_test, )
            elif which_classifier == 4:
                model_name, y_predicted = logreg_count(X_train, y_train, X_test, )
            elif which_classifier == 5:
                model_name, y_predicted = logreg_tf_idf(X_train, y_train, X_test, )
            elif which_classifier == 6:
                model_name, y_predicted = random_forest_count(X_train, y_train, X_test, random_state_,)
            elif which_classifier == 7:
                model_name, y_predicted = random_forest_tf_idf(X_train, y_train, X_test, random_state_,)
            elif which_classifier == 8:
                model_name, y_predicted = adaboost_count(X_train, y_train, X_test, )
            elif which_classifier == 9:
                model_name, y_predicted = adaboost_tf_idf(X_train, y_train, X_test, )
            elif which_classifier == 10:
                model_name, y_predicted = decision_tree_count(X_train, y_train, X_test, random_state_,)
            elif which_classifier == 11:
                tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), )
                counts = tfidf_vectorizer.fit_transform(X_train.values, )
                classifier = DecisionTreeClassifier(random_state=random_state_)
                classifier.fit(X=counts, y=y_train.values, )
                y_predicted = classifier.predict(X=tfidf_vectorizer.transform(X_test), )
                model_name = 'tree/tf-idf'
            else:
                raise NotImplementedError('classifier {} is not implemented'.format(which_classifier))
            ones = sum(y_predicted)
            accuracy_format = 'method: {} accuracy: {:5.2f} dummy classifier accuracy: {:5.2f} difference: {:5.2f}'
            accuracy = accuracy_score(y_pred=y_predicted, y_true=y_test, )
            dummy_accuracy = accuracy_score(y_pred=[0 for _ in range(len(y_predicted))], y_true=y_test, )
            difference = 100 * (accuracy - dummy_accuracy)
            differences.append(difference)
            logger.info(accuracy_format.format(model_name, accuracy, dummy_accuracy, difference))
        mean_difference = array(differences).mean()
        logger.info('mean difference: {:5.2f}'.format(mean_difference))
        if mean_difference > score:
            score = mean_difference
            best_classifier = model_name

    logger.info('best classifier: {} score: {:5.3f}'.format(best_classifier, score, ))
    logger.info('total time: {:5.2f}s'.format(time() - time_start))
