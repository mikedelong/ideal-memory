import pickle
from collections import Counter
from logging import INFO
from logging import basicConfig
from logging import getLogger
from string import punctuation
from time import time

import pandas as pd
from numpy import array
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from spam_classifier import SpamClassifier
from sklearn.svm import SVC


def adaboost_count(x_train, y, test, random_state, ):
    vectorizer = CountVectorizer(ngram_range=(1, 3), )
    counts = vectorizer.fit_transform(x_train.values, )
    return 'ada/count', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                                           n_estimators=100, random_state=random_state,
                                           ).fit(X=counts, y=y.values, ).predict(X=vectorizer.transform(test), )


# todo: score to beat: 0.985 (learning rate = 0.55)
def adaboost_tf_idf(x_train, y, test, random_state, ):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), )
    counts = vectorizer.fit_transform(x_train.values, )
    algorithm = 'SAMME.R'  # was 'SAMME.R'
    learning_rate = 0.55  # was 1.0
    return 'ada/tf-idf', AdaBoostClassifier(algorithm=algorithm, base_estimator=None, learning_rate=learning_rate,
                                            n_estimators=100, random_state=random_state,
                                            ).fit(X=counts, y=y.values, ).predict(X=vectorizer.transform(test), )


def bayes_count(x_train, y, test, ):
    vectorizer = CountVectorizer(ngram_range=(1, 3), )
    counts = vectorizer.fit_transform(x_train.values, )
    return 'Bayes/count', MultinomialNB().fit(X=counts, y=y.values, ).predict(X=vectorizer.transform(test, ), )


def bayes_tf_idf(x_train, y, test, ):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), )
    counts = vectorizer.fit_transform(x_train.values, )
    return 'Bayes/tf-idf', MultinomialNB().fit(X=counts, y=y.values, ).predict(X=vectorizer.transform(test, ), )


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
    return ' '.join(tokens)


def collect(arg, ):
    return Counter([token for value in arg.values for token in str(value).split()])


def decision_tree_count(x_train, y, test, random_state, ):
    vectorizer = CountVectorizer(ngram_range=(1, 3), )
    counts = vectorizer.fit_transform(x_train.values, )
    return 'tree/count', DecisionTreeClassifier(random_state=random_state, ).fit(X=counts, y=y.values, ).predict(
        X=vectorizer.transform(test), )


def decision_tree_tf_idf(x_train, y, test, random_state, ):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), )
    counts = vectorizer.fit_transform(x_train.values, )
    return 'tree/tf-idf', DecisionTreeClassifier(random_state=random_state).fit(X=counts, y=y.values, ).predict(
        X=vectorizer.transform(test), )


def k_neighbors_count(x_train, y, test, neighbors, ):
    vectorizer = CountVectorizer(ngram_range=(1, 3), )
    counts = vectorizer.fit_transform(x_train.values, )
    return 'k-neighbors/count', KNeighborsClassifier(n_neighbors=neighbors, weights='uniform', algorithm='auto',
                                                     leaf_size=30, p=2, metric='minkowski', metric_params=None,
                                                     n_jobs=None, ).fit(X=counts, y=y.values,
                                                                        ).predict(X=vectorizer.transform(test, ))


def k_neighbors_tf_idf(x_train, y, test, neighbors, ):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), )
    counts = vectorizer.fit_transform(x_train.values, )
    return 'k-neighbors/tf-idf', KNeighborsClassifier(n_neighbors=neighbors, weights='uniform', algorithm='auto',
                                                      leaf_size=30, p=2, metric='minkowski', metric_params=None,
                                                      n_jobs=None, ).fit(X=counts, y=y.values,
                                                                         ).predict(X=vectorizer.transform(test, ))


def logreg_count(x_train, y, test, ):
    # https://towardsdatascience.com/spam-detection-with-logistic-regression-23e3709e522
    vectorizer = CountVectorizer(ngram_range=(1, 3), )
    counts = vectorizer.fit_transform(x_train.values, )
    return 'logreg/count', LogisticRegression(penalty='l1', solver='liblinear', ).fit(X=counts, y=y.values, ).predict(
        X=vectorizer.transform(test, ), )


def logreg_tf_idf(x_train, y, test, ):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), )
    counts = vectorizer.fit_transform(x_train.values, )
    return 'logreg/tf-idf', LogisticRegression(penalty='l1', solver='liblinear', ).fit(X=counts, y=y.values, ).predict(
        X=vectorizer.transform(test, ), )


def random_forest_count(x_train, y, test, random_state, ):
    vectorizer = CountVectorizer(ngram_range=(1, 3), )
    counts = vectorizer.fit_transform(x_train.values, )
    return 'randfor/count', RandomForestClassifier(n_estimators=100, random_state=random_state,
                                                   ).fit(X=counts, y=y.values,
                                                         ).predict(X=vectorizer.transform(test, ), )


def random_forest_tf_idf(x_train, y, test, random_state, ):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), )
    counts = vectorizer.fit_transform(x_train.values, )
    return 'randfor/tf-idf', RandomForestClassifier(n_estimators=30, random_state=random_state,
                                                    ).fit(X=counts, y=y.values,
                                                          ).predict(X=vectorizer.transform(test, ), )


def svc_count(x_train, y, test, random_state, ):
    vectorizer = CountVectorizer(ngram_range=(1, 3), )
    counts = vectorizer.fit_transform(x_train.values, )
    return 'svc/count', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr',
                            degree=3, gamma='scale', kernel='rbf', max_iter=-1, probability=False,
                            random_state=random_state, shrinking=True, tol=0.001, verbose=False,
                            ).fit(X=counts, y=y.values, ).predict(X=vectorizer.transform(test, ), )


def svc_tf_idf(x_train, y, test, random_state, ):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), )
    counts = vectorizer.fit_transform(x_train.values, )
    return 'svc/tf-idf', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr',
                             degree=3, gamma='scale', kernel='rbf', max_iter=-1, probability=False,
                             random_state=random_state, shrinking=True, tol=0.001, verbose=False,
                             ).fit(X=counts, y=y.values, ).predict(X=vectorizer.transform(test, ), )


def spam_bow(train, test, ):
    classifier = SpamClassifier(method='bow', grams=1, train_data=train, )
    classifier.train()
    result = classifier.predict(test_data=test, )
    return 'spam/bow', [result[key] for key in sorted(result.keys())]


def spam_tf_idf(train, test, ):
    classifier = SpamClassifier(method='tf-idf', grams=1, train_data=train, )
    classifier.train()
    result = classifier.predict(test_data=test, )
    return 'spam/tf-idf', [result[key] for key in sorted(result.keys())]


if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__, )
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
    train_df['clean'] = train_df['Clause Text'].apply(clean, args=(stopwords,), )
    positive_df = train_df[train_df['Classification'] == 1]

    count = dict(collect(train_df.clean, ), )
    positive_count = dict(collect(positive_df.clean, ), )

    scores = dict()
    for word in dict(count).keys():
        if word in positive_count.keys():
            word_count = count[word]
            if word_count > 5:
                scores[word] = positive_count[word] / word_count

    flat_scores = sorted([(key, value) for key, value in scores.items()], key=lambda x: x[1], reverse=True, )
    top_flat_scores = [item for item in flat_scores if 1 > item[1] > 0.9]
    for score in top_flat_scores:
        logger.info('{} {:5.4f}'.format(score[0], score[1], ), )

    # todo: add code to produce a submittable guess
    run_count = 10
    differences = list()
    random_states = list(range(1, run_count + 1))
    test_size_ = 0.1
    score = -200.0
    best_classifier = ''
    model_name = ''
    for which_classifier in range(15):
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
                model_name, y_predicted = random_forest_count(X_train, y_train, X_test, random_state_, )
            elif which_classifier == 7:
                model_name, y_predicted = random_forest_tf_idf(X_train, y_train, X_test, random_state_, )
            elif which_classifier == 8:
                model_name, y_predicted = adaboost_count(X_train, y_train, X_test, random_state_, )
            elif which_classifier == 9:
                model_name, y_predicted = adaboost_tf_idf(X_train, y_train, X_test, random_state_, )
            elif which_classifier == 10:
                model_name, y_predicted = decision_tree_count(X_train, y_train, X_test, random_state_, )
            elif which_classifier == 11:
                model_name, y_predicted = decision_tree_tf_idf(X_train, y_train, X_test, random_state_, )
            elif which_classifier == 12:
                neighbors_ = 9
                model_name, y_predicted = k_neighbors_count(X_train, y_train, X_test, neighbors_, )
            elif which_classifier == 13:
                neighbors_ = 5
                model_name, y_predicted = k_neighbors_tf_idf(X_train, y_train, X_test, neighbors_, )
            elif which_classifier == 14:
                model_name, y_predicted = svc_count(X_train, y_train, X_test, random_state_, )
            elif which_classifier == 15:
                model_name, y_predicted = svc_tf_idf(X_train, y_train, X_test, random_state_, )

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
