import pickle
from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

import pandas as pd
from collections import Counter
from string import punctuation


def clean(arg, omit):
    tokens = arg.split()
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in {'\t', '\n'} and token not in omit]
    tokens = [token for token in tokens if not str(token).isdigit() and not str(token).isdecimal()]
    tokens = [token.replace('_', '', ).replace('*', '', ).replace('"', '', ) for token in tokens]
    tokens = [token.replace('(', '', ).replace(')', '', ).replace('[', '',).replace(']', '', ) for token in tokens]
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

    with open('./stopwords.txt', 'r') as stopwords_fp:
        stopwords = stopwords_fp.readlines()
    logger.info('stop word count: {}'.format(len(stopwords)))

    train_pkl = './reference/train.pkl'
    logger.info('loading pickled data from {}'.format(train_pkl))
    with open(file=train_pkl, mode='rb') as pickle_fp:
        examples = pickle.load(pickle_fp)

    input_file = './data/AI_ML_Challenge_Training_Data_Set_1_v1.csv'
    logger.info('loading training data from {}'.format(input_file))
    all_columns = ['Clause ID', 'Clause Text', 'Classification', ]
    train_df = pd.read_csv(filepath_or_buffer=input_file, usecols=all_columns, )
    logger.info('headers: {}'.format(list(train_df, ), ))
    logger.info(train_df['Classification'].value_counts(), )
    train_df['clean'] = train_df['Clause Text'].apply(clean, args=(stopwords,))
    positive_df = train_df[train_df['Classification'] == 1]
    negative_df = train_df[train_df['Classification'] == 0]

    count = dict(collect(train_df.clean))
    positive_count = dict(collect(positive_df.clean))
    negative_count = dict(collect(negative_df.clean))

    scores = dict()
    for word in dict(count).keys():
        if word in positive_count.keys():
            word_count = count[word]
            if word_count > 5:
                scores[word] = positive_count[word] / word_count

    flat_scores = [(key, value) for key, value in scores.items()]
    flat_scores = sorted(flat_scores, key=lambda x: x[1], reverse=True)
    top_flat_scores = [item for item in flat_scores if 1 > item[1] > 0.9]
    logger.info(top_flat_scores)

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
