import pickle
from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

import pandas as pd


def clean(arg):
    tokens = arg.split()
    tokens = [item.lower() for item in tokens]
    result = ' '.join(tokens)
    return result


if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', level=INFO, )
    logger.info('started.', )

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
    positive_df = train_df[train_df['Classification'] == 1]
    train_df['clean'] = train_df['Clause Text'].apply(clean)

    # todo replace the tabs and newlines in the data with spaces before proceeding
    logger.info('total time: {:5.2f}s'.format(time() - time_start))
