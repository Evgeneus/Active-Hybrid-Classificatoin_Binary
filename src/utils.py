import numpy as np
import pandas as pd
import random

from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer():
    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=False, max_features=2000, ngram_range=(1, 2))

    def transform(self, X):
        return self.vectorizer.transform(X).toarray()

    def fit(self, X):
        self.vectorizer.fit(X)

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X).toarray()


class CrowdSimulator:

    @staticmethod
    def crowdsource_items(item_ids, gt_items, crowd_acc, max_votes_per_item, crowd_votes_counts):
        '''
        :param gt_items: list of ground truth values fo items to crowdsource
        :param crowd_acc: crowd accuracy range
        :param n: n is max_votes_per_item
        :return: aggregated crwodsourced label on items, cost_round (total number of crowdsourced votes)
        '''
        crodsourced_items = []
        cost_round = 0
        for item_id, gt in zip(item_ids, gt_items):
            in_votes, out_votes = 0, 0
            for _ in range(max_votes_per_item):
                cost_round += 1
                worker_acc = random.uniform(crowd_acc[0], crowd_acc[1])
                worker_vote = np.random.binomial(1, worker_acc if gt == 1 else 1 - worker_acc)
                if worker_vote == 1:
                    in_votes += 1
                else:
                    out_votes += 1
            item_label = 1 if in_votes >= out_votes else 0
            crowd_votes_counts[item_id]['in'] += in_votes
            crowd_votes_counts[item_id]['out'] += out_votes
            crodsourced_items.append(item_label)
        return crodsourced_items, cost_round


def load_data(file_name):
    path_dict = {
        '100000_reviews_lemmatized.csv': '../data/amazon-sentiment-dataset/',
        '5000_reviews_lemmatized.csv': '../data/amazon-sentiment-dataset/',
        'ohsumed_C23_1grams.csv': '../data/ohsumed_data/',
        'loneliness-dataset-2018.csv': '../data/loneliness-dataset-2018/'
    }
    path = path_dict[file_name]
    try:
        data = pd.read_csv(path + file_name, delimiter=',')
    except:
        data = pd.read_csv(path + file_name, delimiter=';')
    X = data['tokens'].values
    y = data['Y'].values
    return X, y


def get_init_training_data_idx(y, size_init_train_data):
   # initial training data
   pos_idx_all = (y == 1).nonzero()[0]
   neg_idx_all = (y == 0).nonzero()[0]
   # randomly select initial balanced training dataset
   train_idx = np.concatenate([np.random.choice(pos_idx_all, size_init_train_data // 2, replace=False),
                               np.random.choice(neg_idx_all, size_init_train_data // 2, replace=False)])
   return train_idx


# random sampling strategy for modAL
def random_sampling(_, X, n_instances=1):
    query_idx = random.sample(range(X.shape[0]), n_instances)

    return query_idx, X[query_idx]
