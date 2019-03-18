import numpy as np
from modAL.models import ActiveLearner


class Learner:

    def __init__(self, params):
        self.clf = params['clf']
        self.sampling_strategy = params['sampling_strategy']
        self.n_instances_query = params['n_instances_query']
        self.out_threshold = params['out_threshold']

    def setup_active_learner(self, X_train_init, y_train_init, X_pool, y_pool):
        # create the pool
        self.X_pool = X_pool
        self.y_pool = y_pool

        # initialize active learner
        self.learner = ActiveLearner(
            estimator=self.clf,
            X_training=X_train_init, y_training=y_train_init,
            query_strategy=self.sampling_strategy
        )

    def query(self):
        l = self.learner
        if self.n_instances_query > len(self.y_pool):
            n_instances = len(self.y_pool)
        else:
            n_instances = self.n_instances_query
        query_idx, _ = l.query(self.X_pool, n_instances=n_instances)
        return query_idx

    def teach(self, query_idx, y_crowdsourced):
        l = self.learner
        l.teach(self.X_pool[query_idx], y_crowdsourced)
        # remove queried instance from the pool
        self.X_pool = np.delete(self.X_pool, query_idx, axis=0)
        self.y_pool = np.delete(self.y_pool, query_idx)
