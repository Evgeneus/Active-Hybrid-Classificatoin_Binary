import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support

from src.utils import get_init_training_data_idx, \
    load_data, Vectorizer, CrowdSimulator
from src.active_learning import Learner
from src.policy import PointSwitchPolicy


def run_experiment(params):
    # parameters for crowd simulation
    crowd_acc = params['crowd_acc']
    max_votes_per_item = params['max_votes_per_item']

    # df_to_print = pd.DataFrame()
    for budget_per_item in params['budget_per_item']:
        for switch_point in params['policy_switch_point']:
            print('Policy switch point: {}'.format(switch_point))
            print('Budget per item: {}'.format(budget_per_item))
            print('************************************')
            results_list = []
            for experiment_id in range(params['experiment_nums']):

                ## load and transform input data
                X, y = load_data(params['dataset_file_name'])
                vectorizer = Vectorizer()
                X = vectorizer.fit_transform(X)

                ## initialize policy
                items_num = y.shape[0]
                B = items_num * budget_per_item
                policy = PointSwitchPolicy(B, switch_point)

                ## initialize crowd votes counter, prior probs
                crowd_votes_counts, prior_prob = {}, {}
                for item_id in range(items_num):
                    crowd_votes_counts[item_id] = {'in': 0, 'out': 0}
                    prior_prob[item_id] = {'in': 0.5, 'out': 0.5}

                ## assign default positive label for all items to classify
                item_labels = {item_id: 1 for item_id in range(items_num)}  # classify all items as in by default
                unclassified_item_ids = np.arange(items_num)  # ids of unclassified so far items
                item_ids_helper = np.arange(items_num)  # to track item ids in AL pool and map them to real item ids


                ## ** START ACTIVE LEARNING ** ##
                # Run Active Learning Box if Budget is available for Active Learning part
                if switch_point != 0:
                    L, item_ids_helper = configure_al_box(params, crowd_votes_counts, item_labels, item_ids_helper, X, y)
                    policy.update_budget_al(params['size_init_train_data'] * max_votes_per_item / 2)

                    i = 0
                    while policy.is_continue_al:
                        ## query items to annotate
                        query_idx = L.query()
                        ## crowdsource queried items
                        gt_items_queried = L.y_pool[query_idx]
                        ## TODO: use max_votes_per_item to compute cost_round (like SM-run)
                        y_crowdsourced, cost_round = CrowdSimulator.crowdsource_items(item_ids_helper[query_idx],
                                                                          gt_items_queried, crowd_acc,
                                                                          max_votes_per_item, crowd_votes_counts)
                        ## Retrain AL with new data
                        L.teach(query_idx, y_crowdsourced)
                        item_ids_helper = np.delete(item_ids_helper, query_idx)

                        ## update budget spent
                        policy.update_budget_al(cost_round)
                        i += 1

                        ## measure performance
                        pre_, rec_, f_, _ = precision_recall_fscore_support(y, L.learner.predict(X),
                                                                            beta=params['beta'], average='binary')
                        print(pre_, rec_, f_)

                    ## at this stage we do not allow to classify items by machines
                    ## we use machins prob output as prior
                    for item_id in range(items_num):
                        prediction = L.learner.predict_proba([X[item_id]]).flatten()
                        prior_prob[item_id] = {'in': prediction[1], 'out': prediction[0]}
                    print('experiment_id {}, AL-Box finished'.format(experiment_id), end=', ')
                ## ** STOP ACTIVE LEARNING ** ##

                # ## Run Crowd-Box if Available Budget
                # if policy.B_crowd:
                #     print('crowd')
                #     policy.B_crowd = policy.B - policy.B_al_spent
                #     smr_params = {
                #         'estimated_predicate_accuracy': sum(crowd_acc) / len(crowd_acc),
                #         'estimated_predicate_selectivity': sum(y) / len(y),
                #         'predicates': predicates,
                #         'item_predicate_gt': item_predicate_gt,
                #         'clf_threshold': params['screening_out_threshold'],
                #         'stop_score': params['stop_score'],
                #         'crowd_acc': crowd_acc,
                #         'prior_prob': prior_prob
                #     }
                #     SMR = ShortestMultiRun(smr_params)
                #     unclassified_item_ids = np.arange(items_num)
                #     # crowdsource items for SM-Run base-round in case poor SM-Run used
                #     if switch_point == 0:
                #         baseround_item_num = 50  # since 50 used in WWW2018 Krivosheev et.al
                #         items_baseround = unclassified_item_ids[:baseround_item_num]
                #         for pr in predicates:
                #             gt_items_baseround = {item_id: item_predicate_gt[pr][item_id] for item_id in
                #                                   items_baseround}
                #             CrowdSimulator.crowdsource_items(items_baseround, gt_items_baseround, pr, crowd_acc[pr],
                #                                              crowd_votes_per_item_al, crowd_votes_counts)
                #             policy.update_budget_crowd(baseround_item_num * crowd_votes_per_item_al)
                #     unclassified_item_ids = SMR.classify_items(unclassified_item_ids, crowd_votes_counts, item_labels)
                #
                #     while policy.is_continue_crowd and unclassified_item_ids.any():
                #         # Check money
                #         if (policy.B_crowd - policy.B_crowd_spent) < len(unclassified_item_ids):
                #             unclassified_item_ids = unclassified_item_ids[:(policy.B_crowd - policy.B_crowd_spent)]
                #         unclassified_item_ids, budget_round = SMR.do_round(crowd_votes_counts, unclassified_item_ids,
                #                                                            item_labels)
                #         policy.update_budget_crowd(budget_round)
                #     print('Crowd-Box finished')
    #
    #             # if budget is over and we did the AL part then classify the rest of the items via machines
    #             if unclassified_item_ids.any() and switch_point != 0:
    #                 predicted = SAL.predict(vectorizer.transform(X[unclassified_item_ids]))
    #                 item_labels.update(dict(zip(unclassified_item_ids, predicted)))
    #
    #             # compute metrics and pint results to csv
    #             metrics = MetricsMixin.compute_screening_metrics(y_screening_dict, item_labels, params['lr'],
    #                                                              params['beta'])
    #             pre, rec, f_beta, loss, fn_count, fp_count = metrics
    #             budget_spent_item = (policy.B_al_spent + policy.B_crowd_spent) / items_num
    #             results_list.append([budget_per_item, budget_spent_item, pre, rec, f_beta, loss, fn_count,
    #                                  fp_count, switch_point])
    #
    #             print('budget spent per item: {:1.3f}, loss: {:1.3f}, fbeta: {:1.3f}, '
    #                   'recall: {:1.3f}, precisoin: {:1.3f}'
    #                   .format(budget_spent_item, loss, f_beta, rec, pre))
    #             print('--------------------------------------------------------------')
    #
    #         df = pd.DataFrame(results_list, columns=['budget_per_item', 'budget_spent_per_item',
    #                                                  'precision', 'recall', 'f_beta', 'loss',
    #                                                  'fn_count', 'fp_count', 'AL_switch_point'])
    #         df = compute_mean_std(df)
    #         df['active_learning_strategy'] = params['sampling_strategy'].__name__ if switch_point != 0 else ''
    #         df_to_print = df_to_print.append(df, ignore_index=True)
    #
    # file_name = params['dataset_file_name'][:-4] + '_experiment_nums_{}_ninstq_{}'.format(params['experiment_nums'],
    #                                                                                       params['n_instances_query'])
    # df_to_print.to_csv('../output/adaptive_machines_and_crowd/{}.csv'.format(file_name), index=False)


# set up active learning box
def configure_al_box(params, crowd_votes_counts, item_labels, item_ids_helper, X, y):
    size_init_train_data = params['size_init_train_data']

    # creating balanced init training data
    train_idx = get_init_training_data_idx(y, size_init_train_data)
    item_ids_helper = np.delete(item_ids_helper, train_idx)

    y_train_init = y[train_idx]
    y = np.delete(y, train_idx)
    X_train_init = X[train_idx]
    X = np.delete(X, train_idx, axis=0)

    for item_id, label in zip(train_idx, y_train_init):
        if label == 1:
            crowd_votes_counts[item_id]['in'] = params['max_votes_per_item'] / 2  # '/2' is optional
        else:
            crowd_votes_counts[item_id]['out'] = params['max_votes_per_item'] / 2
        item_labels[item_id] = label

    # setup active learner
    learner_params = {
        'clf': CalibratedClassifierCV(LinearSVC(class_weight='balanced', C=0.1)),
        'sampling_strategy': params['sampling_strategy'],
        'n_instances_query': params['n_instances_query'],
        'out_threshold': params['out_threshold']
    }
    L = Learner(learner_params)
    L.setup_active_learner(X_train_init, y_train_init, X, y)

    return L, item_ids_helper

# #
# # def compute_mean_std(df):
# #     columns_mean = [c + '_mean' for c in df.columns]
# #     columns_median = [c + '_median' for c in df.columns]
# #     columns_std = [c + '_std' for c in df.columns]
# #     old_columns = df.columns.values
# #     df_mean = df.mean().rename(dict(zip(old_columns, columns_mean)))
# #     df_std = df.std().rename(dict(zip(old_columns, columns_std)))
# #     df_median = df.median().rename(dict(zip(old_columns, columns_median)))
#
#     return pd.concat([df_mean, df_std, df_median])
