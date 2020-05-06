import logging
import pandas as pd
pd.set_option('display.max_columns', None)
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
# from transliterate import translit, get_available_language_codes
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import type_of_target
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

class Task:
    def __init__(self, data, target, type_of_task, type_of_importance_model, estimator_fit,
                 estimator_importance, metrics_to_estimate, n_splits, random_state):

        if data.shape[0] != target.shape[0]:
            raise ValueError('Shapes of data and target are different. Check it out!')

        allowed_types_of_task = ['classification', 'regression']
        if type_of_task not in allowed_types_of_task:
            raise AttributeError(f'type_of_task should be in {allowed_types_of_task}. Got: {type_of_task}')

        self.estimator_fit = estimator_fit
        self.type_of_importance_model = type_of_importance_model

        #! if linear -> preprocess

        self.type_of_task = type_of_task
        self.estimator_importance = estimator_importance
        self.data = data
        self.target = target
        self.metrics_to_estimate = metrics_to_estimate
        self.columns = self.data.columns

        self.random_state = random_state

        self.target_type = type_of_target(target)
        if self.target_type == 'continuous':
            self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        if self.target_type == 'multiclass' or self.target_type == 'binary':
            self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)


    def metrics_on_validation(self, data, target):

        cross_val_dict = cross_validate(estimator=self.estimator_fit,
                       X=data,
                       y=target,
                       cv=self.cv,
                       scoring=self.metrics_to_estimate)

        return {key: cross_val_dict['test_' + key].mean() for key in self.metrics_to_estimate}


    def preprocess_linear(self):
        return self.data


    def importance(self, top_k_features):
        if self.type_of_importance_model == "linear":
            return sorted(zip(self.columns, self.estimator_importance.coef_))[:top_k_features]
        else:
            return sorted(zip(self.columns, self.estimator_importance.feature_importances_))[:top_k_features]


    def run(self, top_k_features=20):

        # self.metrics_on_validation()

        self.estimator_importance.fit(X=self.data, y=self.target)

        top_features = self.importance(top_k_features)
        top_features_list = [val for val, _ in top_features]

        for i in range(len(top_features_list)):
            self.one_run(features_to_proceed=top_features_list[:i+1])



    def one_run(self, features_to_proceed):

        cross_val_dict_top_only = self.metrics_on_validation(data=self.data[features_to_proceed], target=self.target)
        cross_val_dict_top_omitted = self.metrics_on_validation(data=self.data.drop(columns=[features_to_proceed]),
                                                                target=self.target)

        return cross_val_dict_top_only, cross_val_dict_top_omitted

        # cross_val_dict = cross_validate(estimator=estimator_evaluation,
        #                                 X=data,
        #                                 y=target,
        #                                 cv=cv,
        #                                 scoring=list_of_metrics_to_use,
        #                                 n_jobs=-1)
        # scores = {key: cross_val_dict['test_' + key].mean() for key in list_of_metrics_to_use}
        #
        # whole_data_fitted_model = estimator_importance.fit(X=data, y=target)
        #
        #
        # top_features = sorted(zip(data.columns, whole_data_fitted_model.feature_importances_), key=lambda x: -x[1])[
        #                :top_k_features]
        # top_features_list = [val for val, _ in top_features]
        #
        # cross_val_dict_top_only = cross_validate(estimator=estimator_evaluation,
        #                                          X=data[top_features_list],
        #                                          y=target,
        #                                          cv=cv,
        #                                          scoring=list_of_metrics_to_use,
        #                                          n_jobs=-1)
        # scores_top_only = {key: cross_val_dict_top_only['test_' + key].mean() for key in list_of_metrics_to_use}
        #
        # cross_val_dict_top_omitted = cross_validate(estimator=estimator_evaluation,
        #                                             X=data.drop(columns=top_features_list),
        #                                             y=target,
        #                                             cv=cv,
        #                                             scoring=list_of_metrics_to_use,
        #                                             n_jobs=-1)
        # scores_top_omitted = {key: cross_val_dict_top_omitted['test_' + key].mean() for key in
        #                       list_of_metrics_to_use}
        #
        # scores_df = pd.DataFrame([scores, scores_top_only, scores_top_omitted],
        #                          index=['original', 'top_only', 'top_omitted'])
        # scores_df = scores_df.join(scores_df.apply(lambda x: pd.Series(data=x.iloc[0] - x), axis=0),
        #                            rsuffix='_diff')
        # scores_df['num_top_features'] = top_k_features
        #
        # return (scores, scores_top_only, scores_top_omitted), scores_df, top_features






