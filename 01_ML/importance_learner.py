import logging
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import type_of_target
import pickle
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import os
from transliterate import translit
import matplotlib
matplotlib.rc('font', **{'size': 20})


def sort_cols_by_type(data):
    multiclass_cols = []
    binary_cols = []
    cont_cols = []
    other_cols = []

    for col in data:
        if type_of_target(data[col]) == "multiclass":
            multiclass_cols.append(col)
        elif type_of_target(data[col]) == "binary":
            binary_cols.append(col)
        elif type_of_target(data[col]) == "continuous":
            cont_cols.append(col)
        else:
            #index col usually drops here
            other_cols.append(col)
    return multiclass_cols, binary_cols, cont_cols, other_cols


def preprocess_temp(path_data='../DATA/02_TEMP/X_train.csv',
                    path_target='../DATA/02_TEMP/Y_train.csv',
                    use_linear_preprocessing=False,
                    use_translit=True,
                    use_multiclass=False,
                    extra_columns_to_categorize=(),
                    scale=False):


    data = pd.read_csv(path_data, index_col=None)
    target = pd.read_csv(path_target)['Температура']

    if use_translit:
        transliteration_mapping = {col: translit(col, 'ru', reversed=True) for col in data.columns}
        transliteration_mapping_inverse = {val: key for key, val in transliteration_mapping.items()}
        data.columns = transliteration_mapping_inverse.keys()

    if use_linear_preprocessing:
        multiclass_cols, binary_cols, cont_cols, other_cols = sort_cols_by_type(data)
        data = pd.get_dummies(data, columns=multiclass_cols * use_multiclass + binary_cols + list(extra_columns_to_categorize))
        data = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns, index=data.index)

    if scale:
        data = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns, index=data.index)


    return data, target


def preprocess_mag(path_data='../DATA/01_MAG/first.csv',
                    path_target='../DATA/01_MAG/first_targets.csv',
                    use_linear_preprocessing=False,
                    use_translit=True,
                    use_multiclass=False,
                    extra_columns_to_categorize=(),
                    target_type='reg',
                    scale=False):

    if target_type not in ['clf', 'reg', 'bimod']:
        raise AttributeError(f'target_type should be in [clf, reg, bimod]')


    data = pd.read_csv(path_data, index_col=None).drop(columns=['Unnamed: 0'])
    target = pd.read_csv(path_target, index_col=None, names=['magnetic_loss']).reset_index().drop(
        columns=['index'])['magnetic_loss']

    if use_translit:
        transliteration_mapping = {col: translit(col, 'ru', reversed=True) for col in data.columns}
        transliteration_mapping_inverse = {val: key for key, val in transliteration_mapping.items()}
        data.columns = transliteration_mapping_inverse.keys()

    if use_linear_preprocessing:
        multiclass_cols, binary_cols, cont_cols, other_cols = sort_cols_by_type(data)
        data = pd.get_dummies(data, columns=multiclass_cols * use_multiclass + binary_cols + list(extra_columns_to_categorize))
        data = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns, index=data.index)

    if scale:
        data = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns, index=data.index)


    if target_type == 'clf':
        target = pd.cut(target, [-np.inf, 1.05, 1.26, np.inf], labels=[0, 1, 2])

    if target_type == 'bimod':
        target = pd.cut(target, [-np.inf, 1.19, np.inf], labels=[0, 1])

    return data, target


class Task:
    def __init__(self, data, target, type_of_importance_model, estimator_fit,
                 estimator_importance, metrics_to_estimate, n_splits, random_state, dataset, postfix=''):

        if data.shape[0] != target.shape[0]:
            raise ValueError('Shapes of data and target are different. Check it out!')

        allowed_types_of_task = ['classification', 'regression']
        if dataset.upper() not in ['MAG', 'TEMP']:
            raise AttributeError(f'dataset should be in [MAG, TEMP]')



        self.estimator_fit = estimator_fit
        self.estimator_fit_alias = self.return_alias(self.estimator_fit.__class__.__name__)

        self.estimator_importance = estimator_importance
        self.estimator_importance_alias = self.return_alias(self.estimator_importance.__class__.__name__)

        self.type_of_importance_model = type_of_importance_model
        self.dataset = dataset

        self.columns = data.columns
        self.dataset = dataset.upper()

        self.data = data
        self.target = target
        self.metrics_to_estimate = metrics_to_estimate
        self.folder_upper_level = 'experiments'
        self.postfix = postfix

        self.path = os.path.join(self.folder_upper_level, self.dataset, self._generate_subdir_name())
        os.makedirs(self.path)

        self.random_state = random_state

        self.target_type = type_of_target(target)
        if self.target_type == 'continuous':
            self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        if self.target_type == 'multiclass' or self.target_type == 'binary':
            self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    @staticmethod
    def sort_cols_by_type(data):
        multiclass_cols = []
        binary_cols = []
        cont_cols = []
        other_cols = []

        for col in data:
            if type_of_target(data[col]) == "multiclass":
                multiclass_cols.append(col)
            elif type_of_target(data[col]) == "binary":
                binary_cols.append(col)
            elif type_of_target(data[col]) == "continuous":
                cont_cols.append(col)
            else:
                print(type_of_target(data[col]), col)
                other_cols.append(col)
        return multiclass_cols, binary_cols, cont_cols, other_cols


    def return_alias(self, name):
        if name == "RandomForestClassifier":
            return 'RFClf'
        if name == "RandomForestRegressor":
            return 'RFReg'
        if name == "LGBMClassifier":
            return "LGBClf"
        if name == "LGBMRegressor":
            return "LGBReg"
        if name == "LinearRegression":
            return "LinReg"
        if name == "LogisticRegression":
            return "LogReg"
        return name


    def _generate_subdir_name(self, time=None) -> str:
        '''
        Name generation from current time
        :return: str: directory name
        '''
        if not time:
            now = datetime.datetime.now()
        else:
            now = time

        return f'{self.estimator_fit_alias}-{self.estimator_importance_alias}-{self.dataset}-{self.postfix}-{now.strftime("%d-%m-%Y--%H-%M-%S")}'


    def metrics_on_validation(self, data, target):
        cross_val_dict = cross_validate(estimator=self.estimator_fit,
                       X=data,
                       y=target,
                       cv=self.cv,
                       scoring=self.metrics_to_estimate)
        return {key: cross_val_dict['test_' + key].mean() for key in self.metrics_to_estimate}


    def save_myself(self):
        pickle.dump(self, open(os.path.join(self.path, 'model.pickle'), 'wb'))


    def importance(self, top_k_features):
        if self.type_of_importance_model == "linear":
            return sorted(zip(self.columns, self.estimator_importance.coef_), key=lambda x: -x[1])[:top_k_features]
        else:
            return sorted(zip(self.columns, self.estimator_importance.feature_importances_), key=lambda x: -x[1])[:top_k_features]



    def save_features(self, top):
        self.top_features_list = [val for val, _ in top]

        pd.DataFrame(self.top_features).to_csv(os.path.join(self.path, 'top_features.csv'))

        with open(os.path.join(self.path, 'top_features_list.txt'), 'w+') as f:
            for line in self.top_features_list:
                f.write(line + '\n')



    def save_plots(self, concat_scores):
        print('CS:!', concat_scores)
        metrics_to_plot = [metric for metric in concat_scores.columns if metric.startswith('test_')]
        if len(metrics_to_plot) == 0:
            metrics_to_plot = [metric for metric in self.metrics_to_estimate]

        for metric in metrics_to_plot:
            plt.figure(figsize=(20, 15))
            plt.title(f'{self.estimator_fit_alias}-{self.estimator_importance_alias}-{self.postfix}-{self.dataset}-{metric}')

            plt.plot(concat_scores[concat_scores.type == 'omit']['num_features'],
                     concat_scores[concat_scores.type == 'omit'][metric],
                     label='skip-n-features-from-top')

            plt.plot(concat_scores[concat_scores.type == 'only']['num_features'],
                     concat_scores[concat_scores.type == 'only'][metric],
                     label='leave-n-features-from-top')

            plt.xlabel('num features')
            plt.ylabel('score value')
            plt.xticks(concat_scores[concat_scores.type == 'omit']['num_features'])

            plt.legend()
            plt.grid()

            plt.savefig(os.path.join(self.path, f'fig_{metric}'))



    def run(self, top_k_features=20, add_remove=False):

        whole = self.metrics_on_validation(data=self.data, target=self.target)
        pd.DataFrame(whole, index=['whole']).to_csv(os.path.join(self.path, 'whole.csv'))

        self.estimator_importance.fit(X=self.data, y=self.target)

        self.top_features = self.importance(top_k_features)
        self.save_features(self.top_features)

        scores = []

        if add_remove:

            for i in range(len(self.top_features_list)):
                print(i, self.top_features_list[:i+1])
                only, omit = self.one_run(features_to_proceed=self.top_features_list[:i+1])

                only['num_features'] = i + 1
                only['type'] = 'only'

                omit['num_features'] = i + 1
                omit['type'] = 'omit'

                scores.append(only)
                scores.append(omit)

                pd.DataFrame(scores).to_csv(os.path.join(self.path, 'add_remove_scores.csv'))

                self.save_myself()

        self.save_myself()

        self.save_plots(pd.DataFrame(scores))
        return self



    def one_run(self, features_to_proceed):

        cross_val_dict_top_only = self.metrics_on_validation(data=self.data[features_to_proceed], target=self.target)
        cross_val_dict_top_omitted = self.metrics_on_validation(data=self.data.drop(columns=features_to_proceed),
                                                                target=self.target)

        return cross_val_dict_top_only, cross_val_dict_top_omitted