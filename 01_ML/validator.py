from importance_learner import Task, preprocess_temp, preprocess_mag
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
import itertools
from transliterate import translit, get_available_language_codes
from itertools import product

# data_mag = pd.read_csv('../DATA/01_MAG/first.csv', index_col=None).drop(columns=['Unnamed: 0'])
# target_mag = pd.read_csv('../DATA/01_MAG/first_targets.csv', index_col=None, names=['magnetic_loss']).reset_index().drop(columns=['index'])['magnetic_loss']
#
# transliteration_mapping = {col: translit(col, 'ru', reversed=True) for col in data_mag.columns}
# transliteration_mapping_inverse = {val: key for key, val in transliteration_mapping.items()}
# data_mag.columns = transliteration_mapping_inverse.keys()

# data = data[data.columns[:10]].sample(1000)
# target = target[data.index]

#Prepare targets
#CLASSIFICATION
# y_clf = pd.cut(target_mag, [-np.inf, 1.05, 1.26, np.inf], labels=[0, 1, 2])

#REGRESSION
# y_reg = target_mag


#BIMODAL CLF
# y_bimod = pd.cut(target_mag, [-np.inf, 1.19, np.inf], labels=[0, 1])


# data_temp = pd.read_csv('../DATA/02_TEMP/X_train.csv', index_col=None)
# target_temp = pd.read_csv('../DATA/02_TEMP/Y_train.csv')['Температура']
#
# transliteration_mapping = {col: translit(col, 'ru', reversed=True) for col in data_temp.columns}
# transliteration_mapping_inverse = {val: key for key, val in transliteration_mapping.items()}
# data_temp.columns = transliteration_mapping_inverse.keys()

# print(data_temp[data_temp.columns[:10]].sample(5000).shape)
# print(target_temp[data_temp.index].shape)

data_clf_mag, target_clf_mag = preprocess_mag(target_type='clf')
data_bimod_mag, target_bimod_mag = preprocess_mag(target_type='bimod')
data_reg_mag, target_reg_mag = preprocess_mag(target_type='reg')

data_temp, target_temp = preprocess_temp()


reg_data = [("MAG", data_reg_mag, target_reg_mag), ("TEMP", data_temp, target_temp)]
clf_data = [("MAG", data_clf_mag, target_clf_mag), ("MAG", data_bimod_mag, target_bimod_mag)]

estimators_reg = [LGBMRegressor(), RandomForestRegressor(n_estimators=100)]
estimators_clf = [LGBMClassifier(), RandomForestClassifier(n_estimators=100)]

metrics_reg = ['r2', 'neg_mean_squared_error']
metrics_clf = ['accuracy', 'f1_macro']


##REG

for (dataset, data, target), estimator in itertools.product(reg_data, estimators_reg):
    print('==================')
    print('REG')
    print('dataset: ', dataset)
    print('estimator: ', estimator)
    Task(data=data,
         target=target,
         type_of_importance_model='nonlinear',
         estimator_fit=estimator,
         estimator_importance=estimator,
         metrics_to_estimate=metrics_reg,
         n_splits=6,
         random_state=42,
         dataset=dataset).run(top_k_features=20)
    print('==================')



for (dataset, data, target), estimator in itertools.product(clf_data, estimators_clf):
    print('==================')
    print('CLF')
    print('dataset: ', dataset)
    print('estimator: ', estimator)
    Task(data=data,
         target=target,
         type_of_importance_model='nonlinear',
         estimator_fit=estimator,
         estimator_importance=estimator,
         metrics_to_estimate=metrics_reg,
         n_splits=6,
         random_state=42,
         dataset=dataset).run(top_k_features=20)
    print('==================')


# def prepare_data_for_linear
#
# data = data_mag
# for target in [y_clf, y_bimod]:
#      # Task(data=data,
#      #      target=target,
#      #      type_of_task='classification',
#      #      type_of_importance_model='linear',
#      #      estimator_fit=LogisticRegression(),
#      #      estimator_importance=LogisticRegression(),
#      #      metrics_to_estimate=['accuracy', 'f1_macro'], n_splits=5, random_state=42, dataset='mag').run().save_myself()
#
#      Task(data=data,
#           target=target,
#           type_of_task='classification',
#           type_of_importance_model='nonlinear',
#           estimator_fit=LGBMClassifier(),
#           estimator_importance=LGBMClassifier(),
#           metrics_to_estimate=['accuracy', 'f1_macro'], n_splits=5, random_state=42, dataset='mag').run().save_myself()
#
#
#      Task(data=data,
#           target=target,
#           type_of_task='classification',
#           type_of_importance_model='nonlinear',
#           estimator_fit=RandomForestClassifier(n_estimators=100),
#           estimator_importance=RandomForestClassifier(n_estimators=100),
#           metrics_to_estimate=['accuracy', 'f1_macro'], n_splits=5, random_state=42, dataset='mag').run().save_myself()
#
# target = y_reg
#
# Task(data=data,
#      target=target,
#      type_of_task='regression',
#      type_of_importance_model='linear',
#      estimator_fit=LinearRegression(),
#      estimator_importance=LinearRegression(),
#      metrics_to_estimate=['r2', 'neg_mean_squared_error'], n_splits=5, random_state=42, dataset='mag').run().save_myself()
#
# Task(data=data,
#      target=target,
#      type_of_task='regression',
#      type_of_importance_model='nonlinear',
#      estimator_fit=LGBMRegressor(),
#      estimator_importance=LGBMRegressor(),
#      metrics_to_estimate=['r2', 'neg_mean_squared_error'], n_splits=5, random_state=42, dataset='mag').run().save_myself()
#
# Task(data=data,
#      target=target,
#      type_of_task='regression',
#      type_of_importance_model='nonlinear',
#      estimator_fit=RandomForestRegressor(n_estimators=100),
#      estimator_importance=RandomForestRegressor(n_estimators=100),
#      metrics_to_estimate=['r2', 'neg_mean_squared_error'], n_splits=5, random_state=42, dataset='mag').run().save_myself()
#
#
#
# data = data_temp
#
# target = target_temp
#
# Task(data=data,
#      target=target,
#      type_of_task='regression',
#      type_of_importance_model='linear',
#      estimator_fit=LinearRegression(),
#      estimator_importance=LinearRegression(),
#      metrics_to_estimate=['r2', 'neg_mean_squared_error'], n_splits=5, random_state=42, dataset='temp').run().save_myself()
#
# Task(data=data,
#      target=target,
#      type_of_task='regression',
#      type_of_importance_model='nonlinear',
#      estimator_fit=LGBMRegressor(),
#      estimator_importance=LGBMRegressor(),
#      metrics_to_estimate=['r2', 'neg_mean_squared_error'], n_splits=5, random_state=42, dataset='temp').run().save_myself()
#
# Task(data=data,
#      target=target,
#      type_of_task='regression',
#      type_of_importance_model='nonlinear',
#      estimator_fit=RandomForestRegressor(n_estimators=100),
#      estimator_importance=RandomForestRegressor(n_estimators=100),
#      metrics_to_estimate=['r2', 'neg_mean_squared_error'], n_splits=5, random_state=42, dataset='temp').run().save_myself()