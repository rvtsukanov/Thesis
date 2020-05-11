from importance_learner import Task, preprocess_temp, preprocess_mag
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import itertools


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
         dataset=dataset).run(top_k_features=20, add_remove=False)
    print('==================')


## CLF

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
         dataset=dataset).run(top_k_features=20, add_remove=False)
    print('==================')