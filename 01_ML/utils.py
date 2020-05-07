from sklearn.utils.multiclass import type_of_target
import pandas as pd
pd.set_option('display.max_columns', None)
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
from transliterate import translit, get_available_language_codes
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def estimate_power_importance(data, 
                              target,
                              estimator_importance,
                              estimator_evaluation,
                              list_of_metrics_to_use=['accuracy', 'f1_macro'],
                              n_splits=5,
                              top_k_features=20):
    
    target_type = type_of_target(target)
    if target_type == 'continuous':
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    if target_type == 'multiclass' or target_type == 'binary':
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cross_val_dict = cross_validate(estimator=estimator_evaluation, 
                                    X=data,
                                    y=target,
                                    cv=cv,
                                    scoring=list_of_metrics_to_use, 
                                    n_jobs=-1)
    scores = {key: cross_val_dict['test_' + key].mean() for key in list_of_metrics_to_use}
    
    
    whole_data_fitted_model = estimator_importance.fit(X=data, y=target)
    
    if not hasattr(estimator_importance, 'feature_importances_'):
        raise AttributeError(f'Cannot estimate importance of {estimator_importance} estimator')
        
    top_features = sorted(zip(data.columns, whole_data_fitted_model.feature_importances_), key=lambda x: -x[1])[:top_k_features]
    top_features_list = [val for val, _ in top_features]
    
    
    cross_val_dict_top_only = cross_validate(estimator=estimator_evaluation, 
                                    X=data[top_features_list],
                                    y=target,
                                    cv=cv,
                                    scoring=list_of_metrics_to_use, 
                                    n_jobs=-1)
    scores_top_only = {key: cross_val_dict_top_only['test_' + key].mean() for key in list_of_metrics_to_use}
    
    
    
    cross_val_dict_top_omitted = cross_validate(estimator=estimator_evaluation, 
                                    X=data.drop(columns=top_features_list),
                                    y=target,
                                    cv=cv,
                                    scoring=list_of_metrics_to_use, 
                                    n_jobs=-1)
    scores_top_omitted = {key: cross_val_dict_top_omitted['test_' + key].mean() for key in list_of_metrics_to_use}
    

    scores_df = pd.DataFrame([scores, scores_top_only, scores_top_omitted],
                              index=['original', 'top_only', 'top_omitted'])
    scores_df = scores_df.join(scores_df.apply(lambda x: pd.Series(data=x.iloc[0] - x), axis=0), rsuffix='_diff')
    scores_df['num_top_features'] = top_k_features
    
    
    return (scores, scores_top_only, scores_top_omitted), scores_df, top_features