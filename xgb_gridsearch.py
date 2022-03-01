import pandas
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime
from datetime import timedelta
import pandas as pd
from itertools import product
from preprocessing_dataframes import *
from utils import *

EARLY_STOPPING_ROUNDS = 150  # stop if no improvement after 100 rounds
BOOST_ROUNDS = 10000  # we use early stopping so make this arbitrarily high
RANDOMSTATE = 42
EVAL_METRIC = 'rmse' #take from xgboost docs

def get_xgb_gridsearch_model(x_set, y_set, x_houldout_set, y_holdout_set):
    # Set the parameters by cross-validation
    xgb1 = XGBRegressor(verbosity=3)

    parameters = {'nthread': [-1],  # when use hyperthread, xgboost may become slower
                  'objective': ['reg:linear'],
                  'learning_rate': [0.003,0.005,0.01],  # so called `eta` value
                   'max_depth': [6],
                  'min_child_weight': [4],
                  'silent': [1],
                  'subsample': [0.75],
                  'colsample_bytree': [0.7],
                  'n_estimators': [5000]}

    fit_params = {
                  "early_stopping_rounds": 150,
                  "eval_metric": "mae",
                  "eval_set": [[x_houldout_set, y_holdout_set]]
                  }


    gridsearchcv_with_xgb = GridSearchCV(xgb1,
                            parameters,
                            cv=4,
                            n_jobs=-1,
                            verbose=True,
                            )

    gridsearchcv_with_xgb.fit(x_set, y_set, **fit_params)
    print(gridsearchcv_with_xgb.cv_results_)
    mean_score_array = gridsearchcv_with_xgb.cv_results_['mean_test_score']

    mean = sum(mean_score_array)/len(mean_score_array)
    print("mean score array:",mean_score_array)
    print(mean)

    print("xgb gridsearch best params",gridsearchcv_with_xgb.best_params_)
    #
    # kfold = KFold(n_splits=5, random_state=123,shuffle=True)
    # results = cross_val_score(model, dataframe, y, cv=kfold,verbose=1,n_jobs=-1)
    # print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    return gridsearchcv_with_xgb


def custom_cv(x_set, y_set, kfold_splitter, xgb_regressor, verbose=False):
    metrics = []
    best_iterations = []

    for train_index, validation_index in kfold_splitter.split(x_set,y_set):
        x_train = x_set.iloc[train_index]
        y_train = y_set.iloc[train_index]
        x_validation = x_set.iloc[validation_index]
        y_validation = y_set.iloc[validation_index]

        xgb_regressor.fit(x_train, y_train, eval_set=[(x_validation, y_validation)],
                          eval_metric=EVAL_METRIC, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                          verbose=verbose)

        y_pred = xgb_regressor.predict(x_validation)
        rmse = np.sqrt(mean_squared_error(y_validation, y_pred))
        metrics.append(rmse)
        best_iterations.append(xgb_regressor.best_iteration)

    return np.average(metrics), np.std(metrics), np.average(best_iterations)



def get_params_xgb_gridsearch_with_kfold(x_set, y_set, param_dict, kfold,x_holdout_set,y_holdout_set):
    # Set the parameters by cross-validation
    verbose = False
    results = []
    start_time = datetime.now()
    print("%-20s %s" % ("Start Time", start_time))

    for i, d in enumerate(param_dict):
        xgb_regressor = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=BOOST_ROUNDS,
            random_state=RANDOMSTATE,
            n_jobs=-1,
            booster='gbtree',
            **d
        )
        metric_rmse, metric_std, best_iteration = custom_cv(x_set, y_set, kfold, xgb_regressor, verbose=False)

        y_prediction_for_holdout_set=xgb_regressor.predict(x_holdout_set)
        rmse_holdout = rmse(y_holdout_set,y_prediction_for_holdout_set)

        results.append([rmse_holdout,metric_rmse, metric_std, best_iteration, d])

        print("%s %3d rmse_on_holdout: %.6f result mean: %.6f std: %.6f, iter: %.2f" % (
            datetime.strftime(datetime.now(), "%T"), i,rmse_holdout, metric_rmse, metric_std, best_iteration))

    end_time = datetime.now()
    print("%-20s %s" % ("Start Time", start_time))
    print("%-20s %s" % ("End Time", end_time))
    print(str(timedelta(seconds=(end_time - start_time).seconds)))

    results_df = pd.DataFrame(results, columns=['rmse_on_holdout','rmse', 'std', 'best_iter', 'param_dict']).sort_values('rmse_on_holdout')
    print(results_df.head())

    best_params = results_df.iloc[0]['param_dict']
    return best_params, results_df


def my_cv_druce(df, predictors, response, kfolds, regressor, verbose=False):
    """Roll our own CV
    train each kfold with early stopping
    return average metric, sd over kfolds, average best round"""
    metrics = []
    best_iterations = []

    for train_fold, cv_fold in kfolds.split(df):
        fold_X_train=df[predictors].values[train_fold]
        fold_y_train=df[response].values[train_fold]
        fold_X_test=df[predictors].values[cv_fold]
        fold_y_test=df[response].values[cv_fold]
        regressor.fit(fold_X_train, fold_y_train,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      eval_set=[(fold_X_test, fold_y_test)],
                      eval_metric='rmse',
                      verbose=verbose
                     )
        y_pred_test=regressor.predict(fold_X_test)

        metrics.append(rmse(fold_y_test, y_pred_test))
        best_iterations.append(regressor.best_iteration)
    return np.average(metrics), np.std(metrics), np.average(best_iterations)

def cv_over_param_dict(df, param_dict, predictors, response, kfolds, verbose=False):
    return

# def cv_over_param_dict_druce(df, param_dict, predictors, response, kfolds, verbose=False):
#     """given a list of dictionaries of xgb params
#     run my_cv on params, store result in array
#     return updated param_dict, results dataframe
#     """
#     start_time = datetime.now()
#     print("%-20s %s" % ("Start Time", start_time))
#
#     results = []
#
#     for i, d in enumerate(param_dict):
#         xgb = XGBRegressor(
#             objective='reg:squarederror',
#             n_estimators=BOOST_ROUNDS,
#             random_state=RANDOMSTATE,
#             verbosity=1,
#             n_jobs=-1,
#             booster='gbtree',
#             **d
#         )
#
#         metric_rmse, metric_std, best_iteration = my_cv_druce(df, predictors, response, kfolds, xgb, verbose=False)
#         results.append([metric_rmse, metric_std, best_iteration, d])
#
#         print("%s %3d result mean: %.6f std: %.6f, iter: %.2f" % (
#         datetime.strftime(datetime.now(), "%T"), i, metric_rmse, metric_std, best_iteration))
#
#     end_time = datetime.now()
#     print("%-20s %s" % ("Start Time", start_time))
#     print("%-20s %s" % ("End Time", end_time))
#     print(str(timedelta(seconds=(end_time - start_time).seconds)))
#
#     results_df = pd.DataFrame(results, columns=['rmse', 'std', 'best_iter', 'param_dict']).sort_values('rmse')
#     print(results_df.head())
#
#     best_params = results_df.iloc[0]['param_dict']
#     return best_params, results_df


# initial hyperparams
# current_params = {
#     'max_depth': 5,
#     # 'colsample_bytree': 0.5,
#     # 'colsample_bylevel': 0.5,
#     # 'subsample': 0.5,
#     'learning_rate': 0.01,
# }

# na_values = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NULL", "NaN", "n/a", "nan", "null"]
# #these csv's can;t be read well in open office after the processing
# train_original = pd.read_csv('processed data/train_processed.csv', index_col='Id',na_values=na_values, keep_default_na=False)
# test_original = pd.read_csv('processed data/test_processed.csv', index_col='Id',na_values=na_values, keep_default_na=False)
#
# df, testdf = load_and_preprocess_dataframes(train_original, test_original)
#
#
# kfolds = KFold(n_splits=10, shuffle=True, random_state=RANDOMSTATE)
#
# ##################################################
# # round 1: tune depth
# ##################################################
# max_depths = list(range(5, 7))
# grid_search_dicts = [{'max_depth': md} for md in max_depths]
# # merge into full param dicts
# full_search_dicts = [{**current_params, **d} for d in grid_search_dicts]
#
# # cv and get best params
# current_params, results_df = cv_over_param_dict_druce(df, full_search_dicts, df.columns,'SalePrice', kfolds)
#
# ##################################################
# # round 2: tune subsample, colsample_bytree, colsample_bylevel
# ##################################################
# # subsamples = np.linspace(0.01, 1.0, 10)
# # colsample_bytrees = np.linspace(0.1, 1.0, 10)
# # colsample_bylevel = np.linspace(0.1, 1.0, 10)
# # narrower search
# subsamples = np.linspace(0.25, 0.75, 11)
# colsample_bytrees = np.linspace(0.1, 0.3, 5)
# colsample_bylevel = np.linspace(0.1, 0.3, 5)
# # subsamples = np.linspace(0.4, 0.9, 11)
# # colsample_bytrees = np.linspace(0.05, 0.25, 5)
#
# grid_search_dicts = [dict(zip(['subsample', 'colsample_bytree', 'colsample_bylevel'], [a, b, c]))
#                      for a, b, c in product(subsamples, colsample_bytrees, colsample_bylevel)]
# # merge into full param dicts
# full_search_dicts = [{**current_params, **d} for d in grid_search_dicts]
# # cv and get best params
# current_params, results_df = cv_over_param_dict_druce(df, full_search_dicts, predictors, response, kfolds)
#
# # round 3: learning rate
# learning_rates = np.logspace(-3, -1, 5)
# grid_search_dicts = [{'learning_rate': lr} for lr in learning_rates]
# # merge into full param dicts
# full_search_dicts = [{**current_params, **d} for d in grid_search_dicts]
#
# # cv and get best params
# current_params, results_df = cv_over_param_dict(df, full_search_dicts, predictors, response, kfolds, verbose=False)