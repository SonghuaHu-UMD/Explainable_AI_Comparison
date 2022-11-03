# ##############
# Use origin data without transform: linear regression has better performance
# ##############
import optuna
import glob
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import pickle5 as pickle
import pickle
import lightgbm as lgb
import catboost as catb
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import HuberRegressor, Lasso, Ridge, ElasticNet, LassoLars, BayesianRidge
from pycaret.regression import *
from pycaret.utils import check_metric
import time

# dir_path = r'C:\\Users\\songhua\\Cross_Nonlinear\\'
dir_path = r'D:\\Cross_Nonlinear\\'
Dnu = '0630'
Target_n = 'Mobility'
Is_std = 'nonst'
a_tr = 40
s_tr = 20


def mean_absolute_percentage_error(Validate, Predict):
    return np.mean(np.abs((Validate - Predict) / Validate))


####################################################
# Step 0: Prepare data
####################################################
# Read data
data = pd.read_pickle(dir_path + r'Data\visit_data_origin_train.pkl')
data_unseen = pd.read_pickle(dir_path + r'Data\visit_data_origin_test.pkl')

# Set target
p_tr = ['Mobility', 'Trip_Rate', 'Trip_Density']
data['Trip_Rate'] = data['Mobility'] / (data['Total Population'] * 1e4)
data['Trip_Density'] = data['Mobility'] / (data['Area'])
data_unseen['Trip_Rate'] = data_unseen['Mobility'] / (data_unseen['Total Population'] * 1e4)
data_unseen['Trip_Density'] = data_unseen['Mobility'] / (data_unseen['Area'])
p_tr.remove(Target_n)
# Drop others
data = data.drop(['BGFIPS'] + p_tr, axis=1)
data_unseen = data_unseen.drop(['BGFIPS'] + p_tr, axis=1)

####################################################
# Step 1: Directly compare models using default parameters
####################################################
# categorical_features=category_label,high_cardinality_features=['CTFIPS'],remove_perfect_collinearity=True,
exp_reg = setup(data=data, target=Target_n, session_id=2, fold=5, normalize=False, transformation=False,
                transform_target=False, use_gpu=False, n_jobs=-1)

# Compare all models
best_models_va = compare_models(exclude=['ransac', 'lar', 'ada', 'par', 'ard', 'tr', 'kr', 'svm', 'gbr', 'knn'],
                                sort='MAPE', n_select=15, turbo=False)
# Extract results
for kk in exp_reg:
    if type(kk) is list:
        if type(kk[0]) is pd.io.formats.style.Styler: model_metric = kk[1]

# Save models and results
model_metric.to_csv(dir_path + r'Results\\visit_model_metric_%s_%s_%s_va.csv' % (Dnu, Is_std, Target_n))
with open(dir_path + r'Results\\visit_best_record_%s_%s_%s_va.pkl' % (Dnu, Is_std, Target_n), 'wb') as h:
    pickle.dump(best_models_va, h, protocol=pickle.HIGHEST_PROTOCOL)

####################################################
# Step 2: Build the tuning process in pure optuna
####################################################
data_dm_x = data.drop([Target_n], axis=1)  # only features
data_raw_train = data.copy()  # features + target


def tuning_result(date_num, objective_lgb, n_trials, data_unseen):
    '''
    date_num = '1027-xgb-result'
    objective_lgb = objective_xgb
    n_trials = 2
    '''
    path_loc = os.path.join(os.getcwd(), "optuna_" + date_num)
    if not os.path.isdir(path_loc): os.makedirs(path_loc)
    # Build the study
    study_lgb = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="minimize")
    study_lgb.optimize(lambda trial: objective_lgb(trial, data_dm_x, data_raw_train, date_num), n_trials=n_trials)
    # Store the study
    with open(os.path.join(path_loc, "st_%s.pkl" % date_num), "wb") as f:
        pickle.dump(study_lgb, f)
    # # Visualize the study
    # optuna.visualization.plot_intermediate_values(study_lgb).show(renderer="browser")
    # optuna.visualization.plot_contour(study_lgb).show(renderer="browser")
    # optuna.visualization.plot_optimization_history(study_lgb).show(renderer="browser")
    # Load the best model
    trial = study_lgb.best_trial  # best_model = study_lgb.user_attrs["best_booster"]
    matching = [s for s in glob.glob(".\\%s\\*" % ("optuna_" + date_num)) if (str(trial.value) in s) and ('ty_' in s)]
    with open(matching[0], "rb") as fh:
        best_model = pickle.load(fh)
    # for key, value in trial.params.items(): print("    {}: {}".format(key, value))
    # Predict on testing
    if type(best_model) == xgb.core.Booster:
        data_unseen_x = data_unseen.drop(Target_n, axis=1)
        data_unseen_y = data_unseen[Target_n]
        unseen_predictions = pd.DataFrame({'Label': best_model.predict(xgb.DMatrix(data_unseen_x, data_unseen_y)),
                                           Target_n: data_unseen_y})
    else:
        unseen_predictions = predict_model(best_model, data=data_unseen)
    # sns.regplot(x="Mobility", y="Label", data=unseen_predictions)
    print('MAPE: %s' % check_metric(unseen_predictions[Target_n], unseen_predictions.Label, 'MAPE'))
    return study_lgb, best_model


## Tune for lightgbm
def objective_lgb_sk(trial, data_dm_x, data_raw_train, date_num):
    param = {"objective": "regression", "metric": "mape", "verbosity": -1, "boosting_type": "gbdt",
             'feature_pre_filter': False, "n_estimators": trial.suggest_int("n_estimators", 100, 500, 50),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, step=0.01),
             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
             "max_depth": trial.suggest_int("max_depth", 5, 100, step=5),
             "num_leaves": trial.suggest_int("num_leaves", 32, 512, step=32),  # num_leaf, max_leaves, max_leaf 512
             'min_sum_hessian_in_leaf': trial.suggest_int("min_sum_hessian_in_leaf", 0, 100),
             "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),  # sub_feature, colsample_bytree,
             # "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),  # sub_row, subsample, bagging
             # "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),  # subsample_freq
             # "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
             }
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(data_dm_x)))):
        # Add a callback for pruning
        # pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "mape")
        gbm = lgb.LGBMRegressor(**param)
        gbm.fit(data_dm_x.loc[train_idx], data_raw_train.loc[train_idx, Target_n],
                eval_set=[(data_dm_x.loc[valid_idx], data_raw_train.loc[valid_idx, Target_n])], eval_metric='mape',
                verbose=False)
        trial.set_user_attr(key="best_booster", value=copy.deepcopy(gbm))
        accuracy = mean_absolute_percentage_error(data_raw_train.loc[valid_idx, Target_n],
                                                  gbm.predict(data_dm_x.loc[valid_idx]))
        scores.append(accuracy)
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(gbm, f)
    return np.mean(scores)


def objective_xgb_sk(trial, data_dm_x, data_raw_train, date_num):
    # Not use dart as shp not support yet
    param = {"verbosity": 0, "n_jobs": -1, "objective": "reg:squarederror", "eval_metric": "mape", "booster": "gbtree",
             "n_estimators": trial.suggest_int("n_estimators", 100, 500, 50),
             "lambda": trial.suggest_float("lambda", 1e-8, 10, log=True),
             "alpha": trial.suggest_float("alpha", 1e-8, 10, log=True),
             "max_depth": trial.suggest_int("max_depth", 1, 15),
             "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.08, step=0.01),
             "subsample": trial.suggest_float("subsample", 0.5, 1.0),
             "min_child_weight": trial.suggest_int("min_child_weight", 0, 10),
             "colsample_by*": trial.suggest_float("colsample_by*", 0, 1)}

    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(data_dm_x)))):
        # Add a callback for pruning
        # pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-mape")
        gbm = xgb.XGBRegressor(**param)
        gbm.fit(data_dm_x.loc[train_idx], data_raw_train.loc[train_idx, Target_n],
                eval_set=[(data_dm_x.loc[valid_idx], data_raw_train.loc[valid_idx, Target_n])], eval_metric='mape',
                verbose=False)
        trial.set_user_attr(key="best_booster", value=copy.deepcopy(gbm))
        accuracy = mean_absolute_percentage_error(data_raw_train.loc[valid_idx, Target_n],
                                                  gbm.predict(data_dm_x.loc[valid_idx]))
        scores.append(accuracy)
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(gbm, f)
    return np.mean(scores)


## Tune for catboost
def objective_catb(trial, data_dm_x, data_raw_train, date_num):
    param = {"logging_level": 'Verbose', 'eval_metric': 'MAPE', "n_estimators": 1200,
             # "task_type": "GPU", 'devices': '0',
             "max_depth": trial.suggest_int("max_depth", 4, 12),
             "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
             "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.08, step=0.01), }
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(data_dm_x)))):
        gbm = catb.CatBoostRegressor(**param)
        gbm.fit(data_dm_x.loc[train_idx], data_raw_train.loc[train_idx, Target_n],
                eval_set=[(data_dm_x.loc[valid_idx], data_raw_train.loc[valid_idx, Target_n])],
                verbose=0, early_stopping_rounds=100)
        accuracy = mean_absolute_percentage_error(data_raw_train.loc[valid_idx, Target_n],
                                                  gbm.predict(data_dm_x.loc[valid_idx]))
        scores.append(accuracy)
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(gbm, f)
    return np.mean(scores)


# Random Forest Regressor
def objective_rf(trial, data_dm_x, data_raw_train, date_num):
    param = {"n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
             "max_depth": trial.suggest_int("max_depth", 10, 300, step=10),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, step=3),
             "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, step=1), }
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(data_dm_x)))):
        gbm = RandomForestRegressor(**param, n_jobs=-1)
        gbm.fit(data_dm_x.loc[train_idx], data_raw_train.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(data_raw_train.loc[valid_idx, Target_n],
                                                  gbm.predict(data_dm_x.loc[valid_idx]))
        scores.append(accuracy)

    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(gbm, f)
    return np.mean(scores)


# ExtraTreesRegressor
def objective_etr(trial, data_dm_x, data_raw_train, date_num):
    param = {"n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
             "max_depth": trial.suggest_int("max_depth", 10, 300, step=10),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, step=3),
             "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, step=1), }
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(data_dm_x)))):
        gbm = ExtraTreesRegressor(**param, n_jobs=-1)
        gbm.fit(data_dm_x.loc[train_idx], data_raw_train.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(data_raw_train.loc[valid_idx, Target_n],
                                                  gbm.predict(data_dm_x.loc[valid_idx]))
        scores.append(accuracy)

    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(gbm, f)
    return np.mean(scores)


# DecisionTreeRegressor
def objective_dct(trial, data_dm_x, data_raw_train, date_num):
    param = {"max_depth": trial.suggest_int("max_depth", 1, 30, step=2),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, step=2),
             "min_samples_split": trial.suggest_int("min_samples_split", 2, 20, step=2), }
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(data_dm_x)))):
        gbm = DecisionTreeRegressor(**param)
        gbm.fit(data_dm_x.loc[train_idx], data_raw_train.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(data_raw_train.loc[valid_idx, Target_n],
                                                  gbm.predict(data_dm_x.loc[valid_idx]))
        scores.append(accuracy)
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(gbm, f)
    return np.mean(scores)


# MLPRegressor
def objective_mlp(trial, data_dm_x, data_raw_train, date_num):
    param = {"max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
             "alpha": trial.suggest_float("alpha", 0.000001, 0.001, log=True),
             "hidden_layer_sizes": trial.suggest_categorical(
                 "hidden_layer_sizes",
                 [(200,), (150,), (100,), (50,), (200, 150), (150, 100), (100, 50), (150, 100, 50)]), }
    # "activation": trial.suggest_categorical("activation", ['tanh', 'relu'])
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(data_dm_x)))):
        huberlinear = MLPRegressor(**param, learning_rate='constant', early_stopping=True, tol=0.001,
                                   n_iter_no_change=50)
        huberlinear.fit(data_dm_x.loc[train_idx], data_raw_train.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(data_raw_train.loc[valid_idx, Target_n],
                                                  huberlinear.predict(data_dm_x.loc[valid_idx]))
        scores.append(accuracy)
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(huberlinear, f)
    return np.mean(scores)


# HuberRegressor
def objective_hul(trial, data_dm_x, data_raw_train, date_num):
    param = {"alpha": trial.suggest_categorical(
        "alpha", [1e-10 * 6 ** i for i in range(15)] + [1e-10 * 6 ** i / 2 for i in range(15)])}
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(data_dm_x)))):
        huberlinear = HuberRegressor(**param)
        huberlinear.fit(data_dm_x.loc[train_idx], data_raw_train.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(data_raw_train.loc[valid_idx, Target_n],
                                                  huberlinear.predict(data_dm_x.loc[valid_idx]))
        scores.append(accuracy)
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(huberlinear, f)
    return np.mean(scores)


# Lasso
def objective_lasso(trial, data_dm_x, data_raw_train, date_num):
    param = {"alpha": trial.suggest_float("alpha", 0.1, 10, step=0.01)}
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(data_dm_x)))):
        huberlinear = Lasso(**param)
        huberlinear.fit(data_dm_x.loc[train_idx], data_raw_train.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(data_raw_train.loc[valid_idx, Target_n],
                                                  huberlinear.predict(data_dm_x.loc[valid_idx]))
        scores.append(accuracy)
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(huberlinear, f)
    return np.mean(scores)


# Ridge
def objective_ridge(trial, data_dm_x, data_raw_train, date_num):
    param = {"alpha": trial.suggest_float("alpha", 0.1, 10, step=0.01)}
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(data_dm_x)))):
        huberlinear = Ridge(**param)
        huberlinear.fit(data_dm_x.loc[train_idx], data_raw_train.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(data_raw_train.loc[valid_idx, Target_n],
                                                  huberlinear.predict(data_dm_x.loc[valid_idx]))
        scores.append(accuracy)
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(huberlinear, f)
    return np.mean(scores)


# Enet
def objective_enet(trial, data_dm_x, data_raw_train, date_num):
    param = {"alpha": trial.suggest_categorical(
        "alpha", [1e-10 * 6 ** i for i in range(15)] + [1e-10 * 6 ** i / 2 for i in range(15)])}
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(data_dm_x)))):
        huberlinear = ElasticNet(**param)
        huberlinear.fit(data_dm_x.loc[train_idx], data_raw_train.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(data_raw_train.loc[valid_idx, Target_n],
                                                  huberlinear.predict(data_dm_x.loc[valid_idx]))
        scores.append(accuracy)
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(huberlinear, f)
    return np.mean(scores)


# Lasso Least Angle Regression
# "alpha": trial.suggest_categorical("alpha", [1e-10 * 6 ** i for i in range(15)] + [1e-10 * 6 ** i / 2 for i in range(15)])
def objective_llar(trial, data_dm_x, data_raw_train, date_num):
    param = {"alpha": trial.suggest_float("alpha", 0.1, 10, step=0.01)}
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(data_dm_x)))):
        huberlinear = LassoLars(**param)
        huberlinear.fit(data_dm_x.loc[train_idx], data_raw_train.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(data_raw_train.loc[valid_idx, Target_n],
                                                  huberlinear.predict(data_dm_x.loc[valid_idx]))
        scores.append(accuracy)
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(huberlinear, f)
    return np.mean(scores)


Allresults = []
for kk in ['llar', 'ridge', 'lasso', 'enet', 'hul', 'dct', 'xgb_sk', 'lgb_sk', 'catb', 'rf', 'etr', 'mlp']:
    number_trails = a_tr
    if kk in ['mlp', 'rf', 'etr']: number_trails = s_tr
    start_time = time.time()
    study_dct, best_dct = tuning_result('%s_%s_%s-%s-result' % (Dnu, Is_std, Target_n, kk), eval('objective_%s' % kk),
                                        number_trails, data_unseen)
    print("%s: --- %s seconds ---" % (kk, time.time() - start_time))
    Allresults.append([study_dct, best_dct, study_dct.best_value, time.time() - start_time])

with open(dir_path + r'Results\\best_record_%s_%s_%s.pkl' % (Dnu, Is_std, Target_n), 'wb') as h:
    pickle.dump(Allresults, h, protocol=pickle.HIGHEST_PROTOCOL)

# # Read those best models
# All_best_results = []
# for kk in ['llar', 'ridge', 'lasso', 'enet', 'hul', 'dct', 'xgb_sk', 'lgb_sk', 'catb', 'rf', 'etr', 'mlp']:
#     alltrails = [f for f in os.listdir('optuna_%s_%s_%s-%s-result' % (Dnu, Is_std, Target_n, kk)) if 'ty_' in f]
#     scores = [f.split('.pickle')[0] for f in alltrails]
#     scores = [np.float(f.split('-')[-1]) for f in scores]
#     with open('optuna_%s_%s_%s-%s-result/%s' % (Dnu, Is_std, Target_n, kk, alltrails[np.argmin(scores)]), 'rb') as h:
#         best_trail = pickle.load(h)
#     All_best_results.append([kk, scores[np.argmin(scores)], alltrails[np.argmin(scores)], best_trail])
#
# All_best_results = pd.DataFrame(All_best_results)
# All_best_results.columns = ['model', 'score', 'trail', 'meta']
# All_best_results = All_best_results.sort_values(by='score').reset_index(drop=True)
