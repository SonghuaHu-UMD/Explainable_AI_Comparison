# ##############
# Use origin data with transform: tree-based models have better performance
# ##############
import optuna
import glob
import os
import copy
import numpy as np
import pandas as pd
# import pickle5 as pickle
import pickle
import lightgbm as lgb
import catboost as catb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import HuberRegressor, Lasso, Ridge, ElasticNet, LassoLars, BayesianRidge
from pycaret.regression import *
import time
from sklearn import preprocessing

dir_path = r'C:\\Users\\songhua\\Cross_Nonlinear\\'
Dnu = '0630'
Target_n = 'Trip_Rate'
Is_std = 'st'
a_tr = 40
s_tr = 20


def mean_absolute_percentage_error(Validate, Predict):
    return np.mean(np.abs((Validate - Predict) / Validate))


####################################################
# Step 0: Prepare data
####################################################
# Read data
data = pd.read_pickle(dir_path + r'Data\data_origin_train.pkl')
data_unseen = pd.read_pickle(dir_path + r'Data\data_origin_test.pkl')

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

# Set train and test
data_train_x = data.drop([Target_n], axis=1)
data_train_y = data[[Target_n]]
data_test_x = data_unseen.drop([Target_n], axis=1)
data_test_y = data_unseen[[Target_n]]
# Normalise X
normalizer_x = preprocessing.StandardScaler()
normalized_train_x = pd.DataFrame(normalizer_x.fit_transform(data_train_x), columns=data_train_x.columns)
normalized_test_x = pd.DataFrame(normalizer_x.transform(data_test_x), columns=data_test_x.columns)
# Power transfer Y
normalizer_y = preprocessing.PowerTransformer(method='yeo-johnson', standardize=False)
normalized_train_y = pd.DataFrame(normalizer_y.fit_transform(data_train_y), columns=data_train_y.columns)
normalized_test_y = pd.DataFrame(normalizer_y.transform(data_test_y), columns=data_test_y.columns)

####################################################
# Step 1: Directly compare models using default parameters
####################################################
# Set environ
exp_reg = setup(data=data, target=Target_n, session_id=2, fold=5, normalize=True, transformation=False,
                transform_target=True, use_gpu=False, n_jobs=-1)

# # Compare all models
best_models_va = compare_models(exclude=['ransac', 'lar', 'ada', 'par', 'ard', 'tr', 'kr', 'svm', 'gbr', 'knn'],
                                sort='MAPE', n_select=15, turbo=False)
# Extract results
for kk in exp_reg:
    if type(kk) is list:
        if type(kk[0]) is pd.io.formats.style.Styler: model_metric = kk[1]  # pull( )

# Save models and results
model_metric.to_csv(dir_path + r'Results\\model_metric_%s_%s_%s_va.csv' % (Dnu, Is_std, Target_n))
with open(dir_path + r'Results\\best_model_%s_%s_%s_va.pkl' % (Dnu, Is_std, Target_n), 'wb') as h:
    pickle.dump(best_models_va, h, protocol=pickle.HIGHEST_PROTOCOL)


####################################################
# Step 2: Build the tuning process in pure optuna
####################################################
def tuning_result(date_num, objective_f, n_trials, normalized_train_x, normalized_train_y, normalized_test_x,
                  normalized_test_y):
    '''
    date_num = 'test-lgb_sk-result'
    objective_lgb = objective_lgb_sk
    n_trials = 2
    '''
    path_loc = os.path.join(os.getcwd(), "optuna_" + date_num)
    if not os.path.isdir(path_loc): os.makedirs(path_loc)
    # Build the study
    study_lgb = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="minimize")
    study_lgb.optimize(lambda trial: objective_f(trial, normalized_train_x, normalized_train_y, date_num),
                       n_trials=n_trials)
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
    with open(matching[0], "rb") as fh: best_model = pickle.load(fh)
    # with open(r'C:\Users\Administrator\PycharmProjects\Cross_Nonlinear_POI\optuna_1107_st-br-result\ty_3-nan.pickle', "rb") as fh: best_model = pickle.load(fh)
    # for key, value in trial.params.items(): print("    {}: {}".format(key, value))
    # Predict on testing
    print('MAPE: %s' % mean_absolute_percentage_error(
        normalizer_y.inverse_transform(normalized_test_y), normalizer_y.inverse_transform(
            best_model.predict(normalized_test_x).reshape(-1, 1))))
    return study_lgb, best_model


## Tune for lightgbm
def objective_lgb_sk(trial, normalized_train_x, normalized_train_y, date_num):
    param = {"objective": "regression", "metric": "mape", "verbosity": -1, "boosting_type": "gbdt",
             'feature_pre_filter': False,
             "n_estimators": trial.suggest_int("n_estimators", 100, 500, 50),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, step=0.01),
             "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
             "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
             "max_depth": trial.suggest_int("max_depth", 5, 100, step=5),
             "num_leaves": trial.suggest_int("num_leaves", 32, 512, step=32),  # num_leaf, max_leaves, max_leaf 512
             'min_sum_hessian_in_leaf': trial.suggest_int("min_sum_hessian_in_leaf", 0, 100),
             "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),  # sub_feature, colsample_bytree,
             }
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(normalized_train_x)))):
        # Add a callback for pruning
        # pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "mape")
        gbm = lgb.LGBMRegressor(**param)
        gbm.fit(normalized_train_x.loc[train_idx], normalized_train_y.loc[train_idx, Target_n],
                eval_set=[(normalized_train_x.loc[valid_idx], normalized_train_y.loc[valid_idx, Target_n])],
                eval_metric='mape', verbose=False)
        trial.set_user_attr(key="best_booster", value=copy.deepcopy(gbm))
        accuracy = mean_absolute_percentage_error(
            pd.DataFrame(normalizer_y.inverse_transform(normalized_train_y.loc[valid_idx, [Target_n]])), pd.DataFrame(
                normalizer_y.inverse_transform(gbm.predict(normalized_train_x.loc[valid_idx]).reshape(-1, 1))))
        scores.append(accuracy[0])
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(gbm, f)
    return np.mean(scores)


def objective_xgb_sk(trial, normalized_train_x, normalized_train_y, date_num):
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
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(normalized_train_x)))):
        # Add a callback for pruning
        # pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-mape")
        gbm = xgb.XGBRegressor(**param)
        gbm.fit(normalized_train_x.loc[train_idx], normalized_train_y.loc[train_idx, Target_n],
                eval_set=[(normalized_train_x.loc[valid_idx], normalized_train_y.loc[valid_idx, Target_n])],
                eval_metric='mape', verbose=False)
        trial.set_user_attr(key="best_booster", value=copy.deepcopy(gbm))
        accuracy = mean_absolute_percentage_error(
            pd.DataFrame(normalizer_y.inverse_transform(normalized_train_y.loc[valid_idx, [Target_n]])), pd.DataFrame(
                normalizer_y.inverse_transform(gbm.predict(normalized_train_x.loc[valid_idx]).reshape(-1, 1))))
        scores.append(accuracy[0])
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(gbm, f)
    return np.mean(scores)


## Tune for catboost
def objective_catb(trial, normalized_train_x, normalized_train_y, date_num):
    param = {"logging_level": 'Verbose', 'eval_metric': 'MAPE', "n_estimators": 1200,
             # "task_type": "GPU", 'devices': '0',
             "max_depth": trial.suggest_int("max_depth", 4, 12),
             "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
             "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.08, step=0.01), }
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(normalized_train_x)))):
        gbm = catb.CatBoostRegressor(**param)
        gbm.fit(normalized_train_x.loc[train_idx], normalized_train_y.loc[train_idx, Target_n],
                eval_set=[(normalized_train_x.loc[valid_idx], normalized_train_y.loc[valid_idx, Target_n])],
                verbose=0, early_stopping_rounds=100)
        accuracy = mean_absolute_percentage_error(
            pd.DataFrame(normalizer_y.inverse_transform(normalized_train_y.loc[valid_idx, [Target_n]])), pd.DataFrame(
                normalizer_y.inverse_transform(gbm.predict(normalized_train_x.loc[valid_idx]).reshape(-1, 1))))
        scores.append(accuracy[0])
        # # Save train test error
        # learn_error = pd.read_csv(".\\catboost_info\\learn_error.tsv", sep="\t")
        # learn_error.columns = ['iter', 'train_MAPE', 'train_RMSE']
        # test_error = pd.read_csv(".\\catboost_info\\test_error.tsv", sep="\t")
        # test_error.columns = ['iter', 'test_MAPE', 'test_RMSE']
        # learn_error = learn_error.merge(test_error, on='iter')
        # learn_error['trail'] = trial.number
        # learn_error['fold'] = fold_idx
        # learn_error.to_csv(".\\%s\\catb_train_test_%s_%s.csv" % ("optuna_" + date_num, trial.number, fold_idx))
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(gbm, f)
    return np.mean(scores)


# Random Forest Regressor
def objective_rf(trial, normalized_train_x, normalized_train_y, date_num):
    param = {"n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
             "max_depth": trial.suggest_int("max_depth", 10, 300, step=10),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, step=3),
             "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, step=1), }
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(normalized_train_x)))):
        gbm = RandomForestRegressor(**param, n_jobs=-1)
        gbm.fit(normalized_train_x.loc[train_idx], normalized_train_y.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(
            pd.DataFrame(normalizer_y.inverse_transform(normalized_train_y.loc[valid_idx, [Target_n]])), pd.DataFrame(
                normalizer_y.inverse_transform(gbm.predict(normalized_train_x.loc[valid_idx]).reshape(-1, 1))))
        scores.append(accuracy[0])

    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(gbm, f)
    return np.mean(scores)


# ExtraTreesRegressor
def objective_etr(trial, normalized_train_x, normalized_train_y, date_num):
    param = {"n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
             "max_depth": trial.suggest_int("max_depth", 10, 300, step=10),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, step=3),
             "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, step=1), }
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(normalized_train_x)))):
        gbm = ExtraTreesRegressor(**param, n_jobs=-1)
        gbm.fit(normalized_train_x.loc[train_idx], normalized_train_y.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(
            pd.DataFrame(normalizer_y.inverse_transform(normalized_train_y.loc[valid_idx, [Target_n]])), pd.DataFrame(
                normalizer_y.inverse_transform(gbm.predict(normalized_train_x.loc[valid_idx]).reshape(-1, 1))))
        scores.append(accuracy[0])

    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(gbm, f)
    return np.mean(scores)


# DecisionTreeRegressor
def objective_dct(trial, normalized_train_x, normalized_train_y, date_num):
    param = {"max_depth": trial.suggest_int("max_depth", 1, 30, step=2),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, step=2),
             "min_samples_split": trial.suggest_int("min_samples_split", 2, 20, step=2), }
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(normalized_train_x)))):
        gbm = DecisionTreeRegressor(**param)
        gbm.fit(normalized_train_x.loc[train_idx], normalized_train_y.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(
            pd.DataFrame(normalizer_y.inverse_transform(normalized_train_y.loc[valid_idx, [Target_n]])), pd.DataFrame(
                normalizer_y.inverse_transform(gbm.predict(normalized_train_x.loc[valid_idx]).reshape(-1, 1))))
        scores.append(accuracy[0])
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(gbm, f)
    return np.mean(scores)


# MLPRegressor
def objective_mlp(trial, normalized_train_x, normalized_train_y, date_num):
    param = {"max_iter": trial.suggest_int("max_iter", 100, 1000, step=100),
             "alpha": trial.suggest_float("alpha", 0.000001, 0.001, log=True),
             "hidden_layer_sizes": trial.suggest_categorical(
                 "hidden_layer_sizes", [(150,), (100,), (50,), (25,), (150, 100), (100, 50), (150, 100, 50)]), }
    # "activation": trial.suggest_categorical("activation", ['tanh', 'relu'])
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(normalized_train_x)))):
        mlpr = MLPRegressor(**param, learning_rate='constant', early_stopping=True, tol=0.001, n_iter_no_change=50)
        mlpr.fit(normalized_train_x.loc[train_idx], normalized_train_y.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(
            pd.DataFrame(normalizer_y.inverse_transform(normalized_train_y.loc[valid_idx, [Target_n]])), pd.DataFrame(
                normalizer_y.inverse_transform(mlpr.predict(normalized_train_x.loc[valid_idx]).reshape(-1, 1))))
        scores.append(accuracy[0])
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(mlpr, f)
    return np.mean(scores)


# HuberRegressor
def objective_hul(trial, normalized_train_x, normalized_train_y, date_num):
    param = {
        "alpha": trial.suggest_float("alpha", 1e6, 1e7, step=0.1 * 1e6)}
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(normalized_train_x)))):
        huberlinear = HuberRegressor(**param)
        huberlinear.fit(normalized_train_x.loc[train_idx], normalized_train_y.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(
            pd.DataFrame(normalizer_y.inverse_transform(normalized_train_y.loc[valid_idx, [Target_n]])), pd.DataFrame(
                normalizer_y.inverse_transform(huberlinear.predict(normalized_train_x.loc[valid_idx]).reshape(-1, 1))))
        scores.append(accuracy[0])
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(huberlinear, f)
    return np.mean(scores)


# Lasso
def objective_lasso(trial, normalized_train_x, normalized_train_y, date_num):
    param = {
        "alpha": trial.suggest_float("alpha", 0.01, 1, step=0.01), }
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(normalized_train_x)))):
        lassl = Lasso(**param)
        lassl.fit(normalized_train_x.loc[train_idx], normalized_train_y.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(
            pd.DataFrame(normalizer_y.inverse_transform(normalized_train_y.loc[valid_idx, [Target_n]])), pd.DataFrame(
                normalizer_y.inverse_transform(lassl.predict(normalized_train_x.loc[valid_idx]).reshape(-1, 1))))
        scores.append(accuracy[0])
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(lassl, f)
    return np.mean(scores)


# Ridge
def objective_ridge(trial, normalized_train_x, normalized_train_y, date_num):
    param = {
        "alpha": trial.suggest_float("alpha", 1e5, 1e7, step=1e5), }
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(normalized_train_x)))):
        ridl = Ridge(**param)
        ridl.fit(normalized_train_x.loc[train_idx], normalized_train_y.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(
            pd.DataFrame(normalizer_y.inverse_transform(normalized_train_y.loc[valid_idx, [Target_n]])), pd.DataFrame(
                normalizer_y.inverse_transform(ridl.predict(normalized_train_x.loc[valid_idx]).reshape(-1, 1))))
        scores.append(accuracy[0])
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(ridl, f)
    return np.mean(scores)


# Enet
def objective_enet(trial, normalized_train_x, normalized_train_y, date_num):
    param = {
        "alpha": trial.suggest_float("alpha", 0.05, 0.5, step=0.005)}
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(normalized_train_x)))):
        enet = ElasticNet(**param)
        enet.fit(normalized_train_x.loc[train_idx], normalized_train_y.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(
            pd.DataFrame(normalizer_y.inverse_transform(normalized_train_y.loc[valid_idx, [Target_n]])), pd.DataFrame(
                normalizer_y.inverse_transform(enet.predict(normalized_train_x.loc[valid_idx]).reshape(-1, 1))))
        scores.append(accuracy[0])
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(enet, f)
    return np.mean(scores)


# Lasso Least Angle Regression
def objective_llar(trial, normalized_train_x, normalized_train_y, date_num):
    param = {
        "alpha": trial.suggest_float("alpha", 0.0001, 0.0005, step=0.000005), }
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(normalized_train_x)))):
        llar = LassoLars(**param)
        llar.fit(normalized_train_x.loc[train_idx], normalized_train_y.loc[train_idx, Target_n])
        accuracy = mean_absolute_percentage_error(
            pd.DataFrame(normalizer_y.inverse_transform(normalized_train_y.loc[valid_idx, [Target_n]])), pd.DataFrame(
                normalizer_y.inverse_transform(llar.predict(normalized_train_x.loc[valid_idx]).reshape(-1, 1))))
        scores.append(accuracy[0])
    # Save a trained model
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "sr_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(scores, f)
    with open(os.path.join(os.getcwd(), "optuna_" + date_num, "ty_{}-{}.pickle".format(trial.number, np.mean(scores))),
              "wb") as f: pickle.dump(llar, f)
    return np.mean(scores)


Allresults = []
for kk in ['llar', 'enet', 'ridge', 'lasso', 'hul', 'dct', 'mlp', 'xgb_sk', 'lgb_sk', 'catb', 'rf', 'etr']:
    number_trails = a_tr
    if kk in ['mlp', 'rf', 'etr']: number_trails = s_tr
    start_time = time.time()
    study_dct, best_dct = \
        tuning_result('%s_%s_%s-%s-result' % (Dnu, Is_std, Target_n, kk), eval('objective_%s' % kk), number_trails,
                      normalized_train_x, normalized_train_y, normalized_test_x, normalized_test_y)
    print("%s: --- %s seconds ---" % (kk, time.time() - start_time))
    Allresults.append([study_dct, best_dct, study_dct.best_value, time.time() - start_time])

with open(dir_path + r'Results\\best_record_%s_%s_%s.pkl' % (Dnu, Is_std, Target_n), 'wb') as h:
    pickle.dump(Allresults, h, protocol=pickle.HIGHEST_PROTOCOL)

# # Read those best models
# All_best_results = []
# for kk in ['dct', 'br', 'llar', 'enet', 'ridge', 'lasso', 'hul', 'xgb_sk', 'lgb_sk', 'catb', 'rf', 'etr', 'mlp']:
#     alltrails = [f for f in os.listdir('optuna_%s_%s_%s-%s-result' % (Dnu, Is_std, Target_n, kk)) if 'ty_' in f]
#     scores = [f.split('.pickle')[0] for f in alltrails]
#     scores = [np.float(f.split('-')[-1]) for f in scores]
#     with open('optuna_%s_%s_%s-%s-result/%s' % (Dnu, Is_std, Target_n, kk, alltrails[np.argmin(scores)]), 'rb') as h:
#         best_trail = pickle.load(h)
#     All_best_results.append([kk, scores[np.argmin(scores)], alltrails[np.argmin(scores)], best_trail])
# All_best_results = pd.DataFrame(All_best_results)
# All_best_results.columns = ['model', 'score', 'trail', 'meta']
# All_best_results = All_best_results.sort_values(by='score').reset_index(drop=True)
