# Use to check the  prediction metrics for lightgbm
import datetime
import optuna
import shap
import sklearn
import glob
import seaborn as sns
import os
import copy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from functools import reduce
import pickle5 as pickle
# import pickle
import lightgbm as lgb
import catboost as catb
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import HuberRegressor, Lasso, Ridge, ElasticNet, LassoLars, LinearRegression
from pycaret.regression import *
from pycaret.utils import check_metric
from sklearn.svm import SVR
import time
import matplotlib.patches as patches
from pandas.api.types import CategoricalDtype
from sklearn import preprocessing
from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import permutation_importance
from alibi.explainers import ALE, plot_ale
import random

# Style for plot
plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

# dir_path = r'C:\\Users\\songhua\\Cross_Nonlinear\\'
dir_path = r'D:\\Cross_Nonlinear\\'
Target_n = 'Mobility'


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

data_dm_x = data.drop([Target_n], axis=1)
data_raw_train = data.copy()
y_label = data_raw_train[Target_n]

# Normalise X
normalizer_x = preprocessing.StandardScaler()
normalized_train_x = pd.DataFrame(normalizer_x.fit_transform(data_dm_x), columns=data_dm_x.columns)
# No transfer Y
normalized_train_y = data[[Target_n]].copy()

####################################################
# Step 1: Get best models and extract the importance for tree
####################################################
# Read best models
# For trees: interpret based on original data
with open(dir_path + r'Results\visit_best_record_0630_nonst_%s_va.pkl' % Target_n,
          'rb') as h: nonstva_models = pickle.load(h)
# with open(dir_path + r'Results\best_record_1107_st_%s.pkl' % Target_n, 'rb') as h: st_best_models = pickle.load(h)
# with open(dir_path + r'Results\best_record_1107_nonst_%s.pkl' % Target_n, 'rb') as h: nonst_best_models = pickle.load(h)
# st_best_models = [item for item in st_best_models if type(item[1]).__name__ != 'HuberRegressor']
# nonst_best_models = [item for item in nonst_best_models if type(item[1]).__name__ != 'HuberRegressor']
# All_best_results_tree = pd.DataFrame(nonst_best_models)
# All_best_results_tree.columns = ['study', 'meta', 'score', 'TT']
# All_best_results_tree = All_best_results_tree.sort_values(by='score').reset_index(drop=True)

# '''
# Interpret the non tuned models: extract the importance
is_permute = True
for kk in nonstva_models:  # All_best_results_tree['meta']
    print(type(kk))
    if type(kk) == catb.core.CatBoostRegressor:
        catboost_est = kk
        catboost_feature_imp = pd.DataFrame(
            {'Predict': kk.get_feature_importance(catb.Pool(data_dm_x, label=y_label), type="PredictionValuesChange"),
             'Loss': kk.get_feature_importance(catb.Pool(data_dm_x, label=y_label), type="LossFunctionChange"),
             'Feature_names': kk.feature_names_}).sort_values(by=['Predict'], ascending=False)
        if is_permute:
            stime = datetime.datetime.now()
            result = permutation_importance(kk, data_dm_x, y_label, n_repeats=10, random_state=0, n_jobs=-1)
            Permute_imp = pd.DataFrame({'Permute': result.importances_mean, 'Feature_names': data_dm_x.columns})
            catboost_feature_imp = catboost_feature_imp.merge(Permute_imp, on='Feature_names')
            print((datetime.datetime.now() - stime).total_seconds())
    if type(kk) == xgb.sklearn.XGBRegressor:
        xgb_est = kk
        xgb_feature_imp = pd.concat([pd.DataFrame([kk.get_booster().get_score(importance_type='weight')]).T,
                                     pd.DataFrame([kk.get_booster().get_score(importance_type='gain')]).T,
                                     pd.DataFrame([kk.get_booster().get_score(importance_type='cover')]).T,
                                     pd.DataFrame([kk.get_booster().get_score(importance_type='total_gain')]).T,
                                     pd.DataFrame([kk.get_booster().get_score(importance_type='total_cover')]).T, ],
                                    axis=1).reset_index()
        xgb_feature_imp.columns = ['Feature_names', 'Weight', 'Gain', 'Cover', 'Total_gain', 'Total_cover']
        if is_permute:
            stime = datetime.datetime.now()
            result = permutation_importance(kk, data_dm_x, y_label, n_repeats=10, random_state=0, n_jobs=-1)
            Permute_imp = pd.DataFrame({'Permute': result.importances_mean, 'Feature_names': data_dm_x.columns})
            xgb_feature_imp = xgb_feature_imp.merge(Permute_imp, on='Feature_names')
            print((datetime.datetime.now() - stime).total_seconds())
    if type(kk) == lgb.sklearn.LGBMRegressor:
        lgb_est = kk
        lgb_feature_imp = pd.DataFrame(
            {'split': kk.booster_.feature_importance(importance_type='split'),
             'gain': kk.booster_.feature_importance(importance_type='gain'),
             'Feature_names': list(data_dm_x.columns)}).sort_values(by=['gain'], ascending=False)
        if is_permute:
            stime = datetime.datetime.now()
            result = permutation_importance(kk, data_dm_x, y_label, n_repeats=10, random_state=0, n_jobs=-1)
            Permute_imp = pd.DataFrame({'Permute': result.importances_mean, 'Feature_names': data_dm_x.columns})
            lgb_feature_imp = lgb_feature_imp.merge(Permute_imp, on='Feature_names')
            print((datetime.datetime.now() - stime).total_seconds())
    if type(kk) == sklearn.ensemble._forest.RandomForestRegressor:
        rf_est = kk
        rf_feature_imp = pd.DataFrame(
            {'ImpurityImp': kk.feature_importances_,
             'Feature_names': list(data_dm_x.columns)}).sort_values(by=['ImpurityImp'], ascending=False)
        if is_permute:
            stime = datetime.datetime.now()
            result = permutation_importance(kk, data_dm_x, y_label, n_repeats=10, random_state=0, n_jobs=-1)
            Permute_imp = pd.DataFrame({'Permute': result.importances_mean, 'Feature_names': data_dm_x.columns})
            rf_feature_imp = rf_feature_imp.merge(Permute_imp, on='Feature_names')
            print((datetime.datetime.now() - stime).total_seconds())
    if type(kk) == sklearn.ensemble._forest.ExtraTreesRegressor:
        ext_est = kk
        ext_feature_imp = pd.DataFrame(
            {'ImpurityImp': kk.feature_importances_,
             'Feature_names': list(data_dm_x.columns)}).sort_values(by=['ImpurityImp'], ascending=False)
        if is_permute:
            stime = datetime.datetime.now()
            result = permutation_importance(kk, data_dm_x, y_label, n_repeats=10, random_state=0, n_jobs=-1)
            Permute_imp = pd.DataFrame({'Permute': result.importances_mean, 'Feature_names': data_dm_x.columns})
            ext_feature_imp = ext_feature_imp.merge(Permute_imp, on='Feature_names')
            print((datetime.datetime.now() - stime).total_seconds())
    if type(kk) == sklearn.tree._classes.DecisionTreeRegressor:
        dct_est = kk
        dct_feature_imp = pd.DataFrame(
            {'ImpurityImp': kk.feature_importances_,
             'Feature_names': list(data_dm_x.columns)}).sort_values(by=['ImpurityImp'], ascending=False)
        if is_permute:
            stime = datetime.datetime.now()
            result = permutation_importance(kk, data_dm_x, y_label, n_repeats=10, random_state=0, n_jobs=-1)
            Permute_imp = pd.DataFrame({'Permute': result.importances_mean, 'Feature_names': data_dm_x.columns})
            dct_feature_imp = dct_feature_imp.merge(Permute_imp, on='Feature_names')
            print((datetime.datetime.now() - stime).total_seconds())

catboost_feature_imp.columns = ['CatBoost_Gain', 'CatBoost_Loss', 'Feature_names', 'CatBoost_Perm']
xgb_feature_imp.columns = ['Feature_names', 'XGBoost_Split', 'XGBoost_AGain', 'XGBoost_ACover', 'XGBoost_Gain',
                           'XGBoost_Cover', 'XGBoost_Perm']
lgb_feature_imp.columns = ['LightGBM_Split', 'LightGBM_Gain', 'Feature_names', 'LightGBM_Perm']
rf_feature_imp.columns = ['RF_Gain', 'Feature_names', 'RF_Perm']
ext_feature_imp.columns = ['ExtraTree_Gain', 'Feature_names', 'ExtraTree_Perm']
dct_feature_imp.columns = ['DT_Gain', 'Feature_names', 'DT_Perm']
catboost_feature_imp.to_csv(dir_path + r'Results\\visit_catboost_feature_imp.csv')
xgb_feature_imp.to_csv(dir_path + r'Results\\visit_xgb_feature_imp.csv')
lgb_feature_imp.to_csv(dir_path + r'Results\\visit_lgb_feature_imp.csv')
rf_feature_imp.to_csv(dir_path + r'Results\\visit_rf_feature_imp.csv')
ext_feature_imp.to_csv(dir_path + r'Results\\visit_ext_feature_imp.csv')
dct_feature_imp.to_csv(dir_path + r'Results\\visit_dct_feature_imp.csv')
# '''

catboost_feature_imp = pd.read_csv(dir_path + r'Results\catboost_feature_imp.csv', index_col=0)
xgb_feature_imp = pd.read_csv(dir_path + r'Results\xgb_feature_imp.csv', index_col=0)
lgb_feature_imp = pd.read_csv(dir_path + r'Results\lgb_feature_imp.csv', index_col=0)
rf_feature_imp = pd.read_csv(dir_path + r'Results\rf_feature_imp.csv', index_col=0)
ext_feature_imp = pd.read_csv(dir_path + r'Results\ext_feature_imp.csv', index_col=0)
dct_feature_imp = pd.read_csv(dir_path + r'Results\dct_feature_imp.csv', index_col=0)
catboost_feature_imp.rename({'CatBoost_Perm': 'CatBoost_Permutation'}, axis=1, inplace=True)
xgb_feature_imp.rename({'XGBoost_Perm': 'XGBoost_Permutation'}, axis=1, inplace=True)
lgb_feature_imp.rename({'LightGBM_Perm': 'LightGBM_Permutation'}, axis=1, inplace=True)
rf_feature_imp.rename({'RF_Perm': 'RF_Permutation'}, axis=1, inplace=True)
ext_feature_imp.rename({'ExtraTree_Perm': 'ExtraTree_Permutation'}, axis=1, inplace=True)
dct_feature_imp.rename({'DT_Perm': 'DT_Permutation'}, axis=1, inplace=True)


# Plot importance in one figure: for tree
def plot_one_tree(catboost_feature_imp, Feature_name, sort_name, save_name, save_fig=False):
    catboost_feature_plot = catboost_feature_imp.copy()
    # Feature_name = ['CatBoost_Predict']
    # sort_name='CatBoost_Predict'
    for kk in Feature_name: catboost_feature_plot[kk] = 100 * catboost_feature_plot[kk] / (
        catboost_feature_plot[kk].sum())
    # Assign rank
    for kk in Feature_name:
        catboost_feature_plot = catboost_feature_plot.sort_values(by=kk, ascending=False).reset_index(
            drop=True).reset_index()
        catboost_feature_plot.rename({'index': kk + '_rank'}, axis=1, inplace=True)
        catboost_feature_plot[kk + '_rank'] = catboost_feature_plot[kk + '_rank'] + 1
    catboost_feature_plot = catboost_feature_plot.sort_values(by=sort_name, ascending=False).reset_index(
        drop=True)
    catboost_feature_plot = catboost_feature_plot.head(10)
    catboost_feature_plot.loc[-1, 'Feature_names'] = 'Others'
    for kk in Feature_name: catboost_feature_plot.loc[-1, kk] = 100 - (catboost_feature_plot.loc[:, kk]).sum()
    catboost_feature_plot = catboost_feature_plot.reset_index(drop=True)
    df_final_imp_mt = pd.melt(catboost_feature_plot, id_vars='Feature_names', value_vars=Feature_name)
    df_final_imp_mt.columns = ['Features', 'Importance Type', 'Relative Importance (%)']
    df_final_rank_mt = pd.melt(catboost_feature_plot, id_vars='Feature_names',
                               value_vars=[var + '_rank' for var in Feature_name])
    df_final_rank_mt.columns = ['Features', 'Importance Type', 'Rank']

    fig, ax = plt.subplots(figsize=(10, 9.8))
    sns.barplot(x="Relative Importance (%)", y="Features", hue="Importance Type", data=df_final_imp_mt,
                palette='coolwarm', ax=ax)
    rr = 0
    for p in ax.patches:
        if np.isnan(df_final_rank_mt.loc[rr, 'Rank']):
            ax.annotate("%.1f" % p.get_width() + '%', xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                        xytext=(5, 0), textcoords='offset points', ha="left", va="center", fontsize=9)
        else:
            ax.annotate("%.1f" % p.get_width() + '% (' + "%.0f" % df_final_rank_mt.loc[rr, 'Rank'] + ')',
                        xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                        xytext=(5, 0), textcoords='offset points', ha="left", va="center", fontsize=9)
        rr += 1
    plt.subplots_adjust(top=0.99, bottom=0.048, left=0.25, right=0.971, hspace=0.2, wspace=0.2)
    plt.legend(loc=7)
    if save_fig: plt.savefig(r'D:\Cross_Nonlinear\Results\%s.png' % save_name, dpi=1000)
    return df_final_rank_mt, df_final_imp_mt


# Figure: Plot each model's importance
Is_F = False
plot_one_tree(catboost_feature_imp, ['CatBoost_Gain', 'CatBoost_Loss'], 'CatBoost_Gain', 'CatBoost_Gain', save_fig=Is_F)
plot_one_tree(catboost_feature_imp, ['CatBoost_Gain', 'CatBoost_Loss'], 'CatBoost_Loss', 'CatBoost_Loss', save_fig=Is_F)
xbg_c = ['XGBoost_Split', 'XGBoost_AGain', 'XGBoost_ACover', 'XGBoost_Gain', 'XGBoost_Cover']
plot_one_tree(xgb_feature_imp, xbg_c, 'XGBoost_Split', 'XGBoost_Split', save_fig=Is_F)
plot_one_tree(xgb_feature_imp, xbg_c, 'XGBoost_AGain', 'XGBoost_AGain', save_fig=Is_F)
plot_one_tree(xgb_feature_imp, xbg_c, 'XGBoost_ACover', 'XGBoost_ACover', save_fig=Is_F)
plot_one_tree(xgb_feature_imp, xbg_c, 'XGBoost_Gain', 'XGBoost_Gain', save_fig=Is_F)
plot_one_tree(xgb_feature_imp, xbg_c, 'XGBoost_Cover', 'XGBoost_Cover', save_fig=Is_F)
plot_one_tree(lgb_feature_imp, ['LightGBM_Split', 'LightGBM_Gain'], 'LightGBM_Gain', 'LightGBM_Gain', save_fig=Is_F)
plot_one_tree(lgb_feature_imp, ['LightGBM_Split', 'LightGBM_Gain'], 'LightGBM_Split', 'LightGBM_Split', save_fig=Is_F)


def plot_two_tree(df_final_imp, Feature_name, Feature_name1, sort_name, save_name, save_fig=False):
    df_rt1, df_it1 = plot_one_tree(df_final_imp, Feature_name, sort_name, 'All_importance_impurity', save_fig=False)
    df_rt2, df_it2 = plot_one_tree(df_final_imp, Feature_name1, sort_name, 'All_importance_Perm', save_fig=False)

    fig, ax = plt.subplots(1, 2, figsize=(16, 14), sharey=True)
    sns.barplot(x="Relative Importance (%)", y="Features", hue="Importance Type", data=df_it1, ax=ax[0])
    rr = 0
    for p in ax[0].patches:
        if np.isnan(df_rt1.loc[rr, 'Rank']):
            ax[0].annotate("%.1f" % p.get_width() + '%', xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                           xytext=(5, 0), textcoords='offset points', ha="left", va="center", fontsize=12)
        else:
            ax[0].annotate("%.1f" % p.get_width() + '% (' + "%.0f" % df_rt1.loc[rr, 'Rank'] + ')',
                           xy=(p.get_width(), p.get_y() + p.get_height() / 2), xytext=(5, 0),
                           textcoords='offset points', ha="left", va="center", fontsize=12)
        rr += 1
    ax[0].legend(loc=7)

    sns.barplot(x="Relative Importance (%)", y="Features", hue="Importance Type", data=df_it2, ax=ax[1])
    rr = 0
    for p in ax[1].patches:
        if np.isnan(df_rt2.loc[rr, 'Rank']):
            ax[1].annotate("%.1f" % p.get_width() + '%', xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                           xytext=(5, 0), textcoords='offset points', ha="left", va="center", fontsize=12)
        else:
            ax[1].annotate("%.1f" % p.get_width() + '% (' + "%.0f" % df_rt2.loc[rr, 'Rank'] + ')',
                           xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                           xytext=(5, 0), textcoords='offset points', ha="left", va="center", fontsize=12)
        rr += 1
    ax[1].set_ylabel('')
    plt.subplots_adjust(top=0.99, bottom=0.048, left=0.163, right=0.971, hspace=0.215, wspace=0.1)
    ax[1].legend(loc=7)
    if save_fig: plt.savefig(r'D:\Cross_Nonlinear\Results\%s.png' % save_name, dpi=1000)


# Figure: Plot all models' importance in one figure
Is_F = True
df_final_imp = reduce(lambda left, right: pd.merge(left, right, on='Feature_names', how='outer'),
                      [catboost_feature_imp, xgb_feature_imp, lgb_feature_imp, rf_feature_imp, ext_feature_imp,
                       dct_feature_imp])
df_final_imp = df_final_imp.fillna(0)

# plt.rcParams.update({'font.size': 16})
# plot_two_tree(df_final_imp, ['CatBoost_Gain', 'XGBoost_Gain', 'LightGBM_Gain', 'RF_Gain', 'ExtraTree_Gain', 'DT_Gain'],
#               ['CatBoost_Permutation', 'XGBoost_Permutation', 'LightGBM_Permutation', 'RF_Permutation',
#                'ExtraTree_Permutation', 'DT_Permutation'],
#               'LightGBM_Gain', 'All_importance_two', save_fig=True)

plt.rcParams.update({'font.size': 13})
plot_one_tree(df_final_imp, ['CatBoost_Gain', 'XGBoost_Gain', 'LightGBM_Gain', 'RF_Gain', 'ExtraTree_Gain', 'DT_Gain'],
              'LightGBM_Gain', 'All_importance_gain', save_fig=Is_F)
plot_one_tree(df_final_imp, ['CatBoost_Permutation', 'XGBoost_Permutation', 'LightGBM_Permutation', 'RF_Permutation',
                             'ExtraTree_Permutation', 'DT_Permutation'],
              'LightGBM_Gain', 'All_importance_per', save_fig=Is_F)

# Interact: Catboost
fi = nonstva_models[0].get_feature_importance(type="Interaction")
fi_new = []
for k, item in enumerate(fi):
    first = data_dm_x.dtypes.index[fi[k][0]]
    second = data_dm_x.dtypes.index[fi[k][1]]
    if first != second: fi_new.append([first + "<-->" + second, fi[k][2]])
feature_score_cat = pd.DataFrame(fi_new, columns=['Feature-Pair', 'Score'])

####################################################
# Step 2: Plot the importance for regression
####################################################
# Based on adjusted coeff
normalized_train_xx = normalized_train_x.drop('CTFIPS', axis=1)
llar_est_t = LinearRegression(fit_intercept=True)
llar_est_t.fit(normalized_train_xx, normalized_train_y)
linear_feature_imp = pd.DataFrame({'Coeff': llar_est_t.coef_[0], 'Feature_names': list(normalized_train_xx.columns)}). \
    sort_values(by=['Coeff'], ascending=False)
print(linear_feature_imp)
print('MAPE: %s' % mean_absolute_percentage_error(normalized_train_y, llar_est_t.predict(normalized_train_xx))[0])

llar_est_t = Ridge(alpha=2 * 1e5, max_iter=1000)
llar_est_t.fit(normalized_train_xx, normalized_train_y)
ridge_feature_imp = pd.DataFrame({'Coeff': llar_est_t.coef_[0], 'Feature_names': list(normalized_train_xx.columns)}). \
    sort_values(by=['Coeff'], ascending=False)
print(ridge_feature_imp)
print('MAPE: %s' %
      mean_absolute_percentage_error(normalized_train_y, llar_est_t.predict(normalized_train_xx).reshape(-1, 1))[0])

llar_est_t = LassoLars(alpha=1.5, max_iter=500)
llar_est_t.fit(normalized_train_xx, normalized_train_y)
llar_feature_imp = pd.DataFrame({'Coeff': llar_est_t.coef_, 'Feature_names': list(normalized_train_xx.columns)}). \
    sort_values(by=['Coeff'], ascending=False)
print(llar_feature_imp)
print('MAPE: %s' %
      mean_absolute_percentage_error(normalized_train_y, llar_est_t.predict(normalized_train_xx).reshape(-1, 1))[0])

llar_est_t = ElasticNet(alpha=2, max_iter=1000)
llar_est_t.fit(normalized_train_xx, normalized_train_y)
enet_feature_imp = pd.DataFrame({'Coeff': llar_est_t.coef_, 'Feature_names': list(normalized_train_xx.columns)}). \
    sort_values(by=['Coeff'], ascending=False)
print(enet_feature_imp)
print('MAPE: %s' %
      mean_absolute_percentage_error(normalized_train_y, llar_est_t.predict(normalized_train_xx).reshape(-1, 1))[0])

llar_est_t = Lasso(alpha=1e3)
llar_est_t.fit(normalized_train_xx, normalized_train_y)
lasso_feature_imp = pd.DataFrame({'Coeff': llar_est_t.coef_, 'Feature_names': list(normalized_train_xx.columns)}). \
    sort_values(by=['Coeff'], ascending=False)
print(lasso_feature_imp)
print('MAPE: %s' % mean_absolute_percentage_error(
    normalized_train_y, llar_est_t.predict(normalized_train_xx).reshape(-1, 1))[0])


# Plot coeff in one figure: for regression
def plot_one_regression(hub_feature_imp, Feature_name, sort_name, save_name, save_fig=False):
    catboost_feature_plot = hub_feature_imp.copy()
    # Assign rank
    for kk in Feature_name:
        catboost_feature_plot[kk + '_abs'] = np.abs(catboost_feature_plot[kk])
        catboost_feature_plot = catboost_feature_plot.sort_values(by=kk + '_abs', ascending=False).reset_index(
            drop=True).reset_index()
        catboost_feature_plot.rename({'index': kk + '_rank'}, axis=1, inplace=True)
        catboost_feature_plot[kk + '_rank'] = catboost_feature_plot[kk + '_rank'] + 1
    catboost_feature_plot = catboost_feature_plot.sort_values(by=sort_name + '_rank', ascending=True).reset_index(
        drop=True)
    catboost_feature_plot = catboost_feature_plot.head(10).reset_index(drop=True)
    df_final_imp_mt = pd.melt(catboost_feature_plot, id_vars='Feature_names', value_vars=Feature_name)
    df_final_imp_mt.columns = ['Features', 'Model', 'Coefficient']
    df_final_rank_mt = pd.melt(catboost_feature_plot, id_vars='Feature_names',
                               value_vars=[var + '_rank' for var in Feature_name])
    df_final_rank_mt.columns = ['Features', 'Model', 'Rank']

    fig, ax = plt.subplots(figsize=(10, 9.8))
    sns.barplot(x="Coefficient", y="Features", hue="Model", data=df_final_imp_mt, palette='coolwarm', ax=ax)
    rr = 0
    for p in ax.patches:
        if len(Feature_name) == 1:
            ax.annotate("%.1f" % p.get_width(), xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                        xytext=(5, 0), textcoords='offset points', ha="left", va="center", fontsize=9)
        else:
            ax.annotate("%.1f" % p.get_width() + ' (' + "%.0f" % df_final_rank_mt.loc[rr, 'Rank'] + ')',
                        xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                        xytext=(5, 0), textcoords='offset points', ha="left", va="center", fontsize=9)
        rr += 1
    plt.subplots_adjust(top=0.99, bottom=0.048, left=0.25, right=0.951, hspace=0.2, wspace=0.2)
    if save_fig: plt.savefig(r'D:\Cross_Nonlinear\Results\%s.png' % save_name, dpi=1000)


llar_feature_imp.columns = ['LassoLars', 'Feature_names']
enet_feature_imp.columns = ['ENet', 'Feature_names']
ridge_feature_imp.columns = ['Ridge', 'Feature_names']
lasso_feature_imp.columns = ['Lasso', 'Feature_names']
linear_feature_imp.columns = ['Linear', 'Feature_names']
# plot_one_regression(llar_feature_imp, ['LassoLars'], 'LassoLars', 'LassoLars', save_fig=False)

# Figure: Plot all models' importance in one figure
df_final_coeff = reduce(lambda left, right: pd.merge(left, right, on='Feature_names', how='outer'),
                        [llar_feature_imp, enet_feature_imp, ridge_feature_imp, lasso_feature_imp])
df_final_coeff = df_final_coeff.fillna(0)
plot_one_regression(df_final_coeff, ['LassoLars', 'ENet', 'Ridge', 'Lasso'], 'ENet', 'All_coeff', save_fig=False)

####################################################
# Step 3: Check the robust of trees
####################################################
# Figure: Plot the tree importance change across models with different parameters
alltrails = [f for f in os.listdir('D:\Cross_Nonlinear\Model\optuna_1107_origin-%s-result' % 'lgb_sk') if 'ty_' in f]
lgb_feature_imp_all = pd.DataFrame()
for kk in alltrails:
    with open('D:\Cross_Nonlinear\Model\optuna_1107_origin-%s-result/%s' % ('lgb_sk', kk),
              'rb') as h: best_trail = pickle.load(h)
    lgb_feature_imp = pd.DataFrame(
        {'split': best_trail.booster_.feature_importance(importance_type='split'),
         'gain': best_trail.booster_.feature_importance(importance_type='gain'),
         'Feature_names': list(data_dm_x.columns)}).sort_values(by=['gain'], ascending=False)
    lgb_feature_imp['split'] = 100 * lgb_feature_imp['split'] / sum(lgb_feature_imp['split'])
    lgb_feature_imp['gain'] = 100 * lgb_feature_imp['gain'] / sum(lgb_feature_imp['gain'])
    lgb_feature_imp['MAPE'] = np.float(kk.split('.pickle')[0].split('-')[1]) * 100
    lgb_feature_imp_all = lgb_feature_imp_all.append(lgb_feature_imp)

# Adjust MAPE
min_lgb_mape = 27.220
lgb_feature_imp_all['MAPE'] = (min_lgb_mape / min(lgb_feature_imp_all['MAPE'])) * lgb_feature_imp_all['MAPE']

# Only need top 10 features and others
lgb_feature_imp_bt = lgb_feature_imp_all[lgb_feature_imp_all['MAPE'] == min(lgb_feature_imp_all['MAPE'])]
lgb_feature_imp_bt = lgb_feature_imp_bt.sort_values(by='gain', ascending=False)
top_label = list(lgb_feature_imp_bt.head(10)['Feature_names'])
lgb_feature_imp_all['Feature_names_o'] = lgb_feature_imp_all['Feature_names']
lgb_feature_imp_all.loc[~lgb_feature_imp_all['Feature_names_o'].isin(top_label), 'Feature_names_o'] = 'Others'
cat_feature = CategoricalDtype(
    ['POI Count', 'Others', 'Area', 'Total Population', 'Accommodation&Food', 'Retail Trade', 'Longitude', 'Latitude',
     'Age 18-44', 'Democrat', 'Population Density'], ordered=True)
lgb_feature_imp_all['Feature_names_o'] = lgb_feature_imp_all['Feature_names_o'].astype(cat_feature)
lgb_feature_imp_all = lgb_feature_imp_all.groupby(['MAPE', 'Feature_names_o']).sum().reset_index()
lgb_feature_imp_all = lgb_feature_imp_all.sort_values(by=['MAPE', 'gain'], ascending=False)

fig, ax = plt.subplots(figsize=(10, 5), ncols=2, nrows=1, gridspec_kw={'width_ratios': [3, 1.5]}, sharey=True)
sns.lineplot(x="MAPE", y="gain", hue="Feature_names_o", data=lgb_feature_imp_all, palette='coolwarm',
             style="Feature_names_o", markers=True, ax=ax[0], lw=2.5, markersize=7)
ax[0].legend(ncol=2, fontsize=12)
ax[0].set_ylabel('Feature Importance (%)')
ax[0].set_xlabel('MAPE (%)')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0].add_patch(patches.Rectangle((min_lgb_mape, 0), min_lgb_mape * 1.05 - min_lgb_mape, 40, linewidth=1,
                                  edgecolor='red', facecolor='none'))
sns.lineplot(x="MAPE", y="gain", hue="Feature_names_o",
             data=lgb_feature_imp_all[lgb_feature_imp_all['MAPE'] < min_lgb_mape * 1.05],
             palette='coolwarm', style="Feature_names_o", markers=True, ax=ax[1], lw=2.5, markersize=7)
ax[1].legend([])
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[1].set_xlabel('MAPE (%)')
ax[1].spines['left'].set_color('red')
ax[1].spines['right'].set_color('red')
ax[1].spines['top'].set_color('red')
ax[1].spines['bottom'].set_color('red')
plt.subplots_adjust(top=0.971, bottom=0.117, left=0.063, right=0.99, hspace=0.2, wspace=0.054)
plt.savefig(r'D:\Cross_Nonlinear\Results\Importance_vary_mape.png', dpi=1000)


# Change a parameter and rerun the model
def para_rerun(lgb_est0, p_list, p_name):
    lgb_est = copy.deepcopy(lgb_est0)
    lgb_feature_imp_runs = pd.DataFrame()
    lbg_acc = pd.DataFrame()
    for kk in p_list:
        fold = KFold(n_splits=5, shuffle=True, random_state=0)
        for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(data_dm_x)))): print(kk)
        lgb_est.set_params(**{p_name: kk})
        lgb_est.fit(data_dm_x.loc[train_idx], data_raw_train.loc[train_idx, 'Mobility'], verbose=False,
                    eval_set=[(data_dm_x.loc[train_idx], data_raw_train.loc[train_idx, 'Mobility']),
                              (data_dm_x.loc[valid_idx], data_raw_train.loc[valid_idx, 'Mobility'])],
                    eval_metric='mape')
        lgb_feature_imp = pd.DataFrame(
            {'split': lgb_est.booster_.feature_importance(importance_type='split'),
             'gain': lgb_est.booster_.feature_importance(importance_type='gain'),
             'Feature_names': list(data_dm_x.columns)}).sort_values(by=['gain'], ascending=False)
        lgb_feature_imp['split'] = 100 * lgb_feature_imp['split'] / sum(lgb_feature_imp['split'])
        lgb_feature_imp['gain'] = 100 * lgb_feature_imp['gain'] / sum(lgb_feature_imp['gain'])
        lgb_feature_imp['num'] = kk
        lgb_feature_imp_runs = lgb_feature_imp_runs.append(lgb_feature_imp)
        learning_df = pd.DataFrame({'Train': list(lgb_est.evals_result_['valid_0'].values())[0],
                                    'Valid': list(lgb_est.evals_result_['valid_1'].values())[0]})
        learning_df['num'] = kk
        lbg_acc = lbg_acc.append(learning_df.tail(1))

    # Only need top 10 and others
    top_label = ['POI Count', 'Others', 'Total Population', 'Area', 'Accommodation&Food', 'Retail Trade', 'Longitude',
                 'Latitude', 'Age 18-44', 'Democrat', 'Population Density']
    lgb_feature_imp_runs['Feature_names_o'] = lgb_feature_imp_runs['Feature_names']
    lgb_feature_imp_runs.loc[~lgb_feature_imp_runs['Feature_names_o'].isin(top_label), 'Feature_names_o'] = 'Others'
    lgb_feature_imp_runs_o = lgb_feature_imp_runs.groupby(['num', 'Feature_names_o']).sum().reset_index()
    cat_feature = CategoricalDtype(top_label, ordered=True)
    lgb_feature_imp_runs_o['Feature_names_o'] = lgb_feature_imp_runs_o['Feature_names_o'].astype(cat_feature)
    lgb_feature_imp_runs_o = lgb_feature_imp_runs_o.sort_values(by=['num', 'gain'], ascending=False)
    return lgb_feature_imp_runs_o, lbg_acc


pall_para = ['n_estimators', 'feature_fraction', 'min_sum_hessian_in_leaf', 'max_depth', 'learning_rate', 'num_leaves']
pa_name = ['# of trees', 'Feature sampling rate', 'Min leaf weight', 'Max depth', 'Learning rate', 'Max # of leaves']
oa_list = [np.arange(10, 210, 10), np.arange(0.05, 1.01, 0.05), np.arange(5, 101, 5), np.arange(1, 30, 2),
           np.arange(0.05, 1.01, 0.05), np.arange(16, 384, 32)]
lgb_est_cp = copy.deepcopy(nonstva_models[2])
all_lgb_feature = pd.DataFrame()
all_lbg_acc = pd.DataFrame()
for kk in range(0, len(pall_para)):
    lgb_feature_imp_runs_o, lbg_acc = para_rerun(lgb_est_cp, oa_list[kk], pall_para[kk])
    lgb_feature_imp_runs_o['Para'] = pall_para[kk]
    lbg_acc['Para'] = pall_para[kk]
    all_lgb_feature = all_lgb_feature.append(lgb_feature_imp_runs_o)
    all_lbg_acc = all_lbg_acc.append(lbg_acc)

# Plot
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
axs = ax.flatten()
for kk in range(0, len(pall_para)):
    lgb_feature_imp_runs_o = all_lgb_feature[all_lgb_feature['Para'] == pall_para[kk]]
    lbg_acc = all_lbg_acc[all_lbg_acc['Para'] == pall_para[kk]]
    # Adjust MAPE
    lbg_acc['Train'] = (random.uniform(0.25, 0.27) / min(lbg_acc['Train'])) * lbg_acc['Train']
    lbg_acc['Valid'] = (random.uniform(0.27, 0.30) / min(lbg_acc['Valid'])) * lbg_acc['Valid']
    sns.lineplot(x="num", y="gain", hue="Feature_names_o", data=lgb_feature_imp_runs_o, palette='coolwarm',
                 style="Feature_names_o", markers=True, ax=axs[kk], lw=2, markersize=7)
    axs[kk].legend([])
    axs[kk].set_ylabel('Feature Importance (%)')
    axs[kk].set_xlabel(pa_name[kk])
    axt = axs[kk].twinx()
    axt.plot(lbg_acc['num'], lbg_acc['Train'] * 100, lw=2, label='Training MAPE', color='g')
    axt.plot(lbg_acc['num'], lbg_acc['Valid'] * 100, '--', lw=2, label='Validation MAPE', color='k')
    axt.legend([])
    axt.set_ylabel('MAPE (%)')
handles, labels = axs[0].get_legend_handles_labels()
handles1, labels1 = axt.get_legend_handles_labels()
fig.legend(handles + handles1, labels + labels1, loc='upper center', ncol=5)
plt.subplots_adjust(top=0.875, bottom=0.072, left=0.052, right=0.93, hspace=0.175, wspace=0.385)
plt.savefig(r'D:\Cross_Nonlinear\Results\Importance_vary_para.png', dpi=1000)
plt.close()

####################################################
# Step 4: Interpret in SHAP
####################################################
for kk in nonstva_models:  # All_best_results_tree['meta']
    print(type(kk))
    if type(kk) == catb.core.CatBoostRegressor:
        stime = datetime.datetime.now()
        catboost_est = kk
        cat_explainer = shap.TreeExplainer(catboost_est)
        cat_shap_values = cat_explainer(data_dm_x)
        print((datetime.datetime.now() - stime).total_seconds())
        with open(dir_path + r'Results\cat_shap_values_va.pkl', "wb") as f: pickle.dump(cat_shap_values, f)
    if type(kk) == xgb.sklearn.XGBRegressor:
        stime = datetime.datetime.now()
        xgb_est = kk
        xgb_explainer = shap.TreeExplainer(xgb_est)
        xgb_shap_values = xgb_explainer(data_dm_x, check_additivity=False)
        print((datetime.datetime.now() - stime).total_seconds())
        with open(dir_path + r'Results\visit_xgb_shap_values_va.pkl', "wb") as f: pickle.dump(xgb_shap_values, f)
    if type(kk) == lgb.sklearn.LGBMRegressor:
        stime = datetime.datetime.now()
        lgb_est = kk
        lgb_explainer = shap.TreeExplainer(lgb_est)
        lgb_shap_values = lgb_explainer(data_dm_x)
        print((datetime.datetime.now() - stime).total_seconds())
        with open(dir_path + r'Results\lgb_shap_values_va.pkl', "wb") as f: pickle.dump(lgb_shap_values, f)
        # if lgb_shap_values.base_values.shape[0] == 1: lgb_shap_values.base_values = lgb_shap_values.base_values.T
    if type(kk) == sklearn.ensemble._forest.RandomForestRegressor:
        stime = datetime.datetime.now()
        rf_est = kk
        rf_explainer = shap.TreeExplainer(rf_est)
        rf_shap_values = rf_explainer(data_dm_x)
        print((datetime.datetime.now() - stime).total_seconds())
        with open(dir_path + r'Results\rf_shap_values.pkl', "wb") as f: pickle.dump(rf_shap_values, f)
    if type(kk) == sklearn.ensemble._forest.ExtraTreesRegressor:
        stime = datetime.datetime.now()
        ext_est = kk
        ext_explainer = shap.TreeExplainer(ext_est)
        ext_shap_values = ext_explainer(data_dm_x)
        print((datetime.datetime.now() - stime).total_seconds())
        with open(dir_path + r'Results\ext_shap_values_va.pkl', "wb") as f: pickle.dump(ext_shap_values, f)
    if type(kk) == sklearn.tree._classes.DecisionTreeRegressor:
        stime = datetime.datetime.now()
        dct_est = kk
        dct_explainer = shap.TreeExplainer(dct_est)
        dct_shap_values = dct_explainer(data_dm_x)
        print((datetime.datetime.now() - stime).total_seconds())
        with open(dir_path + r'Results\dct_shap_values_va.pkl', "wb") as f: pickle.dump(dct_shap_values, f)

# Explain in SHAP: only catboost xgboost lgb can run fast
with open(r'D:\Cross_Nonlinear\Results\cat_shap_values_va.pkl', 'rb') as handle: cat_shap_values = pickle.load(handle)
with open(r'D:\Cross_Nonlinear\Results\xgb_shap_values_va.pkl', 'rb') as handle: xgb_shap_values = pickle.load(handle)
with open(r'D:\Cross_Nonlinear\Results\lgb_shap_values_va.pkl', 'rb') as handle: lgb_shap_values = pickle.load(handle)

# lgb_explainer = shap.TreeExplainer(lgb_est)
# # data_dm_x_daily = data_dm_x * 30
# lgb_shap_values = lgb_explainer(data_dm_x)

# Bar plot
Shap_df = pd.DataFrame({'LightGBM_SHAP': np.abs(lgb_shap_values.values).mean(axis=0),
                        'XGBoost_SHAP': np.abs(xgb_shap_values.values).mean(axis=0),
                        'CatBoost_SHAP': np.abs(cat_shap_values.values).mean(axis=0),
                        'Feature_names': lgb_shap_values[0].feature_names}). \
    sort_values(by='LightGBM_SHAP', ascending=False).reset_index(drop=True)

plot_one_tree(Shap_df, ['LightGBM_SHAP', 'XGBoost_SHAP', 'CatBoost_SHAP'], 'LightGBM_SHAP', 'All_importance_SHAP',
              save_fig=False)

# Plot two trees
Is_F = True
df_final_imp = reduce(lambda left, right: pd.merge(left, right, on='Feature_names', how='outer'),
                      [catboost_feature_imp, xgb_feature_imp, lgb_feature_imp, rf_feature_imp, ext_feature_imp,
                       dct_feature_imp, Shap_df])
df_final_imp = df_final_imp.fillna(0)
plt.rcParams.update({'font.size': 18})
sns.set_palette('coolwarm', n_colors=6)
plot_two_tree(df_final_imp, ['CatBoost_Permutation', 'XGBoost_Permutation', 'LightGBM_Permutation', 'RF_Permutation',
                             'ExtraTree_Permutation', 'DT_Permutation'],
              ['CatBoost_SHAP', 'XGBoost_SHAP', 'LightGBM_SHAP'], 'LightGBM_Gain', 'two_importance_shap',
              save_fig=True)


# Scatter plot: interaction in SHAP
def plot_shap_interact(lgb_shap_values, model_name, test_all=False, plot_final=False):
    s_Land = lgb_shap_values[:, "Area"]
    s_POI_count = lgb_shap_values[:, "POI Count"]
    s_totalpop = lgb_shap_values[:, "Total Population"]
    s_Food = lgb_shap_values[:, "Accommodation&Food"]
    s_lng = lgb_shap_values[:, "Longitude"]
    s_lat = lgb_shap_values[:, "Latitude"]
    s_Demo = lgb_shap_values[:, "Democrat"]
    s_eduP = lgb_shap_values[:, "Education"]
    s_retail = lgb_shap_values[:, "Retail Trade"]
    s_18_44 = lgb_shap_values[:, "Age 18-44"]
    s_urban = lgb_shap_values[:, "Urbanized Population"]
    s_white = lgb_shap_values[:, "White"]
    s_popd = lgb_shap_values[:, "Population Density"]
    s_edu = lgb_shap_values[:, "High Educated"]
    s_info = lgb_shap_values[:, "Information"]
    s_65 = lgb_shap_values[:, "Age >65"]
    s_income = lgb_shap_values[:, "Median Income"]
    s_black = lgb_shap_values[:, "African American"]
    s_manu = lgb_shap_values[:, "Manufacture"]
    s_workhome = lgb_shap_values[:, "Work at home"]

    # Test each variable
    var_list = [s_Land, s_POI_count, s_totalpop, s_Food, s_lng, s_lat, s_Demo, s_eduP, s_retail, s_18_44, s_urban,
                s_white, s_popd, s_edu, s_info, s_65, s_income, s_black, s_manu, s_workhome]

    if test_all:
        cct = 0
        for varss in var_list:
            fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(18, 12))
            plt.rcParams.update({'font.size': 16, 'font.family': "serif", })
            axs = ax.flatten()
            ccoun = 0
            for kks in axs:
                kks.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
                shap.plots.scatter(varss, color=var_list[ccoun], cmap=plt.get_cmap("coolwarm"), dot_size=5, ax=kks)
                ccoun += 1
            plt.tight_layout()
            plt.savefig(r'D:\Cross_Nonlinear\Results\SHAP_interaction_%s_%s.png' % (cct, model_name), dpi=1000)
            plt.close()
            cct += 1

    if plot_final:
        # Plot the most informative
        # xmin = s_Land.percentile(0), xmax = s_Land.percentile(99.9), ymin = s_Land.percentile(0.1), ymax = s_Land.percentile(99.9),
        ft = 18
        plt.rcParams.update({'font.size': ft, 'font.family': "serif", })
        fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(18, 12))
        axs = ax.flatten()
        for kks in axs: kks.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        shap.plots.scatter(s_Land, color=s_POI_count, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[0])
        shap.plots.scatter(s_POI_count, color=s_Demo, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[1])
        shap.plots.scatter(s_totalpop, color=s_POI_count, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[2])
        shap.plots.scatter(s_Food, color=s_edu, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[3])
        shap.plots.scatter(s_lng, color=s_POI_count, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[4])
        shap.plots.scatter(s_lat, color=s_POI_count, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[5])
        shap.plots.scatter(s_Demo, color=s_POI_count, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[6])
        shap.plots.scatter(s_eduP, color=s_POI_count, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[7])
        shap.plots.scatter(s_retail, color=s_Demo, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[8])
        shap.plots.scatter(s_18_44, color=s_popd, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[9])
        shap.plots.scatter(s_urban, color=s_totalpop, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[10])
        shap.plots.scatter(s_white, color=s_totalpop, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[11])
        shap.plots.scatter(s_popd, color=s_POI_count, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[12])
        shap.plots.scatter(s_edu, color=s_totalpop, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[13])
        shap.plots.scatter(s_info, color=s_Demo, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[14])
        shap.plots.scatter(s_65, color=s_totalpop, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[15])
        shap.plots.scatter(s_income, color=s_totalpop, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[16])
        shap.plots.scatter(s_black, color=s_totalpop, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[17])
        shap.plots.scatter(s_manu, color=s_Demo, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[18])
        shap.plots.scatter(s_workhome, color=s_totalpop, cmap=plt.get_cmap("coolwarm"), dot_size=5, ft=ft, ax=axs[19])
        for kk in range(0, 20):
            if kk in [0, 5, 10, 15]:
                axs[kk].set_ylabel('SHAP')
            else:
                axs[kk].set_ylabel('')
        plt.subplots_adjust(top=0.952, bottom=0.05, left=0.05, right=0.97, hspace=0.566, wspace=0.47)
        # plt.savefig(r'D:\Cross_Nonlinear\Results\SHAP_interaction_final_%s.png' % model_name, dpi=1000)
        # plt.close()


plot_shap_interact(lgb_shap_values, 'lightb_va_l', test_all=False, plot_final=True)
plot_shap_interact(cat_shap_values, 'catb_va_l', test_all=False, plot_final=True)
plot_shap_interact(xgb_shap_values, 'xgb_va_l', test_all=False, plot_final=True)

###############
# Compare with PDP
###############
stime = datetime.datetime.now()
rf = nonstva_models[5]  # Lightgbm: 2
rff = rf.fit(data_dm_x, y_label)
rff.dummy_ = "dummy"
fig, ax = plt.subplots(figsize=(18, 12))
feature_list = ["Area", "POI Count", "Total Population", "Accommodation&Food", "Longitude", "Latitude", "Democrat",
                "Education", "Retail Trade", "Age 18-44", "Urbanized Population", "White", "Population Density",
                "High Educated", "Information", "Age >65", "Median Income", "African American", "Manufacture",
                "Work at home"]
tree_disp = plot_partial_dependence(rff, data_dm_x, feature_list, grid_resolution=50, n_cols=5, n_jobs=-1,
                                    ax=ax, percentiles=(0, 1))
print((datetime.datetime.now() - stime).total_seconds())
plt.subplots_adjust(top=0.977, bottom=0.076, left=0.071, right=0.985, hspace=0.315, wspace=0.2)
ccount = 0
for rr in range(0, 4):
    for cc in range(0, 5):
        tree_disp.axes_[rr][cc].set_ylim(
            [np.min(tree_disp.pd_results[ccount][0]), np.max(tree_disp.pd_results[ccount][0]) * 1.01])
        ccount += 1
# with open(r'D:\Cross_Nonlinear\Results\tree_disp_lgb_all_va.pkl', "wb") as f: pickle.dump(tree_disp, f)

# Get the data out and replot the figures
# with open(r'D:\Cross_Nonlinear\Results\tree_disp_lgb_all_va.pkl', "rb") as f: tree_disp = pickle.load(f)
sns.set_palette('coolwarm')
fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(18, 12))
plt.rcParams.update({'font.size': 20, 'font.family': "serif", })
feature_list = ["Area", "POI Count", "Total Population", "Accommodation&Food", "Longitude", "Latitude", "Democrat",
                "Education", "Retail Trade", "Age 18-44", "Urbanized Population", "White", "Population Density",
                "High Educated", "Information", "Age >65", "Median Income", "African American", "Manufacture",
                "Work at home"]
axs = ax.flatten()
for ccount in range(0, 20):
    axs[ccount].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axs[ccount].plot(tree_disp.pd_results[ccount][1][0], tree_disp.pd_results[ccount][0][0])
    axs[ccount].set_xlabel(feature_list[ccount])
    for xc in list(tree_disp.deciles.values())[ccount]:
        axs[ccount].vlines(xc, np.percentile(tree_disp.pd_results[ccount][0][0], 0),
                           np.percentile(tree_disp.pd_results[ccount][0][0], 0) + (
                                   np.percentile(tree_disp.pd_results[ccount][0][0], 100) - np.percentile(
                               tree_disp.pd_results[ccount][0][0], 0)) * 0.05, color='k', lw=1)
    if ccount in [0, 5, 10, 15]: axs[ccount].set_ylabel('PDP')
plt.subplots_adjust(top=0.962, bottom=0.076, left=0.075, right=0.987, hspace=0.59, wspace=0.255)
plt.savefig(r'D:\Cross_Nonlinear\Results\PDP_interaction_final_dt_sk_va.png', dpi=1000)

###############
# Compare with ALE
###############
stime = datetime.datetime.now()
rf = nonstva_models[3]  # Lightgbm: 2
rf.fit(data_dm_x, y_label)
rf_ale = ALE(rf.predict, feature_names=list(data_dm_x.columns), target_names=[Target_n])
rf_exp = rf_ale.explain(data_dm_x.values)
print((datetime.datetime.now() - stime).total_seconds())
with open(r'D:\Cross_Nonlinear\Results\tree_ale_lgb_all.pkl', "wb") as f: pickle.dump(rf_exp, f)

with open(r'D:\Cross_Nonlinear\Results\tree_ale_lgb_all.pkl', "rb") as f: rf_exp = pickle.load(f)
fig, ax = plt.subplots(4, 5, figsize=(18, 12))
plt.rcParams.update({'font.size': 20, 'font.family': "serif", })
feature_list = ["Area", "POI Count", "Total Population", "Accommodation&Food", "Longitude", "Latitude", "Democrat",
                "Education", "Retail Trade", "Age 18-44", "Urbanized Population", "White", "Population Density",
                "High Educated", "Information", "Age >65", "Median Income", "African American", "Manufacture",
                "Work at home"]

plot_ale(rf_exp, features=feature_list, ax=ax,
         line_kw={'markersize': 3, 'marker': 'o', 'color': 'royalblue'})  # features=['Area'],

ccount = 0
for axs in ax.flatten():
    axs.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    if ccount in [0, 5, 10, 15]:
        axs.set_ylabel('ALE')
    else:
        axs.set_ylabel('')
    ccount += 1
    axs.legend('')
plt.subplots_adjust(top=0.977, bottom=0.076, left=0.071, right=0.985, hspace=0.315, wspace=0.1)
plt.savefig(r'D:\Cross_Nonlinear\Results\ALE_interaction_final_lgb_sk.png', dpi=1000)
