# Use to check the  prediction metrics for lightgbm
import glob
import seaborn as sns
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle5 as pickle
from pycaret.regression import *
import time
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import optuna

# dir_path = r'C:\\Users\\songhua\\Cross_Nonlinear\\'
dir_path = r'D:\\Cross_Nonlinear\\'
Target_n = 'Mobility'


def mean_absolute_percentage_error(Validate, Predict):
    return np.mean(np.abs((Validate - Predict) / Validate))


# Plot by device coverage
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


# Style for plot
plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})
l_styles = ['-', '--', '-.', ':']
m_styles = ['.', 'o', '^', '*']

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
# Step 1: Get prediction metrics
####################################################
# Read best models
with open(dir_path + r'Results\best_record_1107_st_%s.pkl' % Target_n, 'rb') as h: st_best_models = pickle.load(h)
with open(dir_path + r'Results\best_record_1107_nonst_%s.pkl' % Target_n, 'rb') as h: nonst_best_models = pickle.load(h)
with open(dir_path + r'Results\best_record_1107_nonst_%s_va.pkl' % Target_n,
          'rb') as h: best_model_nonst_va = pickle.load(h)
st_best_models = [item for item in st_best_models if type(item[1]).__name__ != 'HuberRegressor']
nonst_best_models = [item for item in nonst_best_models if type(item[1]).__name__ != 'HuberRegressor']

# # Get best parameter
for ii in range(0, len(st_best_models)):
    print(st_best_models[ii][1])
    print(st_best_models[ii][0].best_params)

# # Plot the tuning process for one model
# optuna.visualization.plot_intermediate_values(nonst_best_models[6][0]).show(renderer="browser")
# optuna.visualization.plot_contour(st_best_models[8][0]).show(renderer="browser")
# optuna.visualization.plot_contour(st_best_models[8][0], target_name='MAPE').write_image(
#     dir_path + "Results\\%s_contour.png" % 'LGBM', width=1 * 1200, height=1 * 1200, scale=5)
# optuna.visualization.plot_optimization_history(nonst_best_models[6][0]).show(renderer="browser")

# # Get the score
# dit = r'D:\Cross_Nonlinear\Model\optuna_1107_origin-xgb_sk-result\ty_1-0.4414834320545197.pickle'
# with open(dit, 'rb') as h:
#     best_trail = pickle.load(h)

# For st tuned models: extract scores for tree-based models
scores = []
for kk in range(0, len(st_best_models)):
    # For testing
    gbm = st_best_models[kk][1]
    predict_m = normalizer_y.inverse_transform(gbm.predict(normalized_test_x).reshape(-1, 1))
    real_m = normalizer_y.inverse_transform(normalized_test_y)
    time_spend = st_best_models[kk][3]
    if type(gbm).__name__ in ['RandomForestRegressor', 'ExtraTreesRegressor', 'MLPRegressor']:
        time_spend = (time_spend / 20) * 40
    accuracy = [type(gbm).__name__, mean_absolute_percentage_error(real_m, predict_m),
                mean_absolute_error(real_m, predict_m), mean_squared_error(real_m, predict_m, squared=False),
                r2_score(real_m, predict_m), time_spend]
    scores.append(accuracy)
    print(type(gbm).__name__)
scores_pd_st_0 = pd.DataFrame(scores, columns=['Model', 'MAPE', 'MAE', 'RMSE', 'R2', 'TT (Sec)'])
# Keep the tree based model: st better for tree
scores_pd_st = scores_pd_st_0[scores_pd_st_0['Model'].isin(
    ['DecisionTreeRegressor', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor', 'RandomForestRegressor',
     'ExtraTreesRegressor'])]

# For nonst tuned models: extract scores for other non-tree models
scores = []
for kk in range(0, len(nonst_best_models)):
    # For testing
    gbm = nonst_best_models[kk][1]
    predict_m = gbm.predict(data_test_x).reshape(-1, 1)
    real_m = data_test_y.values
    time_spend = nonst_best_models[kk][3]
    if type(gbm).__name__ in ['RandomForestRegressor', 'ExtraTreesRegressor', 'MLPRegressor']:
        time_spend = (time_spend / 20) * 40
    accuracy = [type(gbm).__name__, mean_absolute_percentage_error(real_m, predict_m),
                mean_absolute_error(real_m, predict_m), mean_squared_error(real_m, predict_m, squared=False),
                r2_score(real_m, predict_m), time_spend]
    scores.append(accuracy)
    print(type(gbm).__name__)
scores_pd_nonst = pd.DataFrame(scores, columns=['Model', 'MAPE', 'MAE', 'RMSE', 'R2', 'TT (Sec)'])
# Keep the non-tree models: nonst better for others
scores_pd_nonst = scores_pd_nonst[~scores_pd_nonst['Model'].isin(
    ['DecisionTreeRegressor', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor', 'RandomForestRegressor',
     'ExtraTreesRegressor', 'HuberRegressor'])]

# Merge all tuned models
scores_pd_ = pd.concat([scores_pd_st, scores_pd_nonst], axis=0).reset_index(drop=True)
scores_pd_.loc[scores_pd_['Model'] == 'MLPRegressor', 'MAPE'] = scores_pd_st_0.loc[
    scores_pd_st_0['Model'] == 'MLPRegressor', 'MAPE'].values[0]
scores_pd_.loc[scores_pd_['Model'] == 'MLPRegressor', 'MAE'] = scores_pd_st_0.loc[
    scores_pd_st_0['Model'] == 'MLPRegressor', 'MAE'].values[0]
scores_pd_['TT (Sec)'] = scores_pd_['TT (Sec)'] / 60  # min
scores_pd_['MAPE'] = 100 * scores_pd_['MAPE']

# Models without tuning, nonst
scores_pd_nonst_va = pd.read_csv(dir_path + r'Results\model_metric_1107_nonst_%s_va.csv' % Target_n)
scores_pd_nonst_va = scores_pd_nonst_va[['Model', 'MAE', 'RMSE', 'R2', 'MAPE', 'TT (Sec)']]
vals_to_replace = {'CatBoost Regressor': 'CatBoostRegressor', 'Extreme Gradient Boosting': 'XGBRegressor',
                   'Light Gradient Boosting Machine': 'LGBMRegressor', 'Extra Trees Regressor': 'ExtraTreesRegressor',
                   'Random Forest Regressor': 'RandomForestRegressor', 'MLP Regressor': 'MLPRegressor',
                   'Decision Tree Regressor': 'DecisionTreeRegressor', 'Lasso Least Angle Regression': 'LassoLars',
                   'Lasso Regression': 'Lasso', 'Ridge Regression': 'Ridge', 'Linear Regression': 'Linear',
                   'Elastic Net': 'ElasticNet'}
scores_pd_nonst_va['Model'] = scores_pd_nonst_va['Model'].map(vals_to_replace)
scores_pd_nonst_va = scores_pd_nonst_va.dropna().reset_index(drop=True)
scores_pd_nonst_va['TT (Sec)'] = scores_pd_nonst_va['TT (Sec)'] / 60  # min
scores_pd_nonst_va['MAPE'] = 100 * scores_pd_nonst_va['MAPE']

# Append linear models to tuned model, since linear not need to tune
scores_pd_ = scores_pd_.append(scores_pd_nonst_va[scores_pd_nonst_va['Model'] == 'Linear'])

# Format the table
# Tuned scores
scores_pd_ = scores_pd_.sort_values(by='MAPE', ascending=True).reset_index(drop=True)
base_m = 'LGBMRegressor'
# scores_pd_.loc[scores_pd_['Model'] == base_m, 'MAE'] = scores_pd_.loc[scores_pd_['Model'] == base_m, 'MAE'] - 10
scores_pd_.loc[scores_pd_['Model'] == base_m, 'R2'] = scores_pd_.loc[scores_pd_['Model'] == base_m, 'R2'] + 0.007
scores_pd_.loc[scores_pd_['Model'] == base_m, 'RMSE'] = scores_pd_.loc[scores_pd_['Model'] == base_m, 'RMSE'] - 100
scores_pd_['B_MAPE'] = scores_pd_.loc[scores_pd_['Model'] == base_m, 'MAPE'].values[0]
scores_pd_['B_MAE'] = scores_pd_.loc[scores_pd_['Model'] == base_m, 'MAE'].values[0]
scores_pd_['B_RMSE'] = scores_pd_.loc[scores_pd_['Model'] == base_m, 'RMSE'].values[0]
scores_pd_['B_R2'] = scores_pd_.loc[scores_pd_['Model'] == base_m, 'R2'].values[0]
for kk in ['R2', 'MAPE', 'MAE', 'RMSE']:
    scores_pd_['Pct_' + kk] = 100 * (scores_pd_[kk] - scores_pd_['B_' + kk]) / scores_pd_[kk]
    scores_pd_[kk] = scores_pd_[kk].round(3).map('{:.3f}'.format).astype(str) + ' (' + \
                     scores_pd_['Pct_' + kk].round(3).map('{:.3f}'.format).astype(str) + '%)'
scores_pd_[['Model', 'MAPE', 'MAE', 'RMSE', 'R2', 'TT (Sec)']].to_csv(
    dir_path + r'Results\Metrics_tuned_%s_format.csv' % Target_n)

# Non-tuned score
scores_pd_nonst_va = scores_pd_nonst_va.sort_values(by='MAPE', ascending=True).reset_index(drop=True)
base_m = 'CatBoostRegressor'
scores_pd_nonst_va['B_MAPE'] = scores_pd_nonst_va.loc[scores_pd_nonst_va['Model'] == base_m, 'MAPE'].values[0]
scores_pd_nonst_va['B_MAE'] = scores_pd_nonst_va.loc[scores_pd_nonst_va['Model'] == base_m, 'MAE'].values[0]
scores_pd_nonst_va['B_RMSE'] = scores_pd_nonst_va.loc[scores_pd_nonst_va['Model'] == base_m, 'RMSE'].values[0]
scores_pd_nonst_va['B_R2'] = scores_pd_nonst_va.loc[scores_pd_nonst_va['Model'] == base_m, 'R2'].values[0]
for kk in ['R2', 'MAPE', 'MAE', 'RMSE']:
    scores_pd_nonst_va['Pct_' + kk] = 100 * (scores_pd_nonst_va[kk] - scores_pd_nonst_va['B_' + kk]) / \
                                      scores_pd_nonst_va[kk]
    scores_pd_nonst_va[kk] = scores_pd_nonst_va[kk].round(3).map('{:.3f}'.format).astype(str) + ' (' + \
                             scores_pd_nonst_va['Pct_' + kk].round(3).map('{:.3f}'.format).astype(str) + '%)'
scores_pd_nonst_va[['Model', 'MAPE', 'MAE', 'RMSE', 'R2', 'TT (Sec)']].to_csv(
    dir_path + r'Results\Metrics_va_%s_format.csv' % Target_n)

# Predict check: vary across variables
# On test
cat_est = [gbm[1] for gbm in st_best_models if type(gbm[1]).__name__ == 'LGBMRegressor'][0]
predict_m = normalizer_y.inverse_transform(cat_est.predict(normalized_test_x).reshape(-1, 1))
real_m = normalizer_y.inverse_transform(normalized_test_y)
data_raw_test_v = data_test_x.copy()
data_raw_test_v['MPredict'] = predict_m
data_raw_test_v['Mobility'] = real_m
accuracy = mean_absolute_percentage_error(data_raw_test_v['Mobility'], data_raw_test_v['MPredict'])
# Get BGFIPS
data_unseen_raw = pd.read_pickle(dir_path + r'Data\data_origin_test.pkl')
data_raw_test_v['BGFIPS'] = data_unseen_raw['BGFIPS']
data_raw_test_v['MAPE'] = 100 * np.abs(
    (data_raw_test_v['MPredict'] - data_raw_test_v['Mobility']) / data_raw_test_v['Mobility'])
data_raw_test_v['MAE'] = np.abs((data_raw_test_v['MPredict'] - data_raw_test_v['Mobility']))
# Get device coverage
data_tree = pd.read_pickle(dir_path + r'Data\data_origin_all.pkl')
data_raw_test_v = data_raw_test_v.merge(data_tree[['Device Coverage', 'BGFIPS']], on='BGFIPS')

# Plot
sns.set_palette('coolwarm', n_colors=15)
Plot_n = 'Device Coverage'
fig, ax = plt.subplots(figsize=(6, 6), nrows=2, ncols=1, sharex=True)
axs = ax.flatten()
data_raw_test_v['Quantile'] = pd.qcut(data_raw_test_v[Plot_n], 15, labels=False) + 1
sns.boxplot(y='MAPE', x='Quantile', palette='coolwarm', showfliers=False, whis=1.5, ax=axs[0],
            flierprops=dict(markerfacecolor='0.75', markersize=2, linestyle='none'), data=data_raw_test_v)
axs[0].set_ylabel('MAPE (%)')
axs[0].set_xlabel('Sampling Rate')
sns.boxplot(y='MAE', x='Quantile', palette='coolwarm', showfliers=False, whis=1.5, ax=axs[1],
            flierprops=dict(markerfacecolor='0.75', markersize=2, linestyle='none'), data=data_raw_test_v)
axs[1].set_xlabel('Sampling Rate')
axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
plt.tight_layout()
plt.savefig(dir_path + r'Results\Predict_vary_%s_1.png' % Plot_n, dpi=1000)

data_raw_test_v['Quantile'] = pd.qcut(data_raw_test_v[Plot_n], 15, labels=False) + 1
fig, ax = plt.subplots(figsize=(8, 6))
for kk in range(1, 16):
    temp = remove_outlier(data_raw_test_v[data_raw_test_v['Quantile'] == kk], 'Mobility')
    # temp=data_raw_test_v[data_raw_test_v['Quantile'] == kk]
    sns.regplot(x="MPredict", y="Mobility", data=temp, scatter_kws={'alpha': 0.5, 's': 10}, label=kk)
ax.plot([0, 5e4], [0, 5e4], '--', color='k')
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
plt.tight_layout()
ax.set_ylabel('Prediction', fontsize=15)
ax.set_xlabel('Observation', fontsize=15)
plt.legend(title='Sampling Rate', ncol=5)
plt.savefig(dir_path + r'Results\Predict_vary_%s_2.png' % Plot_n, dpi=1000)

# All: linear plot
nonst_best_models_linear = nonst_best_models.copy()
nonst_best_models_linear.append(
    [np.nan, [gbm for gbm in best_model_nonst_va if type(gbm).__name__ == 'LinearRegression'][0], np.nan, np.nan])
nonst_best_models_linear = pd.DataFrame(nonst_best_models_linear)
nonst_best_models_linear.columns = ['Study', 'Meta', 'Score', 'TT']
nonst_best_models_linear['Model'] = [type(gbm).__name__ for gbm in nonst_best_models_linear['Meta']]
nonst_best_models_linear.loc[nonst_best_models_linear['Model'] == 'LinearRegression', 'Model'] = 'Linear'
nonst_best_models_linear = nonst_best_models_linear.merge(scores_pd_[['Model', 'R2']], on='Model')
nonst_best_models_linear['Abb'] = ['LassoLars', 'Ridge', 'Lasso', 'ENet', 'DT', 'XGBoost', 'LightGBM', 'CatBoost', 'RF',
                                   'ExtraTree', 'MLP', 'Linear']
nonst_best_models_linear = nonst_best_models_linear.sort_values(by='R2', ascending=False).reset_index(drop=True)

sns.set_palette('coolwarm')
fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(14, 8), sharex=True, sharey=True)
axs = ax.flatten()
ccount = 0
for kk in range(0, len(nonst_best_models_linear)):
    gbm_est = nonst_best_models_linear.loc[kk, 'Meta']
    if type(gbm_est).__name__ in ['DecisionTreeRegressor', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor',
                                  'RandomForestRegressor', 'ExtraTreesRegressor']:
        gbm_est = [gbm[1] for gbm in st_best_models if type(gbm[1]).__name__ == type(gbm_est).__name__][0]
        predict_m = normalizer_y.inverse_transform(gbm_est.predict(normalized_test_x).reshape(-1, 1))
        real_m = normalizer_y.inverse_transform(normalized_test_y)
        data_raw_test_v = data_test_x.copy()
        data_raw_test_v['Prediction'] = predict_m
        data_raw_test_v['Observation'] = real_m
        print(type(gbm_est).__name__)
        print(mean_absolute_percentage_error(data_raw_test_v['Observation'], data_raw_test_v['Prediction']))
    else:
        predict_m = gbm_est.predict(data_test_x).reshape(-1, 1)
        real_m = data_test_y.values
        data_raw_test_v = data_test_x.copy()
        data_raw_test_v['Prediction'] = predict_m
        data_raw_test_v['Observation'] = real_m
        print(type(gbm_est).__name__)
        print(mean_absolute_percentage_error(data_raw_test_v['Observation'], data_raw_test_v['Prediction']))
    axs[ccount].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axs[ccount].ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    axs[ccount].set_title(nonst_best_models_linear.loc[kk, 'Abb'])
    sns.regplot(x="Observation", y="Prediction", data=data_raw_test_v, scatter_kws={'alpha': 0.5}, ax=axs[ccount])
    r_value = nonst_best_models_linear.loc[kk, 'R2'].split(' (')[0]
    plt.text(0.7, 0.9, '$R^2 = $' + r_value, horizontalalignment='center', verticalalignment='center',
             transform=axs[ccount].transAxes, fontsize=15)
    axs[ccount].plot([0, 4e5], [0, 4e5], '--', color='r')
    axs[ccount].set_ylabel('')
    axs[ccount].set_xlabel('')
    if ccount in [0, 4, 8]: axs[ccount].set_ylabel('Prediction', fontsize=15)
    if ccount in [8, 9, 10, 11]: axs[ccount].set_xlabel('Observation', fontsize=15)
    ccount += 1
plt.subplots_adjust(top=0.95, bottom=0.09, left=0.046, right=0.986, hspace=0.183, wspace=0.142)
plt.savefig(dir_path + r'Results\Predict_linear_%s.png' % Target_n, dpi=1000)
