# Use to check the  prediction metrics for lightgbm
import optuna
import sklearn
import glob
import seaborn as sns
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
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import HuberRegressor, Lasso, Ridge, ElasticNet, LassoLars, BayesianRidge
from pycaret.regression import *
from pycaret.utils import check_metric
import time
from sklearn import preprocessing
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# dir_path = r'C:\\Users\\songhua\\Cross_Nonlinear\\'
dir_path = r'D:\\Cross_Nonlinear\\'


def mean_absolute_percentage_error(Validate, Predict):
    return np.mean(np.abs((Validate - Predict) / Validate))


def mape_xgb(Predict, dval_lgb):
    Validate = dval_lgb.get_label()
    return 'mape', np.mean(np.abs((Validate - Predict) / Validate))


####################################################
# Step 0: Prepare data
####################################################
# Read CBG data
with open(dir_path + r'Data\\CBG_XY_09_10.pkl', "rb") as fh: data_tree = pickle.load(fh)
# Remove zero
data_tree.loc[data_tree['raw_stop_counts'] <= 0, 'raw_stop_counts'] = 1
# To category
data_tree['CTFIPS'] = data_tree['BGFIPS'].str[0:5]
data_tree['CTFIPS'] = data_tree['CTFIPS'].astype('float')
data_tree['STFIPS'] = data_tree['STFIPS'].astype('category')
data_tree = data_tree[data_tree['raw_stop_counts'] > 30]
data_tree = data_tree.dropna().reset_index(drop=True)

# Construct train and validation dataset
data_tree['Drive'] = data_tree['Drive_alone_R'] + data_tree['Carpool_R']
data_tree['Walk&Bike'] = data_tree['Walk_R'] + data_tree['Bicycle_R']
data_tree['Accommodation&Food'] = data_tree['72']
data_tree['Retail Trade'] = data_tree['44'] + data_tree['45']
data_tree['Recreation'] = data_tree['71']
data_tree['Health Care'] = data_tree['62']
data_tree['Education'] = data_tree['61']
data_tree['Finance'] = data_tree['52']
data_tree['Transportation'] = data_tree['48'] + data_tree['49']
data_tree['Information'] = data_tree['51']
data_tree['Public'] = data_tree['92']
data_tree['Real Estate'] = data_tree['53']
data_tree['Wholesale Trade'] = data_tree['42']
data_tree['Manufacture'] = data_tree['31'] + data_tree['33'] + data_tree['32']
data_tree['Administration'] = data_tree['56']

predictors = \
    ['Education_Degree_R', 'Unemployed_R', 'Median_income', 'Rent_to_Income', 'GINI', 'Is_Central', 'Democrat_R',
     'Household_Below_Poverty_R',

     'ALAND', 'Lng', 'Lat', 'STFIPS', 'CTFIPS',

     'Public_Transit_R', 'Walk&Bike', 'Worked_at_home_R',

     'Total_Population', 'Population_Density', 'Urbanized_Areas_Population_R', 'Rural_Population_R',
     'White_Non_Hispanic_R', 'HISPANIC_LATINO_R', 'Black_R', 'Indian_Others_R', 'Asian_R', 'Bt_18_44_R',
     'Bt_45_64_R', 'Over_65_R', 'Male_R',

     'POI_count', 'Administration', 'Manufacture', 'Wholesale Trade', 'Real Estate', 'Public', 'Information',
     'Transportation', 'Finance', 'Education', 'Health Care', 'Recreation', 'Retail Trade', 'Accommodation&Food']
# category_label = ['CTFIPS']
sample_size = len(data_tree)  # len(data_tree)
train_test_ratio = 0.9
dataset = data_tree[predictors + ['raw_stop_counts']].sample(sample_size)

# Rename columns
dataset.rename(
    {'Total_Population': 'Total Population', 'Population_Density': 'Population Density',
     'Urbanized_Areas_Population_R': 'Urbanized Population', 'Education_Degree_R': 'High Educated',
     'Unemployed_R': 'Unemployed Rate', 'Rural_Population_R': 'Rural Population', 'Median_income': 'Median Income',
     'Rent_to_Income': 'Rent to Income', 'ALAND': 'Area', 'Lng': 'Longitude', 'Lat': 'Latitude',
     'Is_Central': 'Central', 'Democrat_R': 'Democrat', 'No_Insurance_R': 'No Insurance', 'Time_30_60_R': 'Time 30-60',
     'Household_Below_Poverty_R': 'Poverty', 'Time_10_30_R': 'Time 10-30', 'Time_greater_60_R': 'Time >60',
     'Drive_alone_R': 'Drive', 'Carpool_R': 'Carpool', 'Public_Transit_R': 'Public Transit', 'Bicycle_R': 'Bicycle',
     'Walk_R': 'Walk', 'Worked_at_home_R': 'Work at home', 'HISPANIC_LATINO_R': 'Hispanic', 'LUM_Race': 'Race Mix',
     'Black_R': 'African American', 'Indian_Others_R': 'Indian&Others', 'Asian_R': 'Asian', 'Bt_18_44_R': 'Age 18-44',
     'Bt_45_64_R': 'Age 45-64', 'Over_65_R': 'Age >65', 'Male_R': 'Male', 'White_Non_Hispanic_R': 'White',
     'POI_count': 'POI Count', 'raw_stop_counts': 'Mobility'}, axis=1, inplace=True)

# # Multicollinearity test
# Remove_predictor = ['Mobility', 'STFIPS', ]
# vif_data = pd.DataFrame()
# vif_data["feature"] = dataset.drop(Remove_predictor, axis=1).columns
# vif_data["VIF"] = [variance_inflation_factor(dataset.drop(Remove_predictor, axis=1).values, ix) for
#                    ix in range(len(vif_data["feature"]))]

# One hot STFIPS
one_hot = pd.get_dummies(dataset['STFIPS'])
one_hot.columns = ['STFIPS_' + ii for ii in list(one_hot.columns)]
dataset = dataset.drop('STFIPS', axis=1)
dataset = dataset.join(one_hot)
dataset = dataset.drop(['STFIPS_11'], axis=1)

dataset.loc[:, dataset.dtypes == np.float64] = dataset.loc[:, dataset.dtypes == np.float64].astype(np.float32)
data = dataset.sample(frac=train_test_ratio, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

# Set envir
exp_reg = setup(data=data, target='Mobility', session_id=2, fold=5, normalize=True, transformation=True,
                transform_target=True, use_gpu=False, n_jobs=-1)

# Set train and test
data_train_x = data.drop(['Mobility'], axis=1)
data_train_y = data[['Mobility']]
data_test_x = data_unseen.drop(['Mobility'], axis=1)
data_test_y = data_unseen[['Mobility']]
# Normalise X
normalizer_x = preprocessing.StandardScaler()
normalized_train_x = pd.DataFrame(normalizer_x.fit_transform(data_train_x), columns=data_train_x.columns)
normalized_test_x = pd.DataFrame(normalizer_x.transform(data_test_x), columns=data_test_x.columns)
# Power transfer Y
normalizer_y = preprocessing.PowerTransformer(method='yeo-johnson', standardize=False)
normalized_train_y = pd.DataFrame(normalizer_y.fit_transform(data_train_y), columns=data_train_y.columns)
normalized_test_y = pd.DataFrame(normalizer_y.transform(data_test_y), columns=data_test_y.columns)

# Test linear model
llar_est = LassoLars(alpha=0.0003, max_iter=500)
llar_est.fit(normalized_train_x, normalized_train_y)
llar_feature_imp = pd.DataFrame({'Coeff': llar_est.coef_, 'Feature_names': list(normalized_train_x.columns)}). \
    sort_values(by=['Coeff'], ascending=False)
print(llar_feature_imp)
print('MAPE: %s' % mean_absolute_percentage_error(
    normalizer_y.inverse_transform(normalized_test_y), normalizer_y.inverse_transform(
        llar_est.predict(normalized_test_x).reshape(-1, 1))))
plt.plot(normalizer_y.inverse_transform(normalized_test_y), normalizer_y.inverse_transform(
    llar_est.predict(normalized_test_x).reshape(-1, 1)), 'o')

llar_est = HuberRegressor(alpha=0.7 * 1e7, max_iter=300)
llar_est.fit(normalized_train_x, normalized_train_y)
llar_feature_imp = pd.DataFrame({'Coeff': llar_est.coef_, 'Feature_names': list(normalized_train_x.columns)}). \
    sort_values(by=['Coeff'], ascending=False)
print(llar_feature_imp)
print('MAPE: %s' % mean_absolute_percentage_error(
    normalizer_y.inverse_transform(normalized_test_y), normalizer_y.inverse_transform(
        llar_est.predict(normalized_test_x).reshape(-1, 1))))
plt.plot(normalizer_y.inverse_transform(normalized_test_y), normalizer_y.inverse_transform(
    llar_est.predict(normalized_test_x).reshape(-1, 1)), 'o')

llar_est = Lasso(alpha=0.1, max_iter=1000)
llar_est.fit(normalized_train_x, normalized_train_y)
llar_feature_imp = pd.DataFrame({'Coeff': llar_est.coef_, 'Feature_names': list(normalized_train_x.columns)}). \
    sort_values(by=['Coeff'], ascending=False)
print(llar_feature_imp)
print('MAPE: %s' % mean_absolute_percentage_error(
    normalizer_y.inverse_transform(normalized_test_y), normalizer_y.inverse_transform(
        llar_est.predict(normalized_test_x).reshape(-1, 1))))
plt.plot(normalizer_y.inverse_transform(normalized_test_y), normalizer_y.inverse_transform(
    llar_est.predict(normalized_test_x).reshape(-1, 1)), 'o')

llar_est = Ridge(alpha=5 * 1e6, max_iter=1000)
llar_est.fit(normalized_train_x, normalized_train_y)
llar_feature_imp = pd.DataFrame({'Coeff': llar_est.coef_[0], 'Feature_names': list(normalized_train_x.columns)}). \
    sort_values(by=['Coeff'], ascending=False)
print(llar_feature_imp)
print('MAPE: %s' % mean_absolute_percentage_error(
    normalizer_y.inverse_transform(normalized_test_y), normalizer_y.inverse_transform(
        llar_est.predict(normalized_test_x).reshape(-1, 1))))
plt.plot(normalizer_y.inverse_transform(normalized_test_y), normalizer_y.inverse_transform(
    llar_est.predict(normalized_test_x).reshape(-1, 1)), 'o')

llar_est = ElasticNet(alpha=0.2, max_iter=1000)
llar_est.fit(normalized_train_x, normalized_train_y)
llar_feature_imp = pd.DataFrame({'Coeff': llar_est.coef_, 'Feature_names': list(normalized_train_x.columns)}). \
    sort_values(by=['Coeff'], ascending=False)
print(llar_feature_imp)
print('MAPE: %s' % mean_absolute_percentage_error(
    normalizer_y.inverse_transform(normalized_test_y), normalizer_y.inverse_transform(
        llar_est.predict(normalized_test_x).reshape(-1, 1))))
plt.plot(normalizer_y.inverse_transform(normalized_test_y), normalizer_y.inverse_transform(
    llar_est.predict(normalized_test_x).reshape(-1, 1)), 'o')

from sklearn.model_selection import GridSearchCV

cv = KFold(n_splits=5, random_state=1)
lasso_alphas = np.linspace(0, 0.2, 21)
lasso = Lasso()
grid = dict()
grid['alpha'] = lasso_alphas
gscv = GridSearchCV(lasso, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
results = gscv.fit(normalized_train_x, normalized_train_y)
print('MAE: %.5f' % results.best_score_)
print('Config: %s' % results.best_params_)
