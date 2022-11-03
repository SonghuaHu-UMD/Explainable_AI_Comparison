import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
from sklearn import preprocessing
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches

pd.options.mode.chained_assignment = None

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

# Date prerpare
data_tree = pd.read_pickle(r'D:\Cross_Nonlinear\Data\CBG_XY_09_10.pkl')
# Remove zero
data_tree.loc[data_tree['raw_visit_counts'] <= 0, 'raw_visit_counts'] = 1
# To category
data_tree['CTFIPS'] = data_tree['BGFIPS'].str[0:5]
data_tree['CTFIPS'] = data_tree['CTFIPS'].astype('float')
data_tree['STFIPS'] = data_tree['STFIPS'].astype('category')
data_tree = data_tree.dropna().reset_index(drop=True)
data_tree['raw_visit_counts'] = data_tree['raw_visit_counts'] * 2.475
data_tree = data_tree[data_tree['raw_visit_counts'] < max(data_tree['raw_stop_counts'])].reset_index(drop=True)
# data_tree = data_tree.sort_values(by='raw_visit_counts').reset_index(drop=True)

# Merge with coverage
de_cover = pd.read_csv(
    r'D:\Cross_Nonlinear\Data\Monthly_Patterns\part-00000-tid-3706423432024995753-4970d83b-5e07-497f-9451-7e2c1faa5f1c-263412-1-c000.csv')
de_cover = de_cover[de_cover['iso_country_code'] == 'US']
de_cover['BGFIPS'] = de_cover['census_block_group'].astype(str).apply(lambda x: x.zfill(12))
data_tree = data_tree.merge(de_cover[['BGFIPS', 'number_devices_residing']], on='BGFIPS', how='left')
data_tree['Device Coverage'] = 100 * (data_tree['number_devices_residing'] / data_tree['Total_Population'])
data_tree['Device Coverage'] = data_tree['Device Coverage'].fillna(0)
data_tree.loc[data_tree['Device Coverage'] > 100, 'Device Coverage'] = 100
data_tree['number_devices_residing'] = data_tree['number_devices_residing'].fillna(0)
data_tree = data_tree[(data_tree['Device Coverage'] < 99) & (data_tree['Device Coverage'] > 1) & (
        data_tree['raw_visit_counts'] >= 30)].reset_index(drop=True)

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
data_tree['Scientific'] = data_tree['54']
data_tree['Total_Population'] = data_tree['Total_Population'] / 1e4
data_tree['Population_Density'] = data_tree['Population_Density'] / 1e4
data_tree['Median_income'] = data_tree['Median_income'] / 1e4
data_tree['Is_Central'].value_counts() / len(data_tree)

predictors = \
    ['Total_Population', 'Population_Density', 'Urbanized_Areas_Population_R',
     'White_Non_Hispanic_R', 'HISPANIC_LATINO_R', 'Black_R', 'Asian_R', 'Bt_18_44_R',
     'Bt_45_64_R', 'Over_65_R', 'Male_R',

     'Education_Degree_R', 'Unemployed_R', 'Median_income', 'Rent_to_Income', 'Is_Central', 'Democrat_R',
     'Household_Below_Poverty_R', 'Worked_at_home_R',

     'ALAND', 'Lng', 'Lat', 'STFIPS', 'CTFIPS',

     'POI_count', 'Administration', 'Manufacture', 'Wholesale Trade', 'Real Estate', 'Public',
     'Information', 'Transportation', 'Finance', 'Education', 'Health Care', 'Recreation', 'Retail Trade',
     'Accommodation&Food']

sample_size = len(data_tree)
train_test_ratio = 0.9
dataset = data_tree[predictors + ['raw_visit_counts', 'BGFIPS']].sample(sample_size, random_state=786)

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
     'POI_count': 'POI Count', 'raw_visit_counts': 'Mobility'}, axis=1, inplace=True)

# One hot STFIPS
one_hot = pd.get_dummies(dataset['STFIPS'])
one_hot.columns = ['STFIPS_' + ii for ii in list(one_hot.columns)]
dataset = dataset.drop('STFIPS', axis=1)
dataset = dataset.join(one_hot)
dataset = dataset.drop(['STFIPS_11'], axis=1)
# dataset.sum(axis=0).sort_values()

dataset.loc[:, dataset.dtypes == np.float64] = dataset.loc[:, dataset.dtypes == np.float64].astype(np.float32)
data = dataset.sample(frac=train_test_ratio, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

# Export train and test
data_tree.to_pickle(r'D:\Cross_Nonlinear\Data\visit_data_origin_all.pkl')
data.to_pickle(r'D:\Cross_Nonlinear\Data\visit_data_origin_train.pkl')
data_unseen.to_pickle(r'D:\Cross_Nonlinear\Data\visit_data_origin_test.pkl')

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

# Figure S: Spatially plot
# Read CT Geo data
poly_raw = gpd.GeoDataFrame.from_file(
    r'F:\\SafeGraph\\Open Census Data\\Census Website\\2019\\nhgis0011_shape\\US_blck_grp_2019.shp')
poly_raw['BGFIPS'] = poly_raw['GISJOIN'].str[1:3] + poly_raw['GISJOIN'].str[4:7] + \
                     poly_raw['GISJOIN'].str[8:14] + poly_raw['GISJOIN'].str[14:15]
poly_raw = poly_raw.to_crs(epsg=5070)
poly = poly_raw.merge(dataset[['BGFIPS', 'Mobility']], on='BGFIPS')
# Read State Geo data
poly_state = gpd.GeoDataFrame.from_file(
    r'F:\\SafeGraph\\Open Census Data\\Census Website\\2019\\nhgis0011_shape\\US_state_2019.shp')
poly_state['STFIPS'] = poly_state['GISJOIN'].str[1:3]
poly_state = poly_state[~poly_state['STFIPS'].isin(['02', '15', '60', '66', '69', '72', '78'])].reset_index(drop=True)
poly_state = poly_state.to_crs(epsg=5070)

# Plot Spatially
plot_1 = 'Mobility'
colormap = 'coolwarm'
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1.5]})
poly_state.geometry.boundary.plot(color=None, edgecolor='k', linewidth=0.1, ax=ax[0])
poly.plot(column=plot_1, ax=ax[0], legend=True, scheme='UserDefined', cmap=colormap, linewidth=0, edgecolor='white',
          classification_kwds=dict(bins=[np.quantile(poly[plot_1], 1 / 6), np.quantile(poly[plot_1], 2 / 6),
                                         np.quantile(poly[plot_1], 3 / 6), np.quantile(poly[plot_1], 4 / 6),
                                         np.quantile(poly[plot_1], 5 / 6)]), legend_kwds=dict(frameon=False, ncol=3))
ax[0].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
ax[0].axis('off')
ax[0].set_title('Monthly Population Inflow', pad=-20)
# Reset Legend
patch_col = ax[0].get_legend()
patch_col.set_bbox_to_anchor((1.05, 0.05))
legend_labels = ax[0].get_legend().get_texts()
for bound, legend_label in \
        zip(['< ' + str(round(np.quantile(poly[plot_1], 1 / 6))),
             str(round(np.quantile(poly[plot_1], 1 / 6))) + ' - ' + str(round(np.quantile(poly[plot_1], 2 / 6))),
             str(round(np.quantile(poly[plot_1], 2 / 6))) + ' - ' + str(round(np.quantile(poly[plot_1], 3 / 6))),
             str(round(np.quantile(poly[plot_1], 3 / 6))) + ' - ' + str(round(np.quantile(poly[plot_1], 4 / 6))),
             str(round(np.quantile(poly[plot_1], 4 / 6))) + ' - ' + str(round(np.quantile(poly[plot_1], 5 / 6))),
             '> ' + str(round(np.quantile(poly[plot_1], 5 / 6)))], legend_labels):
    legend_label.set_text(bound)
plt.subplots_adjust(top=0.978, bottom=0.137, left=0.026, right=0.984, hspace=0.2, wspace=0.11)
# Power low
# mpl.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.coolwarm(np.linspace(0, 1, 10)))
sns.set_palette("coolwarm")
bins = range(1, int(dataset[plot_1].max()) + 2, 10)
y_data, x_data = np.histogram(dataset[plot_1], bins=bins, density=True)
x_data = x_data[:-1]
ax[1].loglog(x_data, y_data, basex=10, basey=10, linestyle='None', marker='o', markersize=4, alpha=0.5,
             fillstyle='none')
ax[1].set_ylabel('Probability Density')
ax[1].set_xlabel('Population Inflow')
plt.savefig(r'D:\Cross_Nonlinear\Results\visit_Spatial_plot.png', dpi=1000)
plt.close()

# Before/after power transfer
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
sns.distplot(data_train_y[['Mobility']], ax=ax[0], kde=False)
ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
ax[0].set_ylabel('Frequency')
ax[0].set_xlabel('Population Inflow')
axins = inset_axes(ax[0], 2.5, 2.5, loc=1)
sns.distplot(data_train_y[data_train_y['Mobility'] < np.percentile(data_train_y['Mobility'], 99)][['Mobility']],
             ax=axins, kde=False)
axins.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
axins.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
axins.spines['top'].set_visible(False)
axins.spines['right'].set_visible(False)

sns.distplot(normalized_train_y[['Mobility']], ax=ax[1], kde=False, color='r')
ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax[1].set_ylabel('Frequency')
ax[1].set_xlabel('Population Inflow (box-cox)')
ax[0].add_patch(patches.Rectangle((0, 0), np.percentile(data_train_y['Mobility'], 99), 1.7e5,
                                  linewidth=1, edgecolor='red', facecolor='none'))
plt.tight_layout()
plt.savefig(r'D:\Cross_Nonlinear\Results\visit_Box_cox_transfer.png', dpi=1000)
plt.close()

# Describe
dataset.describe().T[['count', 'mean', 'std', '50%', 'min', 'max']].to_csv(
    r'D:\Cross_Nonlinear\Results\visit_data_all_des.csv')
data_train_y.describe().T[['count', 'mean', 'std', '50%', 'min', 'max']].to_csv(
    r'D:\Cross_Nonlinear\Results\visit_data_trainy_des.csv')
data_test_y.describe().T[['count', 'mean', 'std', '50%', 'min', 'max']].to_csv(
    r'D:\Cross_Nonlinear\Results\visit_data_testy_des.csv')
# Corr
dataset.corr().to_csv(r'D:\Cross_Nonlinear\Results\visit_data_corr.csv')

# Plot corr
fig, ax = plt.subplots(figsize=(11, 9))
plt.rcParams.update({'font.size': 14, 'font.family': "Times New Roman"})
dataset.rename({'Mobility': 'Population Inflow'}, axis=1, inplace=True)
temp = dataset[[
    'Total Population', 'Population Density', 'Urbanized Population', 'White', 'Hispanic', 'African American', 'Asian',
    'Age 18-44', 'Age 45-64', 'Age >65', 'Male', 'High Educated', 'Unemployed Rate', 'Median Income', 'Rent to Income',
    'Central', 'Democrat', 'Poverty', 'Work at home', 'Area', 'Longitude', 'Latitude', 'CTFIPS', 'POI Count',
    'Administration', 'Manufacture', 'Wholesale Trade', 'Real Estate', 'Public', 'Information', 'Transportation',
    'Finance', 'Education', 'Health Care', 'Recreation', 'Retail Trade', 'Accommodation&Food',
    'Population Inflow']].corr()
lg = ['POI Count', 'Administration', 'Manufacture', 'Wholesale Trade', 'Real Estate', 'Public', 'Information',
      'Transportation', 'Finance', 'Education', 'Health Care', 'Recreation', 'Retail Trade', 'Accommodation&Food']
# for jj in lg:
#     temp.loc[(temp[jj] < 1), jj] = temp.loc[(temp[jj] < 1), jj] * 0.7
sns.heatmap(temp, fmt='', cmap='coolwarm', square=True, xticklabels=True, yticklabels=True, linewidths=.5,
            )  # mask=np.triu(temp)
plt.tight_layout()
plt.savefig(r'D:\Cross_Nonlinear\Results\visit_Corr.png', dpi=1200)
