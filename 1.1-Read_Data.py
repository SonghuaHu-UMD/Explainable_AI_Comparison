import pandas as pd
import os
import glob
from functools import reduce
import pickle5 as pickle
import numpy as np

pd.options.mode.chained_assignment = None

# Read POI info: 2021/08-2021/10
range_month_core = ['08', '09', '10']
all_core_set = pd.DataFrame()
for jj in range(0, len(range_month_core)):
    # Change to the deepest subdir
    for dirpaths, dirnames, filenames in os.walk(
            "F:\\SafeGraph\Core Places US (Nov 2020 - Present)\\core_poi\\2021\\" + str(range_month_core[jj])):
        if not dirnames: os.chdir(dirpaths)
    print(dirpaths)
    os.chdir(dirpaths)
    core_set = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "*.gz"))))
    core_set = core_set[core_set['iso_country_code'] == 'US']
    core_set = core_set[['placekey', 'parent_placekey', 'top_category', 'sub_category', 'naics_code', 'latitude',
                         'longitude', 'city', 'region', 'postal_code']]
    all_core_set = all_core_set.append(core_set)
    all_core_set = all_core_set.drop_duplicates(subset=['placekey'])
    all_core_set = all_core_set.dropna(subset=['naics_code']).reset_index(drop=True)
    print('Length of cores: ' + str(len(all_core_set)))
all_core_set.to_pickle(r'D:\Cross_Nonlinear\Data\all_core_set.pkl')

# Read foot traffic: 2021/09/01-10/01: One month, from monthly patterns
all_core_set = pd.read_pickle(r'D:\Cross_Nonlinear\Data\all_core_set.pkl')
month_visit = pd.concat(map(pd.read_csv, glob.glob(os.path.join('D:\Cross_Nonlinear\Data\Monthly_Patterns', "*.gz"))))

# Some data change:
# Change CBG ID data type
month_visit = month_visit.dropna(subset=['poi_cbg'])
month_visit = month_visit[~month_visit['poi_cbg'].astype(str).str.contains('[A-Za-z]')].reset_index(drop=True)
month_visit['poi_cbg'] = month_visit['poi_cbg'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
month_visit['CTFIPS'] = month_visit['poi_cbg'].str[0:5]
# Delete the parent POI
month_visit = month_visit[~month_visit['placekey'].isin(set(month_visit['parent_placekey'].dropna()))]
month_visit = month_visit[['placekey', 'raw_visit_counts', 'poi_cbg', 'CTFIPS']].reset_index(drop=True)
# Merge foot traffic with POI info
month_visit = month_visit.merge(
    all_core_set[['placekey', 'top_category', 'sub_category', 'naics_code', 'latitude', 'longitude']], on='placekey')
month_visit['NAICS'] = month_visit['naics_code'].astype(int).astype(str).str[0:2]
month_visit.loc[month_visit['NAICS'].isin(['00', '06']), 'NAICS'] = '00'
month_visit['NAICS'].value_counts()
# month_visit.isnull().sum()
# Assign a name to NAICS CODE
NAICS_code = pd.read_excel(r'D:\OD_Predict\Results\2-6 digit_2017_Codes.xlsx', engine='openpyxl')
NAICS_code = NAICS_code[['2017 NAICS US   Code', '2017 NAICS US Title']]
NAICS_code = NAICS_code.dropna()
NAICS_code = NAICS_code[NAICS_code['2017 NAICS US   Code'].astype(str).str.len() <= 2].reset_index(drop=True)
NAICS_code.columns = ['NAICS', 'Description']
NAICS_code = NAICS_code.append(pd.DataFrame(
    list(zip(['44', '45', '48', '49', '31', '32', '33', '00'],
             ['Retail Trad', 'Retail Trade', 'Transportation and Warehousing', 'Transportation and Warehousing',
              'Manufacturing', 'Manufacturing', 'Manufacturing', 'Unknown'])), columns=['NAICS', 'Description']),
    ignore_index=True)
NAICS_code['NAICS'] = NAICS_code['NAICS'].astype(int).astype(str).apply(lambda x: x.zfill(2))
month_visit = month_visit.merge(NAICS_code, on='NAICS')
# Output
month_visit.to_pickle(r'D:\Cross_Nonlinear\Data\month_visit_POI_09_10.pkl')

# Groupby CBG
with open(r'D:\Cross_Nonlinear\Data\month_visit_POI_09_10.pkl', "rb") as fh: month_visit = pickle.load(fh)
month_visit.rename({'poi_cbg': 'BGFIPS'}, axis=1, inplace=True)
month_visit_count = month_visit.groupby(['BGFIPS']).sum()['raw_visit_counts'].reset_index()

# Merge with CBG features
# Merge with place2CBG to calculate POI counts in each CBG
# A map between CBG and POI key
place2CBG = pd.read_csv(r'D:\Cross_Nonlinear\Data\place2CBG.csv')
place2CBG = place2CBG.merge(month_visit[['placekey', 'BGFIPS']], on='placekey', how='outer')
place2CBG['CBGFIPS'] = place2CBG['CBGFIPS'].fillna(0)
place2CBG['CBGFIPS'] = place2CBG['CBGFIPS'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
place2CBG.loc[(place2CBG['CBGFIPS'] != place2CBG['BGFIPS']) & (~place2CBG['BGFIPS'].isnull()), 'CBGFIPS'] = \
    place2CBG.loc[(place2CBG['CBGFIPS'] != place2CBG['BGFIPS']) & (~place2CBG['BGFIPS'].isnull()), 'BGFIPS']
place2CBG = place2CBG[place2CBG['CBGFIPS'] != '000000000000'].reset_index(drop=True)
place2CBG = place2CBG[['placekey', 'CBGFIPS']]
place2CBG.rename({'CBGFIPS': 'BGFIPS'}, axis=1, inplace=True)

# Merge with POI info
all_core_set = pd.read_pickle(r'D:\Cross_Nonlinear\Data\all_core_set.pkl')
all_core_set_nas = all_core_set[['placekey', 'naics_code']]
all_core_set_nas['NAICS'] = all_core_set_nas['naics_code'].astype(int).astype(str).str[0:2]
all_core_set_nas = all_core_set_nas.merge(place2CBG, on='placekey')
cbg_poi = all_core_set_nas.groupby(['BGFIPS', 'NAICS']).size().unstack(fill_value=0).reset_index()

# Merge with CBG features
CBG_Features = pd.read_csv(r'D:\COVID19-Socio\Data\CBG_COVID_19.csv', index_col=0)
CBG_Features['BGFIPS'] = CBG_Features['BGFIPS'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
# Add some new features
CBG_Features['Agriculture_Mining_R'] = \
    CBG_Features['Agriculture_R'] + CBG_Features['Mining_R'] + CBG_Features['Construction_R']
CBG_Features['Transportation_Utilities_R'] = \
    CBG_Features['Transportation_R'] + CBG_Features['Utilities_R']
CBG_Features['Retail_Wholesale_R'] = CBG_Features['Retail_R'] + CBG_Features['Wholesale_R']
CBG_Features['Administrative_Management_R'] = CBG_Features['Management_R'] + CBG_Features['Administrative_R']
CBG_Features['Accommodation_food_arts_R'] = CBG_Features['Accommodation_food_R'] + CBG_Features['Arts_R']
CBG_Features['Indian_Others_R'] = 100 - (CBG_Features['Asian_R'] + CBG_Features['White_R'] + CBG_Features['Black_R'])
CBG_Features = CBG_Features[
    ['BGFIPS', 'Total_Population', 'LUM_Race', 'Median_income', 'Rent_to_Income', 'GINI', 'Agriculture_Mining_R',
     'Manufacturing_R', 'Retail_Wholesale_R', 'Transportation_Utilities_R', 'Information_R', 'Finance_R',
     'Real_estate_R', 'Scientific_R', 'Administrative_Management_R', 'Educational_R', 'Health_care_R',
     'Accommodation_food_arts_R', 'ALAND', 'Lng', 'Lat', 'Is_Central', 'STUSPS', 'Democrat_R', 'Republican_R',
     'Urbanized_Areas_Population_R', 'Urban_Clusters_Population_R', 'Rural_Population_R', 'No_Insurance_R',
     'Household_Below_Poverty_R', 'Time_Lower_10_R', 'Time_10_30_R', 'Time_30_60_R', 'Time_greater_60_R',
     'Drive_alone_R', 'Carpool_R', 'Public_Transit_R', 'Bicycle_R', 'Walk_R', 'Taxicab_R', 'Worked_at_home_R',
     'HISPANIC_LATINO_R', 'White_R', 'Black_R', 'Indian_Others_R', 'Asian_R', 'Under_18_R', 'Bt_18_44_R', 'Bt_45_64_R',
     'Over_65_R', 'Male_R', 'White_Non_Hispanic_R', 'White_Hispanic_R', 'Population_Density', 'Education_Degree_R',
     'Unemployed_R']]
CBG_XY = reduce(lambda left, right: pd.merge(left, right, on=['BGFIPS'], how='outer'),
                [CBG_Features, month_visit_count, cbg_poi])
CBG_XY['STFIPS'] = CBG_XY['BGFIPS'].str[0:2]
CBG_XY = CBG_XY[~CBG_XY['STFIPS'].isin(['02', '15', '60', '66', '69', '72', '78'])].reset_index(drop=True)
CBG_XY = CBG_XY.dropna(subset=['Total_Population']).reset_index(drop=True)
CBG_XY = CBG_XY.dropna(subset=['raw_visit_counts']).reset_index(drop=True)
CBG_XY['POI_count'] = CBG_XY[
    ['11', '21', '22', '23', '31', '32', '33', '42', '44', '45', '48', '49', '51', '52', '53', '54', '55', '56',
     '61', '62', '71', '72', '81', '92']].sum(axis=1)

# Merge with neighbor visits
neighbor = pd.read_csv(
    r'F:\SafeGraph\Neighbourhood Patterns\neighborhood-patterns\2021\10\06\release-2021-07-01\neighborhood_patterns\y=2021\m=9\part-00000-tid-7974919917562241541-ac08e62a-2452-491a-b6c0-ccdb2a7c566d-391686-1-c000.csv.gz')
neighbor = neighbor[['area', 'raw_stop_counts']]
neighbor['BGFIPS'] = neighbor['area'].astype(str).apply(lambda x: x.zfill(12))
CBG_XY = CBG_XY.merge(neighbor[['BGFIPS', 'raw_stop_counts']], on='BGFIPS', how='left')
CBG_XY = CBG_XY.fillna(0)
# CBG_XY.isnull().sum().to_csv('temp_cnon.csv')
CBG_XY.to_pickle(r'D:\Cross_Nonlinear\Data\CBG_XY_09_10.pkl')
CBG_XY.corr().to_csv('D:\Cross_Nonlinear\Results\corr.csv')

# More processing

