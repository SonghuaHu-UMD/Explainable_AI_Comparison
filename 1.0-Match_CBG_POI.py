import csv
import datetime
from shapely.geometry import Polygon, Point
import shapefile
from osgeo import ogr
import pandas as pd
import os
import glob
import geopandas as gpd
import pickle5 as pickle

# # Project change
# US_county_2019 = gpd.read_file(r"F:\SafeGraph\Open Census Data\Census Website\2019\nhgis0011_shape\US_county_2019.shp")
# US_county_2019 = US_county_2019.to_crs('epsg:4326')
# US_county_2019.to_file(r'F:\SafeGraph\Open Census Data\Census Website\2019\nhgis0011_shape\US_county_2019_84.shp')
#
# US_cbg_2019 = gpd.read_file(r"F:\SafeGraph\Open Census Data\Census Website\2019\nhgis0011_shape\US_blck_grp_2019.shp")
# US_cbg_2019 = US_cbg_2019.to_crs('epsg:4326')
# US_cbg_2019.to_file(r'F:\SafeGraph\Open Census Data\Census Website\2019\nhgis0011_shape\US_blck_grp_2019_84.shp')

# Load County
sf = shapefile.Reader(r"F:\Project\QAQC\FAF\tl_2017_us_county.shp")
shapeRecs = sf.shapeRecords()
for x in range(len(shapeRecs)):
    if shapeRecs[x].shape.shapeType == 5:
        shapeRecs[x].poly = Polygon(shapeRecs[x].shape.points)
    else:
        print("Unexpected shape type:", shapeRecs[x].shape.shapeType)
        exit(0)

# Load CBG
ctyToCBG = {}  # county FIPS to list of (CBGFIPS, bb, feature)
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(r"F:\SafeGraph\Open Census Data\Census Website\2019\nhgis0011_shape\US_blck_grp_2019_84.shp")
layer = dataSource.GetLayer()
for x in layer:
    cty = x.GetFieldAsString("STATEFP") + x.GetFieldAsString("COUNTYFP")
    FIPS = x.GetFieldAsString("STATEFP") + x.GetFieldAsString("COUNTYFP") + x.GetFieldAsString(
        "TRACTCE") + x.GetFieldAsString("BLKGRPCE")
    bb = x.GetGeometryRef().GetEnvelope()
    if not cty in ctyToCBG:
        ctyToCBG[cty] = []
    ctyToCBG[cty].append((FIPS, bb, x))

# Read POI and match
os.chdir(r'D:\Cross_Nonlinear\Data')
all_core_set = pd.read_pickle(r'D:\Cross_Nonlinear\Data\all_core_set.pkl')
# Only merge those without cbg info
with open(r'D:\Cross_Nonlinear\Data\month_visit_POI_09_10.pkl', "rb") as fh: month_visit = pickle.load(fh)
all_core_set = all_core_set[~all_core_set['placekey'].isin(month_visit['placekey'])].reset_index(drop=True)

start_t = datetime.datetime.now()
with open('place2CBG_subset.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["placekey", "stateFIPS", "countyFIPS", "CBGFIPS"])
    for index, line in all_core_set.iterrows():
        lat = float(line["latitude"])
        lon = float(line["longitude"])
        county = None
        for x in shapeRecs:
            bb = x.shape.bbox
            if lon >= bb[0] and lon <= bb[2] and lat >= bb[1] and lat <= bb[3] and x.poly.contains(Point(lon, lat)):
                county = x
                break
        if county is not None:
            pt = ogr.Geometry(ogr.wkbPoint)
            pt.AddPoint(lon, lat)
            cbgFIPS = None
            rec = county.record.as_dict()
            if rec["GEOID"] in ctyToCBG:
                for cbg, bb, feat in ctyToCBG[rec["GEOID"]]:
                    if lon >= bb[0] and lon <= bb[1] and lat >= bb[2] and lat <= bb[3] and \
                            feat.GetGeometryRef().Contains(pt):
                        cbgFIPS = cbg
                        break
            writer.writerow([line["placekey"], rec["STATEFP"], rec["GEOID"], cbgFIPS])
        else:
            writer.writerow([line["placekey"], None, None, None])
print('%s points to CBG: %s' % (len(all_core_set), datetime.datetime.now() - start_t))

# place2CBG = pd.read_csv(r'D:\Cross_Nonlinear\Data\place2CBG.csv')
# month_visit.merge(place2CBG, on='placekey').to_csv(r'D:\Cross_Nonlinear\Data\temp.csv')
