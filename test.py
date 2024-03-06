import pandas as pd

df = pd.read_excel('/content/drive/MyDrive/SatelliteImages/FloodPlainDevelopment.xlsx')

test_df = df[df['Train/Test']=='Test']

bhalswa_df = df[df['Train/Test']=='Bhalswa']

train_df = df[df['Train/Test']=='Train']

#Testing
test_water = test_df[test_df['Type'] == 'Water']
test_concrete = test_df[test_df['Type'] == 'Concrete']
test_vegetation = test_df[test_df['Type'] == 'Vegetation']
test_silt = test_df[test_df['Type'] == 'Silt']
len(test_water), len(test_concrete), len(test_vegetation), len(test_silt)

water = train_df[train_df['Type'] == 'Water']
concrete = train_df[train_df['Type'] == 'Concrete']
vegetation = train_df[train_df['Type'] == 'Vegetation']
silt = train_df[train_df['Type'] == 'Silt']
len(water), len(concrete), len(vegetation), len(silt)

#Geemap
# reminder that if you are installing libraries in a Google Colab instance you will be prompted to restart your kernal
from IPython.display import Image
import os

try:
    import geemap, ee
except ModuleNotFoundError:
    if 'google.colab' in str(get_ipython()):
        print("package not found, installing w/ pip in Google Colab...")
        !pip install geemap
    else:
        print("package not found, installing w/ conda...")
        !conda install mamba -c conda-forge -y
        !mamba install geemap -c conda-forge -y
    import geemap, ee

try:
        ee.Initialize()
except Exception as e:
        ee.Authenticate()
        ee.Initialize()

#Visualise multiple bands
from glob import glob
!pip install earthpy
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import os

geemap.Report()

u_lon, u_lat = bhalswa_df['Longitude'][1379], bhalswa_df['Latitude'][1379]
from_date,to_date  = bhalswa_df['FromDate'][1379], bhalswa_df['ToDate'][1379]
u_poi = ee.Geometry.Point(u_lon, u_lat)
# get India boundary
#ee.Geometry.Rectangle([minlon, minlat, maxlon, maxlat])
#aoi = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME','India')).geometry()
roi = u_poi.buffer(50000)

# Sentinel-2 image filtered on 2019 and on India
se2 = ee.ImageCollection('COPERNICUS/S2').filterDate(from_date,to_date).filterBounds(roi).median().divide(10000)


rgbViz = {"min":0.0, "max":0.3}

# initialize our map
map1 = geemap.Map()
map1.centerObject(roi, 7)
map1.addLayer(se2.clip(roi), rgbViz, "S2")

vizParams = {
  'bands': ['B5', 'B4', 'B3'],
  'min': 0,
  'max': 0.5,
  'gamma': [0.95, 1.1, 1]
}

# Center the map and display the image.
map1.addLayer(se2.clip(roi), vizParams, 'false color composite')
#fdi = l[7] - (l[5] + 1.068*(l[11]-l[5]))
#ndvi = (l[7]-l[3])/(l[7]+l[3])
NIR = se2.select('B8')
Red = se2.select('B4')
rgbViz1 = {"min":0, "max":2}
#pi = NIR.divide(NIR.add(Red))

#pi_masked = pi.updateMask(pi.lte(1))
#map1.addLayer(pi_masked, {"min": 0, "max": 1, "palette": ['blue', 'green', 'red']}, "Plastic Index")
map1.addLayerControl()
map1

index = 0
for ind in bhalswa_df.index:
    u_lon, u_lat = bhalswa_df['Longitude'][ind], bhalswa_df['Latitude'][ind]
    from_date,to_date  = "2022-06-01", "2022-06-30"
    u_poi = ee.Geometry.Point(u_lon, u_lat)
    roi = u_poi.buffer(7)
    se2 = ee.ImageCollection('COPERNICUS/S2').filterDate(from_date,to_date).filterBounds(roi).median().divide(10000)

    task = ee.batch.Export.image.toDrive(image=se2.clip(roi),
                                     description='Sentinel Floodplain Images',
                                     scale=30,
                                     region=roi,
                                     fileNamePrefix='Bhalswa_Test'+str(index),
                                     crs='EPSG:4326',
                                     fileFormat='GeoTIFF')
    task.start()
    print(index)
    index+=1

task.status()
