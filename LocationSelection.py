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

u_lat, u_lon = 28.744426013903773, 77.17177339686545
u_poi = ee.Geometry.Point(u_lon, u_lat)
# get India boundary
#aoi = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME','India')).geometry()
roi = u_poi.buffer(10000)

# Landsat
lc = ee.ImageCollection('LANDSAT/GLS1975').filterBounds(roi).median().divide(10000)


rgbViz = {"min":0.0, "max":0.3}

# initialize our map
map1 = geemap.Map()
map1.centerObject(roi, 7)
map1.addLayer(lc.clip(roi), rgbViz, "LANDSAT Old")
#fdi = l[7] - (l[5] + 1.068*(l[11]-l[5]))
#ndvi = (l[7]-l[3])/(l[7]+l[3])

vizParams = {
  'bands': ['30', '20', '10'],
  'min': 0,
  'max': 0.02,
  'gamma': [0.95, 1.1, 1]
}

# Center the map and display the image.
map1.addLayer(lc.clip(roi), vizParams, 'false color composite Old')

NIR = lc.select('30')
#Red = lc.select('20')
Green = lc.select('10')

ndwi = NIR.subtract(Green).divide(NIR.add(Green))
#ndvi = NIR.subtract(Red).divide(NIR.add(Red))
palette = [
    'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
    '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
    '012E01', '011D01', '011301']

map1.addLayer(ndwi.clip(roi), {'palette': palette}, "NDwI Old")
ndwi_masked = ndwi.updateMask(ndwi.lte(0))
map1.addLayer(ndwi_masked.clip(roi), {'palette': palette}, "NDwI Old masked")
map1.addLayerControl()
map1


se2_1 = ee.ImageCollection('COPERNICUS/S2').filterDate("2022-02-01","2022-02-28").filterBounds(roi).median().divide(10000)
rgbViz = {"min":0.0, "max":0.3}

map1.addLayer(se2_1.clip(roi), rgbViz, "S2 new")

vizParams = {
  'bands': ['B5', 'B4', 'B3'],
  'min': 0,
  'max': 0.5,
  'gamma': [0.95, 1.1, 1]
}

# Center the map and display the image.
map1.addLayer(se2_1.clip(roi), vizParams, 'false color composite new')

NIR_1 = se2_1.select('B8')
Red_1 = se2_1.select('B4')
Green_1 = se2_1.select('B2')
SWIR_1 = se2_1.select('B11')

ndvi_1 = NIR_1.subtract(Red_1).divide(NIR_1.add(Red_1))
ndvi_green_1 = NIR_1.subtract(Green_1).divide(NIR_1.add(Green_1))
#ndwi_1 = NIR_1.subtract(SWIR_1).divide(NIR_1.add(SWIR_1))
palette = [
    'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
    '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
    '012E01', '011D01', '011301']
#map1.addLayer(ndwi_1.clip(roi), {'palette': palette}, "NDWI New")
map1.addLayer(ndvi_green_1.clip(roi), {'palette': palette}, "NDVI-Green New")
map1.addLayer(ndvi_1.clip(roi), {'palette': palette}, "NDVI New")

ndwi_masked_1 = ndvi_green_1.updateMask(ndvi_green_1.gte(0))
ndwi_masked_1 = ndwi_masked_1.updateMask(ndwi.lte(0))

#se2_masked_1 = se2_1.updateMask(ndvi_masked_1.gte(0))
#map1.addLayer(se2_masked_1.clip(roi), rgbViz, "Flood Plain Development")

map1.addLayer(ndwi_masked_1.clip(roi),{min: 0.1, max: 0.6, 'palette': ['blue', 'red', 'green']} , "Flood Plain Development Type")
#map1.addLayerControl()
map1

features = map1.user_rois.getInfo()

for i in range(len(features['features'])):
  print(features['features'][i]['geometry']['coordinates'])
