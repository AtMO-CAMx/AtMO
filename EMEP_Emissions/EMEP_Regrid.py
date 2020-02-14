"""
Created on Tue Mar 12 15:01:57 2019
https://gis.stackexchange.com/questions/237162/geopanda-write-geodataframe-into-shapefile-or-spatialite
@author: ylong
Edited by jguo in April 2019. 

# Inputs:
# The script uses as inputs (defined in the Regrid_Namelist.txt file) ;
#   - Initial text files downloaded from EMEP website for each pollutant and each sector you want 
# (the projection of these files is XX);
#   - WRF grid (at least the file must contain XLAT and XLONG variables). 
# I would prefer to extract these variables using ncks commands in order to optimize the CPU time needed to read a NETCDF file.

# Method:
# The script generates polygon shapefiles for both WRF and EMEP grids. 
# Then it processes these grids in order to union/intersect EMEP with WRF cells. 
# Script works with LCC WRF projection only. Note outputs files are shapefiles as well for each NFR sectors and expressed as Mg.
# Following the main steps:
# Step 1: Process WRF grid parameters + extents
# Step 2: Create a wrf polygons shapefile
# step 3: Process EMEP emissions files

# Outputs:
# Outputs are:
#   - One shapefile per sector per pollutant reprojected in WGS84. 
# All attributes have been saved (XLAT, XLONG, geometry, country code, etc.),
#   - One text file with the same attributes as in the shapefile.
"""

# Import packages
import gdal
import os
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from osgeo import ogr
from shapely.ops import transform
from shapely.geometry import Point, Polygon
import re


# Function - Read through files
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def subsearch(occur,fname):
    with open(fname) as f: # Open the file (auto-close it too)
        i=0
        for line in f: # Go through the lines one at a time
            m = re.search(occur, line) # Check each line
            i = i+1
            if m: # If we have a match...
                return i # ...return the value
years = '2015'
pollutants = ['NMVOC','SOx','NH3','PM2_5','PMcoarse','CO','NOx']#,'NMVOC','SOx','NH3','PM2.5','PMcoarse','CO'] # ,'PM10'
grid = ['3.0','15.0','45.0'] # WRF RESOLUTION (KM)
for grd in grid: # For each grid
    for pol in pollutants: # For each pollutant
        if grd == '3.0':
            resol = '03'
        elif grd == '15.0':
            resol = '02'
        else:
            resol = '01'
        #------------------
        # USER PARAMETERS
        root  = 'D:/ATMOVISION/EMEP2WRF'                            # ROOT FOLDER
        pp    = 'D:/ATMOVISION/EMEP_EMISSIONS/'+pol+'_'+years        # EMEP EMISSIONS DIR
        pw    = 'D:/ATMOVISION/WRF/WRFout'                          # NETCDF WRF INPUT FOLDER
        d     = 'wrfout_d'+resol+'_camx_20180115.nc'                # WRF FILE NAME
        lono  = 10                                                  # LONGITUDE OF ORIGIN
        lato  = 52                                                  # LATITUDE OF ORIGIN
        lat1  = 52.0                                                # TRUELAT_1
        lat2  = 52.0                                                # TRUELAT_2
        # END USER'S PARAMETERS
        #------------------

        print('---------------------------------------------------------------------------------------')
        print('----------------------------------')
        print('REGRID EMEP/CIEP TO WRF/CAMx GRID for resolution ', grd, 'km')
        print('----------------------------------')
        print('The script generates polygon shapefiles for both WRF and EMEP grids')
        print('Then it processes the these grids in order to union/intersect EMEP with WRF domains')
        print('Script that works for now with LCC WRF projection only')
        print('Note outputs files are shapefiles as well for each NFR sectors and expressed as Mg/km2 ')
        print('---------------------------------------------------------------------------------------')

        # build output folders
        poutWRF = root+'/OUTPUT/WRF'
        if not os.path.exists(poutWRF):
            os.makedirs(poutWRF)

        poutEMEP = root+'/OUTPUT/EMEP'
        if not os.path.exists(poutEMEP):
            os.makedirs(poutEMEP)

        poutGRID = root+'/OUTPUT/GRID_'+str(grd)+'KM'
        if not os.path.exists(poutGRID):
            os.makedirs(poutGRID)

        poutASCII = root+'/OUTPUT/EMEP_2_WRF_'+str(grd)+'KM_ASCII'
        if not os.path.exists(poutASCII):
            os.makedirs(poutASCII)

        #------------------
        # STEP 1 Process WRF grid parameters + extents
        print('Step 1: Process WRF grid parameters + extents')

        os.chdir(pw)

        wrf_out_file = d # Reads the WRF filename from Regrid_Namelist.txt file (in D:/ATMOVISION/EMEP2WRF/WRF_GRID)
        ds_lon = gdal.Open('NETCDF:"'+wrf_out_file+'":XLONG')
        ds_lat = gdal.Open('NETCDF:"'+wrf_out_file+'":XLAT')
        print('\tFile processed: ',d )
        print('\tTRUELAT1: ', lat1 )
        print('\tTRUELAT2: ', lat2 )
        print('\tLAT ORIGIN: ', lato )
        print('\tLON ORIGIN: ', lono )
        print('\tWRF RESOLUTION (KM)',grd)

        # Draw a basemap projection according to WRF grid parameters 
        m = Basemap(width=4500000,height=3500000,
                    rsphere=(6370000.00,6370000.00),\
                    resolution='l',area_thresh=1000.,projection='lcc',\
                    lat_1=lat1,lat_2=lat2,lat_0=lato,lon_0=lono)

        # Create in terms of meters from the edge of the basemap projection
        x,y = m(ds_lon.ReadAsArray(), ds_lat.ReadAsArray()) # Multiplies the original lat and lon by the basemap projection
        c = np.ones_like(x)
        
        #------------------
        # STEP 2 Create a wrf polygons shapefile
        # Export to shapefile
        print('Step 2: Create a wrf polygons shapefile')
        os.chdir(poutWRF)
        wrfname = "WRF_"+str(grd)+"km"
        geometry = []
        geometry1 = []
        geomwrf = []
        resol_new = float(grd)/2*1000 # create a variable that cuts half of one grid cell --> Converts cross centers to point edges

        # Set index for cutting the border cells from WRF grid to fit the CAMx grid 
        # (cut 5 cells for 15km and 45km. Cut 4 cells for 3km)
        if grd == '15.0' or grd == '45.0':
            ind_start = 6
            ind_end   = 5
            print(range(ind_start-2,np.shape(x)[1]-ind_end-1))
            print(range(ind_start-1,np.shape(x)[2]-ind_end))
            for i in range(ind_start-2,np.shape(x)[1]-ind_end-1): 
                for j in range(ind_start-1,np.shape(x)[2]-ind_end):   
                    # Coordinates in meters
                    p1t = (x[0][i][j]-resol_new,y[0][i][j]-resol_new) # Bottom left
                    p1 = m(p1t[0],p1t[1],inverse=True)
                    p2t = (x[0][i+1][j]-resol_new,y[0][i+1][j]-resol_new) # Bottom right
                    p2 = m(p2t[0],p2t[1],inverse=True)
                    p3t = (x[0][i+1][j+1]-resol_new,y[0][i+1][j+1]-resol_new) # Top right
                    p3 = m(p3t[0],p3t[1],inverse=True)
                    p4t = (x[0][i][j+1]-resol_new,y[0][i][j+1]-resol_new) # Top left
                    p4 = m(p4t[0],p4t[1],inverse=True)
                    
                    box = [p1,p2,p3,p4]
                    geometry.append(Polygon(box))
                    #geomwrf.append(Polygon(box))
        elif grd == '3.0':
            ind_start = 3
            ind_end   = 4
            print(range(ind_start,np.shape(x)[1]-ind_end-1))
            print(range(ind_start+1,np.shape(x)[2]-ind_end))
            for i in range(ind_start,np.shape(x)[1]-ind_end-1): 
                for j in range(ind_start+1,np.shape(x)[2]-ind_end):   
                    # Coordinates in meters
                    p1t = (x[0][i][j]-resol_new,y[0][i][j]-resol_new) # Bottom left
                    p1 = m(p1t[0],p1t[1],inverse=True)
                    p2t = (x[0][i+1][j]-resol_new,y[0][i+1][j]-resol_new) # Bottom right
                    p2 = m(p2t[0],p2t[1],inverse=True)
                    p3t = (x[0][i+1][j+1]-resol_new,y[0][i+1][j+1]-resol_new) # Top right
                    p3 = m(p3t[0],p3t[1],inverse=True)
                    p4t = (x[0][i][j+1]-resol_new,y[0][i][j+1]-resol_new) # Top left
                    p4 = m(p4t[0],p4t[1],inverse=True)

                    box = [p1,p2,p3,p4]
                    geometry.append(Polygon(box))
                    #geomwrf.append(Polygon(box))
                
        driver = ogr.GetDriverByName('Esri Shapefile')
        ds = driver.CreateDataSource(wrfname+'.shp')
        layer = ds.CreateLayer('', None, ogr.wkbPolygon)

        # Add one attribute
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn('XLONG', ogr.OFTReal))
        layer.CreateField(ogr.FieldDefn('XLAT', ogr.OFTReal))
        defn = layer.GetLayerDefn()

        ## If there are multiple geometries, put the "for" loop here
        for i in range(0,len(geometry)):
            # Create a new feature (attribute and geometry)
            feat = ogr.Feature(defn)
            feat.SetField('id', i)        
            poly = geometry[i]
            feat.SetField('XLONG', poly.boundary.coords[1][0]) # assuming pos 1 is the lower left corner
            feat.SetField('XLAT', poly.boundary.coords[1][1])  # assuming pos 1 is the lower left corner   
        
            # Make a geometry, from Shapely object
            geom = ogr.CreateGeometryFromWkb(poly.wkb)
            feat.SetGeometry(geom)
            layer.CreateFeature(feat)
            feat = geom = None  # destroy these 
        # Save and close everything
        ds = layer = feat = geom = feat1 = geom1 = None

        # Define the projection of the WRF shapefile
        f = open(wrfname+'.prj','w')
        f.write('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')
        f.close()

        #------------------
        # STEP 3 Process EMEP emissions files 
        print("Step 3: Process EMEP emissions files from ", pp)
        os.chdir(pp)
        # list files within directory
        #FIL = os.listdir(pp)
        FIL = []
        for file in os.listdir(pp):
            if file.endswith(".txt"):
                FIL.append(os.path.join(file))

        for ff in range(0,len(FIL)):
        #for ff in range(0,1):
            os.chdir(pp)
            L = file_len(FIL[ff])
            f = open(FIL[ff],'r')
            for i in range(0,3):
                header = f.readline()  

            country = []
            year = []
            sector = []
            POL = []
            lonlon = []
            latlat = []
            unit = []
            concon = [] 

            for i in range(0,L-3):
                line = f.readline()
                data = line.split(";")
                country.append(data[0])
                year.append(data[1])
                sector.append(data[2])
                POL.append(data[3])
                lonlon.append(float(data[4]))
                latlat.append(float(data[5]))
                unit.append(data[6])
                concon.append(float(data[7].strip('\n')))         

            f.close()  

            # define an output name of emissions (both EMEP and regridded emissions)
            namout = POL[0]+"_"+years+"_"+sector[0].split(' ')[1]
            print('----------------------------------------')
            print('---->EMEP file name : '+namout)
            print('----------------------------------------')
            os.chdir(poutEMEP)
            temparray = []
            for i in range(0,len(lonlon)):
                temparray.append([lonlon[i],latlat[i],concon[i]])
            numpy_point_array = np.array(temparray)  
            df = pd.DataFrame(numpy_point_array)  
            #geometry = [Point(xyz) for xyz in zip(dfthresh[0], dfthresh[1], dfthresh[2])]
            ds = 0.05
            geometry = []
            geomemis = []  

            for i in range(0,len(lonlon)):    
                p1 = (lonlon[i]-ds,latlat[i]-ds)
                #p1 = memis(p1[0],p1[1],inverse=True)
                p2 = (lonlon[i]+ds,latlat[i]-ds)
                p3 = (lonlon[i]+ds,latlat[i]+ds)
                p4 = (lonlon[i]-ds,latlat[i]+ds)
                box = [p1,p2,p3,p4]
                geometry.append(Polygon(box))
                #geomemis.append(Polygon(box))    

            crs = {'init': u'epsg:4326'}
            os.chdir(poutEMEP)

            #gdf = gpd.GeoDataFrame(dfthresh, crs=crs, geometry = geometry)
            #polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=geometry) 

            driver = ogr.GetDriverByName('Esri Shapefile')
            ds = driver.CreateDataSource(namout+'_EMEP.shp')
            dest_srs = ogr.osr.SpatialReference()
            dest_srs.ImportFromEPSG(4326)
            layer = ds.CreateLayer('', dest_srs, ogr.wkbPolygon)
            # Add one attribute
            layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
            layer.CreateField(ogr.FieldDefn('ISO2',ogr.OFTString))
            layer.CreateField(ogr.FieldDefn('YEAR',ogr.OFTInteger))
            layer.CreateField(ogr.FieldDefn('SECTOR',ogr.OFTString))
            layer.CreateField(ogr.FieldDefn('POLLUTANT',ogr.OFTString))
            layer.CreateField(ogr.FieldDefn('LON',ogr.OFTReal))
            layer.CreateField(ogr.FieldDefn('LAT',ogr.OFTReal))
            layer.CreateField(ogr.FieldDefn('UNIT',ogr.OFTString))
            layer.CreateField(ogr.FieldDefn('EMISSION',ogr.OFTReal))
            layer.CreateField(ogr.FieldDefn('Area km2',ogr.OFTReal)) # do we need that ?

            defn = layer.GetLayerDefn()   

            for i in range(0,len(geometry)):
                # Create a new feature (attribute and geometry)
                feat = ogr.Feature(defn)
                feat.SetField('id', i)    
                poly = geometry[i]
                geometer = transform(m,poly) # geometry in meter projected on the basemap 
                area = geometer.area/1000000 # from m2 to km2  

                feat.SetField('ISO2',country[i])
                feat.SetField('YEAR', int(year[i]))
                feat.SetField('SECTOR', sector[i])
                feat.SetField('POLLUTANT', POL[i])
                feat.SetField('LON', lonlon[i])        
                feat.SetField('LAT', latlat[i])
                feat.SetField('UNIT', 'Mg/km2')
                feat.SetField('EMISSION',concon[i]/area) # unit is Mg/km2
                feat.SetField('Area km2',area)
                # Make a geometry, from Shapely object
                geom = ogr.CreateGeometryFromWkb(poly.wkb)
                feat.SetGeometry(geom)
                layer.CreateFeature(feat)
                feat = geom = None  # destroy these    

            # Save and close everything
            ds = layer = feat = geom = None

            from shapely.strtree import STRtree
            os.chdir(poutEMEP)
            g1 = gpd.GeoDataFrame.from_file(namout+"_EMEP.shp") # emission file
            geomemis = g1['geometry']
            os.chdir(poutWRF)
            g2 = gpd.GeoDataFrame.from_file(wrfname+".shp") # WRF domain 
            geomwrf = g2['geometry']             

            # intersect version 2
            from shapely.ops import cascaded_union
            from rtree import index
            #import time
            import progressbar
            idx = index.Index()
            print("\nPopulate R-tree index with bounds of the emission grid cells")
            bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
            for pos, cell in enumerate(geomemis):
                # assuming cell is a shapely object
                bar.update(i)
                idx.insert(pos, cell.bounds)
            data = []
            bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
            print("\nLoop through each Shapely WRF polygons")
            for i,poly in enumerate(geomwrf):
                bar.update(i)
                acel = 0
                emicel = 0
                datacel = []
                for pos in idx.intersection(poly.bounds):
                    emideg = poly.intersection(geomemis[pos])
                    emimet = transform(m,emideg)
                    datacel.append( {'geometry': poly, 'ISO2': g1.ISO2[pos], 'YEAR': g1.YEAR[pos], 'SECTOR': g1.SECTOR[pos], 
                                     'POLLUTANT': g1.POLLUTANT[pos], 'LON': g1.LON[pos], 'LAT':g1.LAT[pos], 'UNIT': 'Mg',
                                     'EMISSION': g1.EMISSION[pos]*(emimet.area/1000000)} )
                PAYS = []
                for dd in range(0,len(datacel)):
                    PAYS.append(datacel[dd]['ISO2'])
                PAYS = list(set(PAYS)) # list occurence of countries crossing the WRF cell

                # si pas d'intersection, alors on balance une ligne vide pour la cellule - on fixe un test pour éviter de boucler sur toutes les cellules
                # if no intersection, then we balance an empty line for the cell - we set a test to avoid looping on all cells
                if (PAYS == []):
                    #NOC = NO COUNTRY
                    data.append( {'geometry': poly, 'ISO2': 'NOC', 'YEAR': int(year[0]), 'SECTOR': sector[0], 'POLLUTANT':POL[0], 'LCP_X': g2.XLONG[i], 
                                  'LCP_Y': g2.XLAT[i],'UNIT': 'Mg', 'EMISSION': 0.0 } )
                else:

                # On filtre par pays (plusieurs lignes si plusieurs pays croisant la cellule WRF)
                    for dd in PAYS:            
                        datared = list( filter(lambda tata: tata['ISO2'] == dd, datacel) )
                        emiemep = 0
                        CC = datared[0]['ISO2']
                        YY = datared[0]['YEAR']
                        SC = datared[0]['SECTOR']
                        UU = datared[0]['UNIT']
                        PP = datared[0]['POLLUTANT']
                        for ddi in range(0,len(datared)):
                            emiemep = emiemep + datared[ddi]['EMISSION'] # Mg
                        # on ecrit les données aggrégées dans le tableau final
                        data.append( {'geometry': poly, 'ISO2': CC, 'YEAR': YY, 'SECTOR': SC, 'POLLUTANT':PP, 'LCP_X': g2.XLONG[i], 
                                      'LCP_Y': g2.XLAT[i],'UNIT': 'Mg', 'EMISSION': emiemep } )

            print('\nWrite to shapefile')
            os.chdir(poutGRID)
            df = gpd.GeoDataFrame(data,columns=['geometry','ISO2','YEAR','SECTOR','POLLUTANT','LCP_X','LCP_Y','UNIT','EMISSION'])
            df.to_file(namout+'_'+wrfname+'.shp')    
            f = open(namout+'_'+wrfname+'.prj','w')
            f.write('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')
            f.close()

            print('\nWrite to ASCII')
            os.chdir(poutASCII)
            nom = namout+'_'+wrfname+'.txt'
            f = open(nom,'w')
            f.write( '# Content: %s grid 2014 emissions in Mg\n' %(df.POLLUTANT[0]) )
            f.write( '# Origin:  CEIP/EMEP to WRF/CAMx regridding process (v2)\n')
            f.write( '# Format: ISO2;YEAR;SECTOR;POLLUTANT;LCP_X;LCP_Y;UNIT;EMISSION\n')    
            for lin in range(0,len(df)):
                x,y = m(df.LCP_X[lin],df.LCP_Y[lin])
                f.write( '%s;%d;%s;%s;%f;%f;%s;%f\n' % ( df.ISO2[lin],df.YEAR[lin],df.SECTOR[lin],df.POLLUTANT[lin],x,y,df.UNIT[lin],df.EMISSION[lin] ) )
            f.close()
            print('Done')
        print('Success')


