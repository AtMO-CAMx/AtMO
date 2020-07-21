#!/usr/local/Anaconda3-5.0/bin/python
import sys
import pandas as pd
import numpy as np
import csv
import os
import xarray as xr
import datetime as dt
from collections import OrderedDict
from netCDF4 import Dataset

# set NetCDF output format here
outfmt = 'NETCDF4'
# Root folder
proot = "/disk4/ATMOVISION/EMEP2WRF/" 
# Read in speciation file
SPEC = pd.read_csv(proot+'/TEMPORAL_ALLOCATION/NFR2CAMx_speciation.csv',sep='\t', index_col = "NFR14")
# GNFR plume height file
height_path = '/disk4/ATMOVISION/EMEP2WRF/TEMPORAL_ALLOCATION/Height_NFR.csv'
# Input folder
pin = "/disk4/ATMOVISION/EMEP2WRF/OUTPUT/2015" 
# Out folder
pout = "/disk4/ATMOVISION/EMEP2WRF/OUTPUT/2015/SPECIATED/"+outfmt+'/'
year = '2015'
# Pollutants in the input files
pollutants = ['CO','NOx','NMVOC','SOx','NH3','PM2_5','PMcoarse','CO'] 
grid = ['3.0','15.0','45.0'] # WRF RESOLUTION (KM)
# mole weight of species, to convert g/hr to mol/hr
molwt = {'NO':30,'NO2':46,'HONO':47,'CO':28,'SO2':64,'SULF':96,'NH3':17,'ETHA':30,'PRPA':44,'PAR':14.3,'ETH':28,'OLE':28,'IOLE':56,'TERP':136,'BENZ':78,'TOL':92,'XYL':106,'ETHY':26,'MEOH':32,'ETOH':46,'FORM':30,'ALD2':44,'ALDX':44,'ACET':58,'KET':28,'NR':44,'IVOA':212}

PtSource = ['A_PublicPower','B_Industry','G_Shipping','H_Aviation','J_Waste']

###### get layering factors #########################################
height = pd.read_csv(height_path)
height['height'] = height['CAMx height (m)'].str.split(' - ')
height.set_index('CEIP NFR code',inplace=True)

LayerH = [0,54.83,133.73,237.24,366.08,529.46,737.54,984.68,1392.20,1818.60,2265.56,2735.07,
          3613.12,4491.27,5368.32,6243.65,7988.52,9734.51,12456.71,14387.52,16313.36]

layers = np.arange(0,21)

#for sector in PtSource:
layerDiv = pd.DataFrame(index=PtSource,columns=layers[1:])

# emission fraction within each layer
for sector in PtSource:
    top = int(height.loc[sector,'height'][1])
    bot = int(height.loc[sector,'height'][0])
    for l in layers[1:]:
        h = LayerH[l]
        h0 = LayerH[l-1]
        if (h > bot > h0) & (top > h):
            layerDiv.loc[sector,l] = (h-bot)/(top-bot)
        elif (bot < h0) & (top > h):
            layerDiv.loc[sector,l] = (h-h0)/(top-bot)
        elif (bot < h0) & (h0 < top < h):
            layerDiv.loc[sector,l] = (top-h0)/(top-bot)
        else:
            layerDiv.loc[sector,l] = 0
################################################################
# get date indices            
Months = (pd.date_range('2015/01/01 00:00:00', 
                      freq='M', periods=12)-pd.offsets.MonthBegin(1)).strftime('%Y%m%d').tolist()
dates = []
for month in Months:
    Mon = pd.date_range(month, freq='W-MON', periods=2)
    days = pd.date_range(Mon[1],freq='D',periods=7).strftime('%Y%m%d').tolist()
    dates+=days
  
for grd in grid: # For each grid
    if grd == '3.0':
        print("grid = ", grd)
        i = 212
        j = 167
        XORIG = -570000. 
        YORIG = -666900.
    elif grd == '15.0':
        print("grid = ", grd)
        i = 86
        j = 71
        XORIG = -1167000.
        YORIG = -1128900.
    elif grd == '45.0':
        print("grid = ", grd)
        i = 69
        j = 58
        XORIG = -2097000.
        YORIG = -1833900.
    for day in dates:
        print(day)
        jday = dt.datetime.strptime(day,'%Y%m%d').strftime('%Y%j')
        starttime = day+'00'
        endtime = day+'23'

        # timestamps to be used in the out NetCDF
        datetimes = pd.date_range(dt.datetime.strptime(starttime,'%Y%m%d%H'), freq='H', periods=24)
        JDAY_S = [x.strftime('%Y%j') for x in datetimes]
        HMS_S = [x.strftime('%H0000') for x in datetimes]
        JDAY_E = JDAY_S
        HMS_E = [(x+dt.timedelta(hours=1)).strftime("%H0000") for x in datetimes]
        HMS_E[-1] = '240000'
        
        # pull out sector name and pollutant names
        for sector in PtSource: # one file for each day and sector and all pollutants
            print(sector)
            specs = []
            secname = sector[2:]
            data_grid = []
            for pol in pollutants: # For each pollutant
                if pol == 'PM2_5': pol_ = 'PM25'
                else: pol_ = pol
############################## Data speciation, cropping and reshaping #################################################
                # Read in file (hourly, timeshifted)
                gridid = "WRF_"+grd+"km"
                inCSV= pin+'/CSV_'+str(gridid)+'/'
                #inname = "TEMP_"+sector+"_"+year+"_"+pol+"_"+gridid
                inname = pol+"_"+year+"_"+sector+"_"+gridid
                infile = pd.read_csv(inCSV+inname+".csv")
                # Factors for converting NFR to CAMx
                if sector == 'C_OtherStatComb':
                    sector = 'C_OtherStationaryComb'
                # Pull out values for a select sector and pollutant
                if not isinstance(SPEC[sector][pol_],pd.Series):
                    val = [SPEC[sector][pol_]]
                else: val = SPEC[sector][pol_].values

                # CAMx pollutant
                row = SPEC.loc[pol_] # Pull out the rows for the pollutant we are looking at
                if not isinstance(row['Unnamed: 14'],pd.Series):
                    spec = [row['Unnamed: 14']]
                else: spec = row['Unnamed: 14'].tolist()
                specs+=spec # all species
                
                # Multiply each row by a speciation factor
                
                for fac in range(0,len(val)):
                    infile1 = infile
                    infile2 = infile1.loc[:,starttime:endtime].apply(lambda x: x*val[fac]/100) # species emission = pol emission * factor

                    infile2['ISO2']        = infile1['ISO2']
                    infile2['YEAR']        = infile1['YEAR']
                    infile2['SECTOR']      = infile1['SECTOR']
                    infile2['LON']         = infile1['LCP_X']
                    infile2['LAT']         = infile1['LCP_Y']
                    infile2['POLLUTANT']   = infile1['POLLUTANT']
                    infile2['EMISSION']    = infile1['EMISSION']
                    cols = list(infile2.columns.values) # List of all columns

                    # Convert to gridded format
                    sum_data1 = infile2
                    sum_data = sum_data1.loc[:,starttime:endtime]
                    # Append each hour's grid to the next hour
                    temp1 = []
                    for k in range(len(sum_data.columns)):
                        temp = sum_data.iloc[:,k].values
                        temp[np.isnan(temp)] = 0
                        temp = np.flipud(np.reshape(temp,(j,i)))
                        temp_layers = []
                        # append each layer's emission to the next layer
                        for lay in layers[1:]:
                            temp_l = temp * layerDiv.loc[sector,lay]
                            temp_layers.append(temp_l)
                        temp1.append(temp_layers)
                        data_grid1 = np.array(temp1)
                    # add buffer cells to 3km and 15km domains
                    if grd in {'3.0','15.0'}:
                        data_grid1[:,:,0,:] = 0
                        data_grid1[:,:,-1,:] = 0
                        data_grid1[:,:,:,0] = 0
                        data_grid1[:,:,:,-1] = 0
                    else: pass
                    data_grid.append(data_grid1)
    
######################################### Create netCDF #################################################################
### Create netCDF4 https://stackoverflow.com/questions/34923646/how-to-create-a-netcdf-file-with-python-netcdf4
#########################################################################################################################
            pout_grd = pout+grd+'km'
            pout_sec = pout_grd+"/"+secname+'_test'
            if not os.path.exists(pout_grd): os.makedirs(pout_grd)
            if not os.path.exists(pout_sec): os.makedirs(pout_sec)
            root_grp = Dataset(pout_grd+"/"+secname+"_test/CAMx_"+sector+"_Compressed_3D_"+day+"_"+grd+"km_netcdf4.nc", 'w', format=outfmt)
            root_grp.description = "Gridded 3D emissions for CAMx "+sector+" for "+day

            ndim = np.shape(data_grid1) # Size of the matrix ndim*ndim
            tdim = ndim[0]
            zdim = ndim[1]

            # Set dimensions
            root_grp.createDimension('TSTEP', tdim)
            root_grp.createDimension('DATE-TIME', 2)
            root_grp.createDimension('LAY', zdim)
            root_grp.createDimension('COL', i)
            root_grp.createDimension('ROW', j)
            root_grp.createDimension('VAR', len(specs))
            
            # Set variables - NOT DONE
            X           = root_grp.createVariable('X', 'f8', ('COL',),zlib=True)
            Y           = root_grp.createVariable('Y', 'f8', ('ROW',),zlib=True)
            TFLAG       = root_grp.createVariable('TFLAG', 'i4', ('TSTEP','VAR','DATE-TIME',),zlib=True)
            ETFLAG      = root_grp.createVariable('ETFLAG', 'i4', ('TSTEP','VAR','DATE-TIME',),zlib=True)
            longitude   = root_grp.createVariable('longitude', 'f8', ('ROW','COL',),zlib=True)
            latitude    = root_grp.createVariable('latitude', 'f8', ('ROW','COL',),zlib=True)

            spec_var = []
            for s in range(0,len(specs)):
                if specs[s] is np.nan: specs[s] = 'NA'
                spec_var.append(root_grp.createVariable(specs[s], 'f4', ('TSTEP','LAY','ROW','COL',),zlib=True))
                # Add local attributes to variables
                if specs[s] in molwt:
                    spec_var[s].units = "mol hr-1"
                else:
                    spec_var[s].units = "g hr-1"
                spec_var[s].long_name = specs[s]
                spec_var[s].var_desc = specs[s] + " emissions"
                spec_var[s].coordinates = "latitude longitude" 
                
            # Add local attributes to variables 
            X.units = "km"
            X.long_name = "X coordinate"
            X.var_desc = "X cartesian distance from projection origin"
            
            Y.units = "km"
            Y.long_name = "Y coordinate"
            Y.var_desc = "Y cartesian distance from projection origin"
            
            TFLAG.units = "YYYYDDD,HHMMSS" ;
            TFLAG.long_name = "Start time flag" ;
            TFLAG.var_desc = "Timestep start date and time" ;
            
            ETFLAG.units = "YYYYDDD,HHMMSS" ;
            ETFLAG.long_name = "End time flag" ;
            ETFLAG.var_desc = "Timestep end date and time" ;
            
            longitude.units = "Degrees east" ;
            longitude.long_name = "Longitude" ;
            longitude.var_desc = "Longitude degrees east"
            longitude.coordinates = "latitude longitude" ;

            latitude.units = "Degrees north" ;
            latitude.long_name = "Latitude" ;
            latitude.var_desc = "Latitude degrees north" ;
            latitude.coordinates = "latitude longitude" ;
            
            # data
            DT_S = [val for val in [[int(x),int(y)] for x,y in zip(JDAY_S,HMS_S)] for _ in range(len(specs))]
            DT_E = [val for val in [[int(x),int(y)] for x,y in zip(JDAY_E,HMS_E)] for _ in range(len(specs))]
       
            TFLAG[:]  = DT_S
            ETFLAG[:] = DT_E
            latitude[:] = np.flipud(np.reshape(sum_data1[['LAT']].values,(j,i)))
            longitude[:] = np.flipud(np.reshape(sum_data1[['LON']].values,(j,i)))
            x_range =  np.arange(0, float(grd)*i, float(grd))
            y_range =  np.arange(0, float(grd)*j, float(grd))
            X[:] = x_range
            Y[:] = y_range
            for s in range(len(spec_var)): 
                if specs[s] in molwt:
                    spec_var[s][:,:,:,:] = data_grid[s]/molwt[specs[s]]
                else:
                    spec_var[s][:,:,:,:] = data_grid[s]
            
            # Add global attributes
            root_grp.SDATE = np.int32(jday)
            root_grp.SDATEC = np.int32(day)
            root_grp.STIME = np.int32(0)
            root_grp.TSTEP = np.int32(10000) # [HHMMSS]
            root_grp.NSTEPS = np.int32(tdim)
            root_grp.NCOLS = np.int32(i)
            root_grp.NROWS = np.int32(j)
            root_grp.NLAYS = np.int32(zdim)
            root_grp.NVARS = np.int32(len(specs))
            root_grp.P_ALP = 52.
            root_grp.P_BET = 52.
            root_grp.P_GAM = 10.
            root_grp.XCENT = 10.
            root_grp.YCENT = 52.
            root_grp.XORIG = XORIG 
            root_grp.YORIG = YORIG
            root_grp.XCELL = float(grd) *1000 # Domain in km
            root_grp.YCELL = float(grd) *1000
            root_grp.IUTM  = np.int32(0) # ?
            root_grp.CPROJ = np.int32(2) # ?
            root_grp.ITZON = np.int32(0) # ?
            root_grp.VAR_LIST = '      '.join(specs)
            root_grp.NAME = "EMISSIONS"
            root_grp.history = "Created " + dt.datetime.today().strftime("%d%m%Y")
            root_grp.FILEDESC = "EMISSIONS"
            root_grp.FTYPE = np.int32(1) # ? [IO-API file type = CUSTOM3]
            root_grp.CDATE = np.int32(dt.datetime.today().strftime("%Y%j")) # [IO-API file creation date]
            root_grp.CTIME = np.int32(dt.datetime.today().strftime("%k%M%S")) # [IO-API file creation time]
            root_grp.WDATE = np.int32(dt.datetime.today().strftime("%Y%j")) # [IO-API file write date]
            root_grp.WTIME = np.int32(dt.datetime.today().strftime("%k%M%S"))  # [IO-API file write time]
            root_grp.GDTYP = np.int32(2) # ? [IO-API map projection]
            root_grp.NTHIK = np.int32(1)
            root_grp.VGTYP = np.int32(6) # ? [IO-API grid type = H: m above ground]
            root_grp.VGTOP = 10000. # ? [IO-API grid top for sigma coordinates]
            root_grp.VGLVLS = np.int32(0) # ? [IO-API levels from 0 to NLAYS]
            root_grp.GDNAM =  " "
            root_grp.UPNAM =  " "
            root_grp.UPDSC =  " "
            
            root_grp.close()            
