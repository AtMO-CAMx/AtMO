#!/usr/local/Anaconda3-5.0/bin/python
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import datetime as dt

# Global variables
####################################### User Inputs ################################
# Input Paths
MAINDIR = os.path.abspath('')
TEMPALL = os.path.join(MAINDIR,'TEMPORAL_ALLOCATION')
SEASONAL = os.path.join(TEMPALL, 'SEASONAL-FACS')

# output path
OUTPATH = './TEMPORAL_ALLOCATION/SEASONAL-FACS_adjusted.csv'
# country subject to factor adjustment
cnty_adj = ['France','Belgium','Netherlands','Luxembourg','Germany','Switzerland','Liechtenstein','Poland','Denmark','Norway','Sweden','Finland','Estonia','Latvia','Lithuania','United_Kingdom','Ireland']
# new factors to apply
NH3_factors = [0.35,0.70,1.20,1.80,1.40,1.10,1.10,1.35,1.70,0.55,0.40,0.35]
#####################################################################################

seasonal_header = ['COUNTRYCODE','SNAPSECTOR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
COUNTRY = os.path.join(TEMPALL, 'COUNTRIES-LIST.data')
country_list_headers = ['InDomain','ACRONYM','Code','Name','UNKNOWN','MENUT12']
country_lst = pd.read_csv(COUNTRY, sep='\s+', names=country_list_headers)
seasonal = pd.read_csv(SEASONAL, sep='\s+', names=seasonal_header)

# Modify the seasonal factors dataframe
seasonal_w_name = seasonal.merge(country_lst[['Code','Name']],left_on='COUNTRYCODE',right_on='Code',how='left')
seasonal_adj = seasonal_w_name[(seasonal_w_name['Name'].isin(cnty_adj))&(seasonal_w_name['SNAPSECTOR']==10)]
for i in range(12):
    seasonal_adj.iloc[:,i+2] = NH3_factors[i]
adj_ind = seasonal_adj.index.to_list() 
seasonal_no_adj = seasonal_w_name.drop(labels=adj_ind)
seasonal_new = pd.concat([seasonal_no_adj,seasonal_adj]).sort_index()
seasonal_final = seasonal_new.iloc[:,:-2].drop_duplicates()
seasonal_final.to_csv(OUTPATH,index=False)



