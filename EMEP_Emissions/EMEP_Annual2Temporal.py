"""
Created by yshi
Lasted edited by yshi on 11/5/2019

This script converts annual emission CSV to hourly emission CSV using seasonal,weekly and hourly factors as cross references

Inputs:
    - Gridded annual emission files by GNFR sectors in CSV.
      *GNFR sectors: 'A_PublicPower', 'B_Industry', 'C_OtherStationaryComb',
       'D_Fugitive', 'E_Solvents', 'F_RoadTransport', 'G_Shipping',
       'H_Aviation', 'I_Offroad', 'J_Waste', 'K_AgriLivestock', 'L_AgriOther',
       'M_Other'
    - Seasonal, Weekly and Hourly factors by NFR emission sector
      *NFR sectos: '1','2','3','4','5','6','7','8','9','10'
    - Winter and Summer timeshift by country
    - Country code, name, abbreviation cross reference

Outputs:
    - Gridded hourly/temporal emission files by GNFR sectors in CSV
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import datetime as dt


def sector2name(factors,group):
        factors = factors.merge(nfr,on='SNAPSECTOR',how='left')
        factors_n = factors.groupby(group)[factors.columns[2:-1]].mean()
        return factors_n
    
def shp2df(shp):
    df = gpd.GeoDataFrame.from_file(shp)
    # merge winter and summer timeshift with countries
    df['EMISSIONS'] = df.groupby(['LCP_X','LCP_Y'], as_index=False)['EMISSION'].transform('sum')
    df = pd.DataFrame(df.drop(columns=['geometry','EMISSION']))
    df = df.rename(columns={'EMISSIONS': 'EMISSION'})
    df = df.drop_duplicates(subset=['LCP_X','LCP_Y','EMISSION'],keep='first')
    print(len(df))
    df['SECTOR'] = df['SECTOR'].map(lambda x: str(x)[4:])
    df2 = pd.merge(df,countries,left_on='ISO2',right_on='ACRONYM',how='left')
    return df2


############################## USER INPUTS #############################################################
# Input folder
MAINDIR = '/disk4/ATMOVISION/EMEP2WRF' 
TEMPALL = os.path.join(MAINDIR,'TEMPORAL_ALLOCATION')  # Location of factor inputs

# Output folder
OUTTEMP = os.path.join(MAINDIR,'OUTPUT','2015')

# input data to calculate factors
COUNTRY_W = os.path.join(TEMPALL, 'COUNTRIES-LIST-winter.data')
COUNTRY_S = os.path.join(TEMPALL, 'COUNTRIES-LIST-summer.data')
COUNTRY = os.path.join(TEMPALL, 'COUNTRIES-LIST.data')
NFR = os.path.join(TEMPALL, 'NFR-CODES')
SEASONAL = os.path.join(TEMPALL, 'SEASONAL-FACS')
HOURLY = os.path.join(TEMPALL, 'HOURLY-FACS')
WEEKLY = os.path.join(TEMPALL, 'WEEKLY-FACS')

########################################################################################################

country_headers = ['InDomain','ACRONYM','COUNTRYCODE','Name','TimeShift']
country_list_headers = ['InDomain','ACRONYM','Code','Name','UNKNOWN','MENUT12']
seasonal_header = ['COUNTRYCODE','SNAPSECTOR',
                   'JAN','FEB','MAR','APR','MAY','JUN',
                   'JUL','AUG','SEP','OCT','NOV','DEC']
weekly_header = ['COUNTRYCODE','SNAPSECTOR',
                 'MON','TUE','WED','THU','FRI','SAT','SUN']
hourly_header = ['MENUT12','WEEKDAY','SNAPSECTOR']+list(range(24))
nfr_header = ['NFRCODE','NFRNAME','SNAPSECTOR']

df_w = pd.read_csv(COUNTRY_W, sep='\t', names=country_headers)
df_s = pd.read_csv(COUNTRY_S, sep='\t', names=country_headers)
country_lst = pd.read_csv(COUNTRY, sep='\s+', names=country_list_headers)
nfr = pd.read_csv(NFR, sep='\t', index_col=0, names=nfr_header)
seasonal = pd.read_csv(SEASONAL, sep='\s+', names=seasonal_header)
hourly = pd.read_csv(HOURLY, sep='\s+', names=hourly_header, index_col=False)
weekly = pd.read_csv(WEEKLY, sep='\s+', names=weekly_header)

NFRNAME = nfr.NFRNAME.tolist() # All sectors
POL = ['CO','NH3','NMVOC','NOx','NH3','PM2_5','PMcoarse','SOx'] # All pollutants
#YR = ['2015'] # year
GRIDS = ['3.0','15.0','45.0'] # All gird sizes
yr = '2015'

print("Source files imported")
################################################################################################
# Perform some basic cleaning and manipulation 
 
countries_in = pd.merge(weekly['COUNTRYCODE'],seasonal['COUNTRYCODE'],on='COUNTRYCODE').drop_duplicates()
countries = pd.merge(df_s[['ACRONYM','COUNTRYCODE','TimeShift']],df_w[['COUNTRYCODE','TimeShift']],on='COUNTRYCODE',how='inner').rename(columns=
                {'TimeShift_x': 'TimeShift_s', 'TimeShift_y': 'TimeShift_w'})
countries = pd.merge(countries_in,countries,on='COUNTRYCODE',how='inner')

# cleaning of GNFR to SNAP sectors mapping for convenience
nfr.at[13,'SNAPSECTOR'] = '1,2,3,4,5,6,7,8,9,10'  # Convert "All" to numbers
nfr.SNAPSECTOR = nfr.SNAPSECTOR.apply(lambda x: x.split(','))
s = nfr.apply(lambda x: pd.Series(x['SNAPSECTOR']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'SNAPSECTOR'
nfr = nfr.drop('SNAPSECTOR', axis=1).join(s)
nfr['SNAPSECTOR'] = nfr['SNAPSECTOR'].astype(int) 

# remove the MENUT12 column
hourly = hourly[hourly.MENUT12 == 0]
hourly = hourly.drop(columns=['MENUT12'])


weekly = sector2name(weekly,['COUNTRYCODE','NFRNAME'])
hourly = sector2name(hourly,['WEEKDAY','NFRNAME'])
seasonal = sector2name(seasonal,['COUNTRYCODE','NFRNAME'])

#############################################################################################
datetimes = []
Months = (pd.date_range(yr+'/01/01 00:00:00', freq='M', periods=12)-pd.offsets.MonthBegin(1)).strftime('%Y%m%d').tolist()
for month in Months:
        Mon = pd.date_range(month, freq='W-MON', periods=2)
        hours = pd.date_range(Mon[1],freq='H',periods=168).strftime('%Y%m%d%H').tolist()
        datetimes+=hours
DST_start = dt.datetime.strptime(yr+'032901','%Y%m%d%H')
DST_end = dt.datetime.strptime(yr+'102501','%Y%m%d%H')
##############################################################################################    
starttime = dt.datetime.now()
for sector in NFRNAME:
    # generate factor file for each sector
    factors = countries.copy()
    factors['SECTOR'] = sector
    factors = pd.concat([factors, pd.DataFrame(columns=datetimes)],sort=False)
    #factors = factors.rename({0: 'COUNTRYCODE'}, axis=1).reset_index(drop=True)
    if sector == 'C_OtherStatComb':
        fsector = 'C_OtherStationaryComb'
    else: fsector = sector
    for i, row in factors.iterrows():
        countrycode = row['COUNTRYCODE']
        for col in datetimes:
            date = dt.datetime.strptime(col,'%Y%m%d%H')
            wd = date.weekday()
            mo = date.month
            hr = date.hour
            if date < DST_start or date > DST_end:
                TimeShift = 'TimeShift_w'
            else:
                TimeShift = 'TimeShift_s'
            timeshift = row[TimeShift]
            shifted_hr = int(hr + timeshift)
            if shifted_hr >= 24:
                wd += 1
                if wd > 6:
                    wd = 0
                shifted_hr = 0
            seasonal_factor = seasonal.loc[countrycode,sector][mo-1]
            weekly_factor = weekly.loc[countrycode,sector][wd]
            hourly_factor = hourly.loc[wd+1,sector][shifted_hr]
            factors.at[i,col] = seasonal_factor*weekly_factor*hourly_factor

    for grd in GRIDS:
        for pol in POL:
            grdpth = 'GRID_'+grd+'KM'
            filename = pol+'_'+yr+'_'+fsector+'_WRF_'+grd+'km.shp'
            savedir = 'CSV_WRF_%skm' %grd
            savename = pol+'_'+yr+'_'+fsector+'_WRF_'+grd+'km.csv'
            savepath = os.path.join(OUTTEMP,savedir,savename)
            print('\nProcessing '+filename+' now...')
            df_n = shp2df(os.path.join(OUTPUT,grdpth,filename))            
            df2 = pd.merge(df_n,factors,left_on='COUNTRYCODE',right_on='COUNTRYCODE',how='left');#left
            bigdf = df2.drop(columns=['ACRONYM_x','ACRONYM_y','TimeShift_w_y','TimeShift_s_y','SECTOR_y','COUNTRYCODE','TimeShift_s_x','TimeShift_w_x'])
            bigdf = bigdf.rename(columns={'SECTOR_x':'SECTOR'})
            bigdf.fillna(1,inplace=True)
            for tm in datetimes:
                bigdf[tm] = bigdf['EMISSION']*bigdf[tm]*1000000/365/24
            if not os.path.isdir(os.path.join(OUTTEMP,savedir)):
                os.mkdir(os.path.join(OUTTEMP,savedir))
            bigdf.to_csv(savepath, index=False)
            print(filename+' completed.')
            
