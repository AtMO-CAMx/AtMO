#!/usr/bin/python

"""
Created by yshi
Lasted edited by yshi on 11/5/2019

This script takes speciated EMEP emission NetCDF and remove the inner part of the domain where local emissions are non-zero.
Inputs:
    - Gridded speciated EMEP emission NetCDFs
    - Local emission vs. CAMx grid cells cross reference file

Outputs:
    - Gridded speciated EMEP emission files without local emission domain
"""


base = '/disk4/ATMOVISION/Local_emissions_database/'
outdir = '/disk4/ATMOVISION/EMEP2WRF/OUTPUT/2015/SPECIATED/NETCDF4/3.0km/3.0km_outer/'
xref_path = base + 'CAMx3km_spatialjoin_grille3km.csv'
xref = pd.read_csv(xref_path)    
empty_cells = xref[xref['Id_km']!=0]
zero_i = empty_cells.i.astype(int)
zero_j = empty_cells.j.astype(int)

sectors = ['A_PublicPower', 'B_Industry', 'C_OtherStationaryComb',
       'D_Fugitive', 'E_Solvents', 'F_RoadTransport', 'G_Shipping',
       'H_Aviation', 'I_Offroad', 'J_Waste', 'K_AgriLivestock', 'L_AgriOther',
       'M_Other']
for sector in sectors:
    print(sector)
    sec_name = sector[2:]
    if sector == 'F_RoadTransport':
        sec_name = 'RoadTransport_noIVOA'
    elif sector == 'K_AgriLivestock':
        sec_name = 'AgriLivestock_adjust'
    elif sector == 'L_AgriOther':
        sec_name = 'AgriOther_adjust'
    else: pass
    sec_dir = '/disk4/ATMOVISION/EMEP2WRF/OUTPUT/2015/SPECIATED/NETCDF4/3.0km/'+sec_name
    out_sec = outdir+sec_name
    if not os.path.exists(out_sec):
        os.mkdir(out_sec)
    else: pass
    files = [x for x in os.listdir(sec_dir) if x.endswith('.nc')]
    for f in files:
        print(f)
        fpath = sec_dir+'/'+f
        ds = xr.open_dataset(fpath)
        ds_var = [v for v in ds.variables if len(ds[v].dims)==4]
        for v in ds_var:
            ds[v].values[:,:,zero_j,zero_i]=0
        ds.attrs['history'] = "Created " + datetime.today().strftime("%d%m%Y")
        ds.attrs['CDATE'] = np.int32(datetime.today().strftime("%Y%j")) # [IO-API file creation date]
        ds.attrs['CTIME'] = np.int32(datetime.today().strftime("%k%M%S")) # [IO-API file creation time]
        ds.attrs['WDATE'] = np.int32(datetime.today().strftime("%Y%j")) # [IO-API file write date]
        ds.attrs['WTIME'] = np.int32(datetime.today().strftime("%k%M%S"))  # [IO-API file write time]
        outpath = out_sec+'/'+f[:-3]+'_outer'+f[-3:]
        ds.to_netcdf(outpath)
