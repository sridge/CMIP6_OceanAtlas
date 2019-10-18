# not all fo these importa may be necessary?

from easy_coloc import lib_easy_coloc
import xarray as xr
import pandas as pd
import cartopy as cart
import matplotlib.pylab as plt
from matplotlib import cm
import datetime
import cmocean
import numpy as np
import dateutil
import intake
import dask

def model_to_glodap(ovar_name=None, 
                            model=None, 
                            catalog_path='../catalogs/pangeo-cmip6.json',
                            qc_path='../qc'):
    '''
    generate_model_section(ovar_name, model)
    
    Input
    ==========
    ovar_name : variable name (eg 'dissic')
    model : model name (eg CanESM5)
    
    Output
    ===========
    ds : dataset of section output
    '''
    institue = {'CanESM5':'CCCma',
                'CNRM-ESM2-1':'CNRM-CERFACS',
                'IPSL-CM6A-LR':'IPSL',
                'MIROC-ES2L':'MIROC',
                'UKESM1-0-LL':'MOHC',
                'GISS-E2-1-G-CC':'NASA-GISS',
                'GISS-E2-1-G':'NASA-GISS'
               }

    # Get CMIP6 output from intake_esm
    col = intake.open_esm_datastore(catalog_path)
    cat = col.search(experiment_id='historical', 
                     table_id='Omon', 
                     variable_id=ovar_name, 
                     grid_label='gn')

    # dictionary of subset data
    dset_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True}, 
                                    cdf_kwargs={'chunks': {}})

    # Put data into dataset
    ds = dset_dict[f'CMIP.{institue[model]}.{model}.historical.Omon.gn']

    
    # Rename olevel to lev 
    coord_dict = {'olevel':'lev'} # a dictionary for converting coordinate names
    if 'olevel' in ds.dims:
        ds = ds.rename(coord_dict)

    # load GLODAP station information from csv file
    # drop nans, reset index, and drop uneeded variable
    df = pd.read_csv(f'{qc_path}/GLODAPv2.2019_COORDS.csv')
    df = df.dropna()
    df = df.reset_index().drop('Unnamed: 0', axis=1)

    # Genearte times list and put into dataframe
    times = [f'{int(year)}-{int(month):02d}' for year,month in zip(df.year,df.month)]
    df['dates'] = times 

    # Find unique dates, these are the sample dates
    sample_dates = df['dates'].sort_values().unique()
    
    # Parse the historical period
    sample_dates = sample_dates[0:125]
    sample_dates = [dateutil.parser.parse(date) - pd.Timedelta('16 day') for date in sample_dates]

    # shift dates to middle of the month
    ds['time'] = pd.date_range(start=f'{ds.time.dt.year[0].values}-{ds.time.dt.month[0].values:02}',
                            end=f'{ds.time.dt.year[-1].values}-{ds.time.dt.month[-1].values:02}',
                            freq='MS')

    # ==========================================
    # Here we start making the ovar dataset
    # ==========================================
    # Trim the dates to sample_dates
    ovar = ds[ovar_name].sel(time=sample_dates)
    ovar['lat'] = ds.latitude
    ovar['lon'] = ds.longitude

    # create source grid and target section objects
    # this requires lon,lat from stations and the source grid dataset containing lon,lat
    proj = lib_easy_coloc.projection(df['longitude'].values,df['latitude'].values,grid=ovar,
                                     from_global=True)

    # 4-D max for easy_coloc. Not entirely sure what we are squeezing out?
    ovar = ovar.squeeze() 

    # run the projection on the WOA analyzed temperature (t_an)
    fld = np.zeros((len(sample_dates),len(ovar.lev),len(df)))

    # 
    for ind in range(5, 130, 5):
        dates = sample_dates[ind-5:ind]
        fld_tem = proj.run(ovar.sel(time=dates)[:])
        fld[ind-5:ind,:,:] = fld_tem

    # create datarray with sampling information
    sampled_var = xr.DataArray(fld,
                               dims=['time','lev','all_stations'],
                               coords={'time':ovar['time'],
                                       'lev':ovar['lev'],
                                       'all_stations':df.index.values,
                                       'dx':('all_stations',df.dx.values),
                                       'bearing':('all_stations',df.bearing.values),
                                       'lat':('all_stations',df.latitude.values),
                                       'lon':('all_stations',df.longitude.values),
                                      },
                               attrs={'units':ovar.units,
                                      'long_name':ovar.long_name
                                     }
                              )

    # Glodap expo codes
    expc = pd.read_csv(f'{qc_path}/FILTERED_GLODAP_EXPOCODE.csv')

    # convert datarray to dataset
    # This grabs everything
    ds = sampled_var.to_dataset(name=ovar.name)

    return ds