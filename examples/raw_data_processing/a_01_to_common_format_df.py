# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import glob

import pyarrow.parquet as pq

import time

from flasc import time_operations as to

import multiprocessing as mp


import os

resamp_dir = '/Users/dzalkind/Projects/AWAKEN/Engie_SCADA/processed/resampled_1'
if not os.path.exists(resamp_dir):
    os.makedirs(resamp_dir)


def load_data():
    # Load the filtered data
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", "00_raw_data")
    df = pd.read_feather(os.path.join(data_path, "df_scada_60s.ftr"))
    return df

def get_time_array(dt,month,year):


    dt = np.timedelta64(dt,'s')

    start_date = np.datetime64(f'{year:4d}-{month:02d}')
    end_date = start_date + np.timedelta64(1,'M')

    time_array = np.arange(start_date,end_date,dt)
    
    return time_array

def a_01_resample(inputs):
    month = inputs['month']
    year = inputs['year']
    wt_number = inputs['wt_number']

    base_dir = '/Users/dzalkind/Projects/AWAKEN/Engie_SCADA/farm=WIF_USKPL/'

    filedir = os.path.join(
        base_dir,
        f'ent_code=WIT_USKPL_SS001_WT{wt_number:03d}',
        f'year={year:4d}',
        f'month={month:02d}'
    )

    files = glob.glob(os.path.join(filedir,'*.parquet'))

    if len(files) > 1:
        print('WARNING: more than one file in month!!!')
    else:
        df_engie = pd.read_parquet(files[0],engine='pyarrow')

    scada_mapping = {
        'WindSpeed':f'ws_{wt_number:03d}',
        'WindDirection':f'wd_{wt_number:03d}',
        'InternalStatus':f'status_{wt_number:03d}',
        'NacelleAngle':f'yaw_{wt_number:03d}',
        'ActivePower':f'pow_{wt_number:03d}'   
    }

    # Set up raw and new dataframes
    df_raw = {}
    df_new = {}

    drop_cols = [col for col in df_engie.columns if col not in ['date','value']]

    tt = get_time_array(1,month,year)

    for channel, new_name in scada_mapping.items():
        df_raw[channel] = df_engie.loc[df_engie['tag'] == channel].reset_index(drop=True)
        df_raw[channel].sort_values(by='date',inplace=True)
        df_raw[channel].rename(columns={'date':'time'},inplace=True)
        df_raw[channel].drop(labels=drop_cols,axis=1,inplace=True)
        df_raw[channel].reset_index(drop=True,inplace=True)
        
        
        print(f'{channel} -> {new_name}')
        if channel in ['WindDirection','NacelleAngle']:
            circ = True
        else:
            circ = False
            
        if channel in ['NacelleAngle','InternalStatus']:
            interp_method = 'previous'
        else:
            interp_method = 'linear'
        
        df_new[new_name] = to.df_resample_by_interpolation(df_raw[channel],tt,circ,interp_method,max_gap=600)

    # Join dataframes
    for val in df_new:
        df_new[val].rename(columns={'value':val},inplace=True)

    to_join = [df[1] for df in df_new.items()]

    joined_df = to_join[0].set_index('time')

    for dj in to_join[1:]:
        if 'time' in dj:
            dj.set_index('time',drop=True,inplace=True)
        joined_df = joined_df.join(dj)

    joined_df.reset_index(inplace=True)
    joined_df.to_feather(os.path.join(resamp_dir,f'resamp_wt{wt_number:03d}_{year:4d}_{month:02d}.ftr'))



if __name__ == "__main__":
    # In this script, we rename the arbitrarily named variables from the
    # SCADA data to our common format: "wd_000", "wd_001", ..., "ws_000",
    # "ws_001", and so on. This helps to further automate and align
    # the next steps in data processing.

    months = [12]
    years = [2021]
    wt_numbers = range(1,89)

    cores = 4  # Running 8 bonked my computer!!

    input_list = []

    for year in years:
        for month in months:
            for wt_number in wt_numbers:
                inputs = {}
                inputs['year'] = year
                inputs['month'] = month
                inputs['wt_number'] = wt_number
                input_list.append(inputs)

    # Run cases
    if cores == 1:
        for inp in input_list:
            a_01_resample(inp)

    else:
        p = mp.Pool(cores)
        with p:
            p.map(a_01_resample,input_list)

        