import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from datetime import datetime
import multiprocessing as mp

import gc


plot_dir = '/srv/data/nfs/scada/processed/plots_test/'
base_dir = '/srv/data/nfs/scada/dap_raw'

def a_01_plot(inputs):

    # load raw data (from DAP)
    month = inputs['month']
    year = inputs['year']
    wt_number = inputs['wt_number']

    file = os.path.join(base_dir,f'kp.turbine.z02.00.{year}{month:02d}01.000000.wt{wt_number:03d}.parquet')

    print(f'Reading {file}')
    plot_raw = True
    try:
        df_engie = pd.read_parquet(file)

        scada_mapping = {
            'WindSpeed':f'ws_{wt_number-1:03d}',
            'WindDirection':f'wd_{wt_number-1:03d}',
            'Status':f'status_{wt_number-1:03d}',
            'NacelleAngle':f'yaw_{wt_number-1:03d}',
            'ActivePower':f'pow_{wt_number-1:03d}'   
        }

        # Set up raw and new dataframes
        df_raw = {}

        drop_cols = [col for col in df_engie.columns if col not in ['date','value']]

        for channel, new_name in scada_mapping.items():
            df_raw[channel] = df_engie.loc[df_engie['tag'] == channel].reset_index(drop=True)
            df_raw[channel].sort_values(by='date',inplace=True)
            df_raw[channel].rename(columns={'date':'time'},inplace=True)
            df_raw[channel].drop(labels=drop_cols,axis=1,inplace=True)
            df_raw[channel].reset_index(drop=True,inplace=True)
    except:
        print(f"Unable to read {file}")
        plot_raw = False


    # load resampled data
    file = f'/srv/data/nfs/scada/processed/resampled_1/resamp_wt{wt_number:03d}_{year}_{month:02d}.ftr'
    print(f'Reading {file}')
    plot_resamp = True
    try:
        df_resamp = pd.read_feather(file)
    except:
        print(f"Unable to read {file}")
        plot_resamp = False


    # do plots
    fig, axs = plt.subplots(len(scada_mapping),1)
    fig.set_size_inches(24,2*len(scada_mapping))

    for i_ax, channel in enumerate(scada_mapping):
        # Plot resampled
        if plot_resamp:
            axs[i_ax].plot(df_resamp['time'],df_resamp[scada_mapping[channel]],label='resampled')        

        # Plot raw
        if plot_raw:
            axs[i_ax].plot(df_raw[channel].time,df_raw[channel].value,'.',label='raw')
        
        # Formatting and axis labels
        axs[i_ax].set_ylabel(channel)
    
    axs[-1].legend(loc='lower right')
    [a.set_xlim([pd.Timestamp('2022-01-01 00:00:00'), pd.Timestamp('2022-01-01 03:00:00')]) for a in axs]
    axs[3].set_ylim([150,250])
    fig.savefig(os.path.join(plot_dir,f'{year}.{month:02d}.wt{wt_number:03d}.png'))



def main():
    months = range(2,3)
    years = [2022]
    wt_numbers = range(1,89)

    cores = 1  # Running 8 bonked my computer!!

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'Running a_01 to resample SCADA @ {current_time}')
    
    input_list = []

    for year in years:
        for month in months:
            for wt_number in wt_numbers:
                inputs = {}
                inputs['year'] = year
                inputs['month'] = month
                inputs['wt_number'] = wt_number

                filename = os.path.join(plot_dir,f'{year}.{month:02d}.wt{wt_number:03d}.png')
            
                if not os.path.exists(filename):
                    input_list.append(inputs)
                    print(f'Generating {filename}')
                    sys.stdout.flush()

    # Run cases
    if cores == 1:
        for inp in input_list:
            a_01_plot(inp)

    else:
        p = mp.Pool(cores)
        with p:
            p.map(a_01_plot,input_list)

if __name__=="__main__":
    main()