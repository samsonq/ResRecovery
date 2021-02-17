import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '../src')

from utils import *

mbit_rate = 1/125000

# Comparing Byte Stream
def subplot_byte_stream(df1_lst, df2_lst, res_lst, byte_dir, xlim):

  color_lst = sns.color_palette()
  # preprocess data for plotting 
  x = np.arange(xlim)
  df1_bytes = [df.groupby('Time')[byte_dir].sum()  * mbit_rate for df in df1_lst]
  df2_bytes = [df.groupby('Time')[byte_dir].sum()  * mbit_rate for df in df2_lst]
  download_dict = {res_lst[i]: (df1_bytes[i], df2_bytes[i], color_lst[i]) for i in np.arange(len(res_lst))}

  # set up plot structure and labeling
  sns.set_style('whitegrid')
  fig, axes = plt.subplots(5, 2, figsize=(24, 18), sharex=True, sharey=True)

  # plot line graphs
  plot_idx = 0
  for res in download_dict.keys():
    sns.lineplot(x, download_dict[res][0][:xlim], label=res, color=download_dict[res][2][:xlim], ax=axes[plot_idx, 0])
    sns.lineplot(x, download_dict[res][1][:xlim], label=res, color=download_dict[res][2][:xlim], ax=axes[plot_idx, 1])
    axes[plot_idx, 0].set_title("Iman - " + res, fontsize=18)
    axes[plot_idx, 1].set_title("Stephen - " + res, fontsize=18)
    plot_idx += 1

  # aesthetic
  plt.suptitle('Download - All Resolutions (Action) - Youtube', fontsize=24)
  plt.subplots_adjust(top=0.95)
  plt.setp(axes, xlim=(0, 360), xticks=[0, 60, 120, 180, 240, 300, 360], ylim=(-1, 50), yticks=[0, 10, 20, 30, 40, 50])

  for ax in axes.flat:
      ax.set_xlabel("Seconds (from start)", fontsize=24)
      ax.set_ylabel("Mbps", fontsize=24)
      ax.label_outer()
  
  fig.show()

def rolling_bytes_stream(df_lst, res_lst, xlim, window_size_small, window_size_large, sample_size):

  color_lst = sns.color_palette()
  # preprocess data for plotting
  x_s = np.arange(xlim)
  x_l = np.arange(window_size_large, xlim + window_size_large)
  pre_rolling = [convert_ms_df(df).resample(sample_size, on='Time').sum()[['pkt_size']] for df in df_lst]
  rolling_s_sum = [df.rolling(window_size_small).mean().fillna(0) * mbit_rate for df in pre_rolling]
  rolling_l_sum = [df.rolling(window_size_large).mean().fillna(0) * mbit_rate for df in pre_rolling]
  rolling_sum_dict = {res_lst[i]: (rolling_s_sum[i], rolling_l_sum[i], color_lst[i]) for i in np.arange(len(res_lst))}
    
  # set up plot structure and labeling
  sns.set_style('whitegrid')
  fig, axes = plt.subplots(5, 2, figsize=(24, 18), sharex=False, sharey=True)

  # plot line graphs
  plot_idx = 0
  for res in rolling_sum_dict.keys():
    sns.lineplot(
      x_s, rolling_sum_dict[res][0]['pkt_size'][:xlim],label=res, color=rolling_sum_dict[res][2], ax=axes[plot_idx, 0])
    
    sns.lineplot(
      x_l, rolling_sum_dict[res][1]['pkt_size'][window_size_large:xlim+window_size_large], label=res, color=rolling_sum_dict[res][2], ax=axes[plot_idx, 1])

    axes[plot_idx, 0].set_title("6s Rolling Average - " + res, fontsize=18)
    axes[plot_idx, 1].set_title("60s Rolling Average - " + res, fontsize=18)
    
    plot_idx += 1

  # aesthetic
  plt.suptitle('Moving Average - 6s v 60s', fontsize=24)
  plt.subplots_adjust(top=0.95)
  #plt.setp(axes, xlim=(0, 360), xticks=[0, 60, 120, 180, 240, 300, 360], ylim=(-1, 50), yticks=[0, 10, 20, 30, 40, 50])

  for ax in axes.flat:
      ax.set_xlabel("Seconds (from start)", fontsize=24)
      ax.set_ylabel("Mbps", fontsize=24)
      ax.label_outer()
  
  fig.show()

# Exploring Data Peaks
def preprocess_data_peaks(data_lst, byte_dir):
  peak_df = pd.DataFrame()

  for df in data_lst:
    temp_download_df = df[[byte_dir]].loc[get_peak_loc(df, byte_dir)] * mbit_rate
    temp_download_df.columns = ['Mbps']
    temp_download_df['Direction'] = 'Download'
    temp_download_df['resolution'] = df['resolution'][0]
    peak_df = pd.concat((peak_df, temp_download_df))
  return peak_df

def subplot_peak_boxplot(peaks_df1, peaks_df2):
  sns.set_style('whitegrid')
  fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

  sns.boxplot(data=peaks_df1, x="resolution", y="Mbps", linewidth=2, ax=axes[0])
  axes[0].set_title("Peaks (All Resolutions) - Stephen", fontsize=20)

  sns.boxplot(data=peaks_df2, x="resolution", y="Mbps", linewidth=2, ax=axes[1])
  axes[1].set_title("Peaks (All Resolutions) - Iman", fontsize=20)

  sns.despine(left=True)

  for ax in axes.flat:
      ax.set_xlabel("Resolutions", fontsize=14)
      ax.set_ylabel("Mbps", fontsize=14)
      ax.label_outer()
  
  fig.show()

def subplot_peak_kde_hist(peaks_df1, peaks_df2, res_lst):
  fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
  plt.setp(axes, xlim=(0, 25), xticks=[0, 5, 10, 15, 20, 25])

  sns.histplot(data=peaks_df1, x="Mbps", hue="resolution", multiple="stack", legend=False, ax=axes[0,0])
  axes[0, 0].set_title("Peak Histogram - Stephen", fontsize=16)
  sns.kdeplot(data=peaks_df1, x="Mbps", hue="resolution", legend=False, ax=axes[0,1])
  axes[0, 1].set_title("Peak Density - Stephen", fontsize=16)

  sns.histplot(data=peaks_df2, x="Mbps", hue="resolution", multiple="stack", legend=False, ax=axes[1,0])
  axes[1, 0].set_title("Peak Histogram - Iman", fontsize=16)
  sns.kdeplot(data=peaks_df2, x="Mbps", hue="resolution", legend=False, ax=axes[1,1])
  axes[1, 1].set_title("Peak Density - Iman", fontsize=16)

  fig.legend(
    res_lst[::-1],
    loc="lower center",
    title="Resolution",
    ncol=len(res_lst)
  )

  fig.subplots_adjust(bottom=0.1)

  plt.suptitle('Peak Distributions', fontsize=20)
  plt.subplots_adjust(top=0.90)
  
  fig.show()
