#!/usr/bin/env python
# coding: utf-8

# DL apprach see
# 
# https://www.analyticsvidhya.com/blog/2019/01/introduction-time-series-classification/
# 
# https://towardsdatascience.com/how-to-use-convolutional-neural-networks-for-time-series-classification-56b1b0a07a57
#     
# or with big guns use attention:
#     
# https://towardsdatascience.com/attention-for-time-series-classification-and-forecasting-261723e0006d
# 
# Data:
# 
# https://timeseriesclassification.com/
# 
# Misc:
# 
# https://tsfresh.readthedocs.io/en/latest/text/quick_start.html
# 
# https://github.com/hfawaz/bigdata18
# 
# https://paperswithcode.com/paper/an-empirical-evaluation-of-generic
# 
# https://paperswithcode.com/paper/multilevel-wavelet-decomposition-network-for
# 
# Human activity recognition:
# 
# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
# 
# CNN, QRNN
# 
# https://stats.stackexchange.com/questions/403502/is-it-a-good-idea-to-use-cnn-to-classify-1d-signal
# 
# Temporal convolutions
# 
# https://arxiv.org/pdf/1803.01271.pdf
# 
# kaggle freesound
# 
# https://github.com/sainathadapa/kaggle-freesound-audio-tagging



#get_ipython().run_line_magic('matplotlib', 'inline')
import os
import gc
import sys
import pandas as pd
from pandas.plotting import scatter_matrix
from pandas import set_option
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.colors as colors
from sklearn import preprocessing
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from classification_utilities import display_cm, display_adj_cm
import scipy
from scipy.fftpack import fft
from scipy import ndimage
from skimage import util
import librosa
import librosa.display
import pywt
from pathlib import Path
from scipy.ndimage import map_coordinates
import psutil
from memory_profiler import profile



def show_mem_usage():
    '''Displays memory usage from inspection
    of global variables in this notebook'''
    gl = sys._getframe(1).f_globals
    vars= {}
    for k,v in list(gl.items()):
        # for pandas dataframes
        if hasattr(v, 'memory_usage'):
            mem = v.memory_usage(deep=True)
            if not np.isscalar(mem):
                mem = mem.sum()
            vars.setdefault(id(v),[mem]).append(k)
        # work around for a bug
        elif isinstance(v,pd.Panel):
            v = v.values
        vars.setdefault(id(v),[sys.getsizeof(v)]).append(k)
    total = 0
    for k,(value,*names) in vars.items():
        if value>1e6:
            print(names,"%.3fMB"%(value/1e6))
        total += value
    print("%.3fMB"%(total/1e6))




print(show_mem_usage())


data_dir = Path('data')

data_images = Path('data/train_images')

GR_MAX=190
DPI=54
IMG_HEIGHT=256

# 1528 x 390
# figsize=((IMG_HEIGHT*4)/51.905454545, IMG_HEIGHT/50.487272727)

# 1100 x 275
FIGSIZE = ((IMG_HEIGHT * 4) / 72.101395041, IMG_HEIGHT / 71.600132231)

set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None

filename = 'data/CAX_LogFacies_Train_File.csv'
training_data = pd.read_csv(filename)
TRAIN_DATA=True


training_data.info()




training_data[training_data.GR > 195].count()
# ### replace -ve values with zero
training_data[training_data.GR < 0].count()
training_data.loc[training_data.GR < 0, 'GR'] = 0



# ### Preliminary Facies types
# 
# <pre>
# 0: shale
# 1: silty sst/interbedded
# 2: sandstone
# 1: silty sst/interbedded
# 4: fining up silty sst/interbedded
# </pre>



facies_colors = ['#6E2C00','#DC7633','#F4D03F',
                 '#2E86C1', '#AED6F1']





#      0=shale   1=mixed1  2=sst   3=mixed2 4=mixed3
ccc = ['#996633','blue','yellow','red','green']
cmap_facies = colors.ListedColormap(ccc[0:len(ccc)], 'indexed')




def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError
    if x.size < window_len:
        raise ValueError
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError
    
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y  




def smooth_resample( y, newlen ):
    """ resample y to newlen, with B-spline smoothing """
    #n = len(y)
    downsampled =  scipy.signal.resample(y, newlen)
    #newgrid = np.linspace( 0, n - 1, newlen )
    #return map_coordinates( y, newgrid, mode="nearest", prefilter=False )
    return downsampled




def plot_well(well):
    cluster=np.repeat(np.expand_dims(test_well['label'].values,1), 100, 1)
    gr_values=well.GR.values.tolist()
    y=range(len(gr_values))

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 12))
    ax[0].plot(gr_values, y)
    ax[0].invert_yaxis()
    ax[0].set_ylim(len(gr_values),0)
    ax[1].invert_yaxis()
    ax[1].set_ylim(len(gr_values),0)
    im=ax[1].imshow(cluster, interpolation='none', aspect='auto',cmap=cmap_facies,vmin=0,vmax=4)



@profile
def plot_facies(well, figsize=(18, 8), repeat=100, save_name=None):
    cluster=np.repeat(np.expand_dims(well['label'].values,1), repeat, 1)
    rotated_img = ndimage.rotate(cluster, 90)

    if save_name:
        plt.rcParams["figure.figsize"] = figsize
        plt.imsave(f'{save_name}.png', rotated_img,cmap=cmap_facies,vmin=0,vmax=4)
        del rotated_img, cluster
        plt.close()
    else:
        im=plt.imshow(rotated_img, interpolation='none', aspect='auto',cmap=cmap_facies,vmin=0,vmax=4)



@profile
def plot_signal(time, signal, figsize=(18, 4), save_name=None):
    plt.rcParams["figure.figsize"] = figsize
    plt.plot(time, signal, color='k')
    plt.xlim([time[0], time[-1]])
    plt.ylim([0, GR_MAX])
    if save_name:
        plt.axis('off')
        spec = plt.imshow
        plt.savefig(f'{save_name}.png', transparent = False, bbox_inches = 'tight', pad_inches = 0)
        del spec
        plt.close()




def delta_signal(signal):
    yarr=np.asarray(signal)
    yarr=np.diff(signal)
    yarr = np.insert(yarr, 0, signal[0], axis=0)
    assert len(yarr)==len(signal)
    return yarr




def y_trend(signal, N=15):
    y_mean=smooth(signal,window_len=N)
    #sample back out to originallength
    y=smooth_resample(y_mean, len(signal))
    y_delta_trend=delta_signal(y)
    return y_delta_trend




def rect(x,y,w,h,c):
    ax = plt.gca()
    polygon = plt.Rectangle((x,y),w,h,color=c, antialiased=False)
    ax.add_patch(polygon)

@profile
def rainbow_fill(X,Y, trend_len, cmaps=['Blues_r','Reds_r']):
    '''Option for one or two cmpas per plot'''
    if len(cmaps)==1:
        cmap=plt.get_cmap(cmaps[0])
    else:
        cmap_1=plt.get_cmap(cmaps[0])
        cmap_2=plt.get_cmap(cmaps[1])
    plt.plot(X,Y,lw=0)  # Plot so the axes scale correctly, lw=linewidth
    plt.xlim([X[0], X[-1]])
    plt.ylim([0, GR_MAX])
    dx = X[1]-X[0]
    N  = float(len(X))
    
    ydeltas=y_trend(Y, trend_len)

    for n, (x,y, yd) in enumerate(zip(X,Y, ydeltas)):
        if len(cmaps)==1:
            color = cmap(y/GR_MAX)
        else:
            if yd>0:
                color = cmap_1(y/GR_MAX)
            else:
                color = cmap_2(y/GR_MAX)
        rect(x,0,dx,y,color)

def mask_fill(X,Y, facies, cmap='Set1'):
    '''Option for one or two cmpas per plot'''
    #facies=well['label'].values
    cmap=cmap_facies
    #plt.get_cmap(cmap)

    plt.plot(X,Y,lw=0)  # Plot so the axes scale correctly, lw=linewidth
    plt.xlim([X[0], X[-1]])
    plt.ylim([0, GR_MAX])
    dx = X[1]-X[0]
    N  = float(len(X))
    #some wells dont have all 5 facies
    #assert(len(list(set(facies))))==5
    for n, (x,y, f) in enumerate(zip(X,Y, facies)):
        color = cmap(f)
        rect(x,0,dx,y,color)

@profile
def fill_signal(time, signal, figsize=(18, 4), save_name=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    y2=signal*0
    assert len(time)==len(signal)
    ax.plot(time, signal, time, y2, color='black')
    ax.fill_between(time, signal, y2, where=signal >y2, facecolor='black')
    ax.set_xlim([time[0], time[-1]])
    if save_name:
        ax.axis('off')
        spec = plt.imshow
        plt.savefig(f'{save_name}.png', transparent = False, bbox_inches = 'tight', pad_inches = 0)
        del spec
        plt.close()



@profile
def rainbow_signal(time, signal, trend_len, figsize=(18, 4), cmaps=['jet'], save_name=None):
    plt.rcParams["figure.figsize"] = figsize
    rainbow_fill(time, signal, trend_len, cmaps)
    if save_name:
        plt.axis('off')
        spec = plt.imshow
        plt.savefig(f'{save_name}.png', transparent = False, bbox_inches = 'tight', pad_inches = 0)
        del spec
        plt.close()


def masked_signal(time, signal, facies, figsize=(18, 4), cmap='Set1', save_name=None, dpi=None):
    plt.rcParams["figure.figsize"] = figsize
    mask_fill(time, signal, facies, cmap)
    if save_name:
        plt.axis('off')
        #spec = plt.imshow
        if dpi:
            plt.savefig(f'{save_name}.png', transparent = False, bbox_inches = 'tight', pad_inches = 0, dpi=dpi)
        else:
            plt.savefig(f'{save_name}.png', transparent = False, bbox_inches = 'tight', pad_inches = 0)
        #del spec
        plt.close()



@profile
def plot_wavelet(ax, time, signal, scales, waveletname = 'cmor', 
                 cmap = plt.cm.Spectral, title = '', ylabel = '', xlabel = ''):
    
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)
    #print(f'levels: {contourlevels}, frequencies: {frequencies}, power: {power}' )
    #contour([X, Y,] Z, [levels], **kwargs)
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)
    return yticks, ylim, frequencies




def plot_well_wavelet(ax, time, signal, scales, waveletname = 'cmor', 
                 cmap = plt.cm.Spectral, title = '', ylabel = '', xlabel = '', plot_axes=True):
    
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [1/32,1/16,1/8,1/4,1/2, 1, 2, 4, 8, 16, 32]
    contourlevels = np.log2(levels)
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    if plot_axes:
        ax.set_title(title, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_xlabel(xlabel, fontsize=18)
    
        yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    if waveletname=='morl' or waveletname=='shan':
        ax.set_ylim(7.0, 1.0)
    elif waveletname=='cgau1' or waveletname=='mexh' or waveletname=='cgau2':
        ax.set_ylim(8.0, 2.0)
    else:
        ax.set_ylim(8.0, 1.0)
    if not plot_axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)




def plot_wavelet_tranform(x, y, figsize, title, waveletname = 'cmor', plot_axes=True, save_name=None, cmap= plt.cm.Spectral):
    scales = np.arange(1, 128)
    fig, ax = plt.subplots(figsize=(18, 4))
    plot_well_wavelet(ax, x, y, scales, title=title, waveletname = waveletname, plot_axes=plot_axes, cmap=cmap)
    if save_name:
        plt.axis('off')
        spec = plt.imshow
        plt.rcParams["figure.figsize"] = figsize
        plt.savefig(f'{save_name}', transparent = False, bbox_inches = 'tight', pad_inches = 0) 
        del spec
        plt.close()
    else:
        pass
        #plt.show()

def plt_smth_spec(data, figsize, save_name=None):

    NFFT =16
    noverlap=14
    dt = 0.5
    fs = int(1.0 / dt)
    ham_window = np.hamming(NFFT)

    scale = 'dB'
    gr_dm=de_mean(data)
    gr_rm=smooth(gr_dm,window_len=5)
    gr_sm = smooth(gr_dm, window_len=5, window='flat')
    plt.figure(figsize=figsize)

    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(gr_sm, window=ham_window, NFFT=NFFT, Fs=fs, noverlap=noverlap, scale=scale, cmap='viridis')
    plt.xticks([])
    plt.yticks([])
    if save_name:
        plt.axis('off')
        spec = plt.imshow
        plt.rcParams["figure.figsize"] = figsize
        plt.savefig(f'{save_name}.png', transparent = False, bbox_inches = 'tight', pad_inches = 0)
        del spec
        plt.close()



def plt_spec(data, figsize, save_name=None):
    NFFT =16
    noverlap=14
    pad_to=32
    dt = 0.5
    N = len(data)
    fs = 1/dt
    scale = 'dB'
    ham_window = np.hamming(NFFT)
    detrend=None
    mode='psd'
    gr_dm=de_mean(data)
    plt.figure(figsize=figsize)
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(gr_dm, window=ham_window, NFFT=NFFT, Fs=fs, noverlap=noverlap, scale=scale, pad_to=pad_to, detrend=detrend, mode=mode)
    plt.xticks([])
    plt.yticks([])
    if save_name:
        plt.axis('off')
        spec = plt.imshow
        plt.rcParams["figure.figsize"] = figsize
        plt.savefig(f'{save_name}.png', transparent = False, bbox_inches = 'tight', pad_inches = 0)
        del spec
        plt.close()





def plot_well_hz(well, figsize=(12, 8)):
    cluster=np.repeat(np.expand_dims(well['label'].values,1), 100, 1)
    #cluster[:,0:10]=cluster[:,10:20]
    #cluster[:,89:99]=cluster[:,10:20]
    gr_values=well.GR.values.tolist()
    x=range(len(gr_values))

    f, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    ax[0].plot(x, gr_values)
    ax[0].set_xlim(0, len(gr_values))
    ax[1].set_xlim(0, len(gr_values))
    rotated_img = ndimage.rotate(cluster, 90)
    im=ax[1].imshow(rotated_img, interpolation='none', aspect='auto',cmap=cmap_facies,vmin=0,vmax=4)



# ## Spectrograms



def get_ave_values(xvalues, yvalues, n = 5):
    signal_length = len(xvalues)
    if signal_length % n == 0:
        padding_length = 0
    else:
        padding_length = n - signal_length//n % n
    xarr = np.array(xvalues)
    yarr = np.array(yvalues)
    xarr.resize(signal_length//n, n)
    yarr.resize(signal_length//n, n)
    xarr_reshaped = xarr.reshape((-1,n))
    yarr_reshaped = yarr.reshape((-1,n))
    x_ave = xarr_reshaped[:,0]
    y_ave = np.nanmean(yarr_reshaped, axis=1)
    return x_ave, y_ave




def de_mean(signal):
    # removing the mean of the signal
    mean_removed = np.ones_like(signal)*np.mean(signal)
    signal_demean = signal - mean_removed
    return signal_demean



# First lets load the el-Nino dataset, and plot it together with its time-average
def plot_signal_plus_average(ax, time, signal, average_over = 5):
    time_ave, signal_ave = get_ave_values(time, signal, average_over)
    ax.plot(time, signal, label='signal')
    ax.plot(time_ave, signal_ave, label = 'time average (n={})'.format(5))
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Amplitude', fontsize=16)
    ax.set_title('Signal + Time Average', fontsize=16)
    ax.legend(loc='upper right')



def get_fft_values(y_values, T, N):
    xf = np.linspace(0.0, N*T, N)
    yf_ = fft(y_values)
    yf = 2.0/N * np.abs(yf_[0:N/2])
    return xf, yf

def get_fft_values2(y_values, T, N):
    N2 = 2 ** (int(np.log2(N)) + 1) # round up to next highest power of 2
    xf = np.linspace(0.0, 1.0/(2.0*T), N2//2)
    yf_ = fft(y_values)
    yf = 2.0/N2 * np.abs(yf_[0:N2//2])
    return xf, yf

def plot_fft_plus_power(ax, time, signal, plot_direction='horizontal', yticks=None, ylim=None):
    dt = time[1] - time[0]
    N = len(signal)
    fs = 1/dt
    
    variance = np.std(signal)**2
    f_values, fft_values = get_fft_values2(signal, dt, N)
    fft_power = variance * abs(fft_values) ** 2
    if plot_direction == 'horizontal':
        ax.plot(f_values, fft_values, 'r-', label='Fourier Transform')
        ax.plot(f_values, fft_power, 'k--', linewidth=1, label='FFT Power Spectrum')
    elif plot_direction == 'vertical':
        scales = 1./f_values
        scales_log = np.log2(scales)
        ax.plot(fft_values, scales_log, 'r-', label='Fourier Transform')
        ax.plot(fft_power, scales_log, 'k--', linewidth=1, label='FFT Power Spectrum')
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels(yticks)
        ax.invert_yaxis()
        ax.set_ylim(ylim[0], -1)
    ax.legend()



pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info()[0]/2.**30
print('memory use:', memoryUse)


def create_mask_plots(facies_data, gr_data, data_images, i, figsize, smth_y):
    masked_signal(range(len(smth_y)), smth_y, facies_data, figsize=figsize, cmap='Set1',
                  save_name=f'{data_images}/well_{i}_smth_5_masked')
    masked_signal(range(len(gr_data)), gr_data, facies_data, figsize=figsize, cmap='Set1',
                  save_name=f'{data_images}/well_{i}_masked')

@profile
def plot_wells_mask(well_ids):
    '''Note use of epsolin sclars to the DPI to exactly match the image size of the facies image 1100x275
    Couldn't work out how to get mpl to imsave to a specific size'''

    #1100 x 275
    figsize = FIGSIZE
    print(f'figsize: {figsize}')
    for i in well_ids:
        if (i % 10)==0:
            print(f'--plot_wells_mask() processing well {i}')
            memoryUse = py.memory_info()[0] / 2. ** 30
            print(f'id: {id}, memory use: {memoryUse}, mem: {show_mem_usage()}') 
        well = training_data[training_data['well_id'] == i]
        gr_data=well.GR.values
        facies_data = well.label.values
        smth_y=smooth(gr_data, window_len = 5)
        time=range(len(gr_data))
        #assert len(gr_data)==1100
        create_mask_plots(facies_data, gr_data, data_images, i, figsize, smth_y)
        gc.collect()




well_ids=training_data.well_id.unique().tolist()
well_ids_cut=well_ids[3700:]
#well_ids_cut=well_ids_cut[::-1]
plot_wells_mask(well_ids_cut)








def create_smooth_plots(time, well, gr_data, data_images, i, figsize, smth_y, smoother):
    if TRAIN_DATA:
        plot_facies(well, figsize, repeat=int(len(smth_y) / 4), save_name=f'{data_images}/well_{i}_facies')
    plot_signal(range(len(smth_y)), smth_y, figsize=figsize, save_name=f'{data_images}/well_{i}_GR_line_smth_{smoother}')
    rainbow_signal(range(len(smth_y)), smth_y, trend_len=20, figsize=figsize, cmaps=['Blues_r', 'Reds_r'],
                   save_name=f'{data_images}/well_{i}_GRsmth_{smoother}')

#@profile
def create_basic_plots(time, well, gr_data, data_images, i, figsize, smth_y=None):
    if TRAIN_DATA:
        plot_facies(well, figsize, repeat=int(len(gr_data) / 4), save_name=f'{data_images}/well_{i}_facies')
    plot_signal(range(len(gr_data)), gr_data, figsize=figsize, save_name=f'{data_images}/well_{i}_GR_line')
    #rainbow_signal(time, gr_data, trend_len=20, figsize=figsize, cmaps=['Blues_r', 'Reds_r'],
    #               save_name=f'{data_images}/well_{i}_GR')


def plot_wells_basic(well_ids):
    '''Note use of epsolin sclars to the DPI to exactly match the image size of the facies image 1100x275
    Couldn't work out how to get mpl to imsave to a specific size'''
    figsize=((IMG_HEIGHT*4)/51.905454545, IMG_HEIGHT/50.487272727)
    print(f'figsize: {figsize}')
    for i in well_ids:
        if (i % 10)==0:
            print(f'processing well {i}')
            memoryUse = py.memory_info()[0] / 2. ** 30
            print(f'id: {id}, memory use: {memoryUse}, mem: {show_mem_usage()}') 
        well = training_data[training_data['well_id'] == i]
        gr_data=well.GR.values
        smth_y=smooth(gr_data, window_len = 5)
        time=range(len(gr_data))
        #assert len(gr_data)==1100
        create_basic_plots(time, well, gr_data, data_images, i, figsize, smth_y)
        #create_smooth_plots(time, well, gr_data, data_images, i, figsize, smth_y, smoother=5)
        gc.collect()




#plot_wells_basic(well_ids)




def plot_wells_extended(well_ids):
    '''Note use of epsolin sclars to the DPI to exactly match the image size of the facies image 1100x275
    Couldn't work out how to get mpl to imsave to a specific size'''

    figsize=FIGSIZE
    for i in well_ids:
        if (i % 100)==0:
            print(f'processing well {i}')
            memoryUse = py.memory_info()[0] / 2. ** 30
            print(f'--plot_wells_extended() id: {id}, memory use: {memoryUse}, mem: {show_mem_usage()}')
        well = training_data[training_data['well_id'] == i]
        gr_data=well.GR.values
        smth_y=smooth(gr_data, window_len = 5)
        time=range(len(gr_data))
        assert len(gr_data)==1100
        #plot_facies(well, figsize, repeat=int(len(gr_data)/4), save_name=f'{data_images}/well_{i}_facies')
        #plot_signal(range(len(smth_y)), smth_y, figsize, save_name=f'{data_images}/well_{i}_GR_line_smth_5')
        #rainbow_signal(range(len(smth_y)), smth_y, trend_len=20, figsize=figsize, cmaps=['Blues_r','Reds_r'], save_name=f'{data_images}/well_{i}_GR_smth_5')
        plt_spec(gr_data, figsize, save_name=f'{data_images}/well_{i}_spec')
        plt_smth_spec(gr_data, figsize, save_name=f'{data_images}/well_{i}_spec_smth_5')
        #gr_dm=de_mean(gr_data)
        #plot_wavelet_tranform(range(len(gr_data)), gr_dm/10, figsize, title='', waveletname = 'cmor', plot_axes=False, save_name=f'{data_images}/well_{i}_cmor_wvlt', cmap= plt.cm.Spectral)
        #plot_wavelet_tranform(gr_time, gr_dm/10, figsize, title='', waveletname = 'cgau2', plot_axes=False, save_name=f'{data_images}/well_{i}_cgau2_wvlt', cmap= plt.cm.Spectral)
        #plot_wavelet_tranform(gr_time, gr_dm/10, figsize, title='', waveletname = 'shan', plot_axes=False, save_name=f'{data_images}/well_{i}_shan_wvlt', cmap= plt.cm.Spectral)
        gc.collect()

#plot_wells_extended(well_ids)

def plot_wells_square(well_ids):
    '''Note use of epsolin sclars to the DPI to exactly match the image size of the facies image 1100x275
    Couldn't work out how to get mpl to imsave to a specific size'''
    pma = path_lbl / f'{size[0]}'
    os.makedirs(pma, exist_ok=True)
    pmi = path_lbl / f'{size[0]}'
    os.makedirs(pma, exist_ok=True)
    figsize = (12, 12 * 1.0265)
    for i in well_ids:
        if (i % 100) == 0:
            print(f'processing well {i}')
            memoryUse = py.memory_info()[0] / 2. ** 30
            print(f'id: {id}, memory use: {memoryUse}, mem: {show_mem_usage()}')
        well = training_data[training_data['well_id'] == i]
        gr_data = well.GR.values
        smth_y = smooth(gr_data, window_len=5)
        time = range(len(gr_data))
        # assert len(gr_data)==1100
        masked_signal(range(len(smth_y)), smth_y, facies_data, figsize=figsize, cmap='Set1',
                      save_name=f'{data_images}/well_{i}_smth_5_masked', dpi=100)
        masked_signal(range(len(gr_data)), gr_data, facies_data, figsize=figsize, cmap='Set1',
                      save_name=f'{data_images}/well_{i}_masked', dpi=100)

        gc.collect()