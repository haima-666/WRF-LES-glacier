import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import *
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from utils import *
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors

mpl.rcParams['figure.dpi'] = 150
# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

scatter_size = 10

# 定义颜色映射
colors = [(-20, 'blue'), (0, 'white'), (20, 'red')]
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

def time_filter(df):
    # df = df[(df['time'].dt.month>=7) & (df['time'].dt.month<=9)]
    # df = df[(df['time']>='2023-8-24') & (df['time']<'2023-10-1') & ((df['time'].dt.month>=7) & (df['time'].dt.month<=9))]
    df = df[(df['time'] >= '2023-7-21') & (df['time'] < '2023-9-10')]

    for i in df.columns:
        if i in ['AirT', 'Ta_Avg', 'T2m', 'Ta', 'Tair']:
            Tairname = i
            break
        else:
            pass
    # df = df.dropna(subset=[Tairname])
    # the = np.percentile(df[Tairname], 40)
    # df = df[df[Tairname]> the]

    return df

def time_filter1(df):
    df = df[(df['time'] >= '2023-7-1') & (df['time'] < '2023-10-10')]
    return df
def climate_filter(adf, df, climate_vari):
    awsdf = adf.loc[:, ['time', climate_vari]]
    awsdf = awsdf.dropna(subset=[climate_vari])
    if climate_vari in ['Sin', 'DR_Avg']:
        awsdf = awsdf[awsdf[climate_vari]>1]
    condition1 = (awsdf[climate_vari] > np.percentile(awsdf[climate_vari], 80))
    condition2 = (awsdf[climate_vari] < np.percentile(awsdf[climate_vari], 20))
    awsdfhigh = awsdf[condition1]
    awsdflow = awsdf[condition2]

    dfh = pd.merge(awsdfhigh, df, on='time', how='inner')
    dfhigh = dfh.drop(climate_vari, axis=1)

    dfl = pd.merge(awsdflow, df, on='time', how='inner')
    dflow = dfl.drop(climate_vari, axis=1)

    return dfhigh, dflow
def exception_remove(df_compare):
    ###exception detect
    df_compare = df_compare.dropna().reset_index(drop=True)
    low_threshold = np.percentile(df_compare['bias'], 20)
    high_threshold = np.percentile(df_compare['bias'], 80)
    iqr = high_threshold - low_threshold
    exee = np.where(
        (df_compare['bias'] < (low_threshold - 1.5 * iqr)) | (df_compare['bias'] > (high_threshold + 1.5 * iqr)))
    df_compare = df_compare.drop(exee[0]).reset_index(drop=True)

    return df_compare

class parlung4_orignal:
    def __init__(self):
        self.climate = readfile('parlung4/4par4745_2009-5-21to2023-10-16.csv')
        self.climate['uwi'] = UWI_calcul(self.climate['wd'], 30)
        self.h4745 = readfile('parlung4/4par4745_2009-5-21to2023-10-16.csv')
        self.h4686 = readfile('parlung4/4par4686_2023-8-23to2023-10-16.csv')
        self.h4833 = readfile('parlung4/4par4833_2021-5-15to2023-10-6.csv')
        self.h5200 = readfile('parlung4/4par5200_2010-7-2to2013-5-8.csv')
        self.h5400 = readfile('parlung4/4par5400_2014-6-11to2022-10-5.csv')
        self.h5500 = readfile('parlung4/4par5500_2006-6-25to2007-4-21.csv')
        self.h4641 = readfile('parlung4/4par4641_2021-5-15to2023-10-16.csv')
        self.h4674 = readfile('parlung4/4par4674_2021-5-15to2023-10-16.csv')
        self.h4400 = readfile('parlung4/4par4400_2014-11-8to2024-1-17.csv')
        self.h4600 = readfile('parlung4/4par4600_2006-6-1to2023-11-7.csv')
        self.aws =  [self.h4745, self.h4833]
        self.aws = [x.loc[:, ['time', 'Tair']] for x in self.aws]
        self.fl = [6303, 5214, 6885, 6885+790, 6885+900]
        self.fl_length =7615
        self.altitude = [4745, 4833, 4686, 4674, 4641]
        self.nomfl = [x/self.fl_length for x in self.fl]
        self.tantou = zip(self.aws, self.altitude, self.fl, self.nomfl)
        self.lapse = -0.0065
        self.p_tname = ['Tair', 'AirT', 'Ta_Avg', 'T2m', 'Ta']

    def compute_ts(self, df, altitude, old=False):
        df4600off = self.h4600
        df4400off = self.h4400
        dflapse = pd.DataFrame()
        if old:
            dflapse['time'] = df4600off['time']
            dflapse['Test'] = df4600off['Tair'] + self.lapse * (altitude - 4600)
        else:
            dflapse['time'] = df4400off['time']
            dflapse['Test'] = df4400off['AirT'] + self.lapse * (altitude - 4400)

        for name in df.columns:
            if name in self.p_tname:
                T_ambient_name = name
        df = df.loc[:, ['time', T_ambient_name]]
        df_c = pd.merge(dflapse, df, how='inner', on='time')

        df_c = df_c.dropna(subset=[T_ambient_name, 'Test'])
        df_c['bias'] = df_c['Test'] - df_c[T_ambient_name]
        df_c = exception_remove(df_c)
        # mean = np.nanmean(df_c['bias'])
        # std = np.nanstd(df_c['bias'])
        w, b, r, p, std = stats.linregress(df_c['Test'], df_c[T_ambient_name])
        return w

    def compute_bias(self,df, altitude, old=False):
        df4600off = self.h4600
        df4400off = self.h4400
        dflapse = pd.DataFrame()
        if old:
            dflapse['time'] = df4600off['time']
            dflapse['Test'] = df4600off['Tair'] + self.lapse * (altitude - 4600)
        else:
            dflapse['time'] = df4400off['time']
            dflapse['Test'] = df4400off['AirT'] + self.lapse * (altitude - 4400)

        for name in df.columns:
            if name in self.p_tname:
                T_ambient_name = name
        df = df.loc[:, ['time', T_ambient_name]]
        df_c = pd.merge(dflapse, df, how='inner', on='time')

        df_c = df_c.dropna(subset=[T_ambient_name, 'Test'])
        df_c['bias'] = df_c['Test'] - df_c[T_ambient_name]
        df_c['Tobs'] = df_c[T_ambient_name]
        return df_c

class parlung4old(parlung4_orignal):
    def __init__(self):
        super().__init__()
        self.aws =  [self.h4745, self.h4833, self.h4686,
                     self.h4674, self.h4641]
        self.aws = [x.loc[:, ['time', 'Tair']] for x in self.aws]
        self.fl = [6303, 5214, 6885, 6885+790, 6885+900]
        self.fl_length =7615
        self.altitude = [4745, 4833, 4686, 4674, 4641]
        self.nomfl = [x/self.fl_length for x in self.fl]
        self.tantou = zip(self.aws, self.altitude, self.fl, self.nomfl)
        self.lapse = -0.0065
        self.p_tname = ['Tair', 'AirT', 'Ta_Avg', 'T2m', 'Ta']

class parlung4():
    def __init__(self):
        super().__init__()
        self.h4648 = readfile('./parlung4/new_iceT/4par4648_2021-05-15to2023-04-16.csv')
        self.h4687 = readfile('./parlung4/new_iceT/4par4687_2021-05-15to2023-04-16.csv')
        self.h4768 = readfile('./parlung4/new_iceT/4par4768_2021-05-15to2023-04-16.csv')
        self.h4809 = readfile('./parlung4/new_iceT/4par4809_2021-05-15to2023-04-16.csv')
        self.h4841 = readfile('./parlung4/new_iceT/4par4841_2021-05-15to2023-04-25.csv')
        self.h4909 = readfile('./parlung4/new_iceT/4par4909_2021-05-15to2023-04-25.csv')

        self.aws =  [self.h4648, self.h4687, self.h4768, self.h4809,
                     self.h4841, self.h4909]
        self.fl = [6885+900, 6885+790,6646, 6223, 5698, 5040]
        self.fl_length = 7615
        self.altitude = [4648, 4687, 4768, 4809, 4841, 4909]
class parlung94:
    def __init__(self):
        self.h4948 = readfile('94/94par4948_2023-7-21to2023-10-19.csv')
        self.climate = readfile('94/94aws20191004to20231016.csv')
        self.climate['uwi'] = UWI_calcul(self.climate['WD_Avg'], 345)
        self.climate['time'] = self.climate['time'] + pd.to_timedelta(-12, unit='h')
        self.h5261 = readfile('94/94par5261_2023-7-21to2023-10-20.csv')
        self.h5302 = readfile('94/94par5302_2023-7-21to2023-10-20.csv')
        self.h5028 = readfile('94/94par5028_2023-7-21to2023-10-19.csv')
        self.h4600 = readfile('parlung4/4par4600_2006-6-1to2023-11-7.csv')
        self.h5098 = readfile('94/94par5098_2023-7-21to2023-10-19.csv')
        self.h5218 = readfile('94/94par5218_2023-7-21to2023-10-19.csv')

        self.aws = [self.h5028, self.h5098, self.h5261, self.h5218, self.h5302]
        self.aws_withoff = [self.h5028, self.h5098, self.h5261, self.h5218, self.h5302, self.h4948]
        self.aws = [x.loc[:, ['time', 'Tair']] for x in self.aws]
        self.altitude = [5028, 5098, 5261, 5218, 5302]
        self.fl = [1885, 1565, 821, 995, 738]
        self.fl_length = 2272
        self.nomfl = [x/self.fl_length for x in self.fl]
        self.tantou = zip(self.aws, self.altitude, self.fl, self.nomfl)
        self.tantou_withoff = zip(self.aws_withoff, [5028, 5098, 5261, 5218, 5302, 4948], [1885, 1565, 821, 995, 738, 3785], [x/self.fl_length for x in [1885, 1565, 821, 995, 738, 3785]])
        self.lapse = -0.0065
        self.p_tname = ['Tair', 'AirT', 'Ta_Avg', 'T2m', 'Ta']

    def compute_ts(self, df, altitude):
        dfoff = self.h4600
        dflapse = pd.DataFrame()

        dflapse['time'] = dfoff['time']
        dflapse['Test'] = dfoff['Tair'] + self.lapse * (altitude - 4600)

        for name in df.columns:
            if name in self.p_tname:
                Tobs = name
        df = df.loc[:, ['time', Tobs]]
        df_c = pd.merge(dflapse, df, how='inner', on='time')

        df_c = df_c.dropna(subset=[Tobs, 'Test'])
        df_c['bias'] = df_c['Test'] - df_c[Tobs]
        df_c = exception_remove(df_c)
        # mean = np.nanmean(df_c['bias'])
        # std = np.nanstd(df_c['bias'])
        w, b, r, p, std = stats.linregress(df_c['Test'], df_c[Tobs])
        return w

    def compute_bias(self, df, altitude):
        dfoff = self.h4600
        dflapse = pd.DataFrame()

        dflapse['time'] = dfoff['time']
        dflapse['Test'] = dfoff['Tair'] + self.lapse * (altitude - 4600)

        for name in df.columns:
            if name in self.p_tname:
                Tobs = name
        df = df.loc[:, ['time', Tobs]]
        df_c = pd.merge(dflapse, df, how='inner', on='time')

        df_c = df_c.dropna(subset=[Tobs, 'Test'])
        df_c['bias'] = df_c['Test'] - df_c[Tobs]
        df_c = exception_remove(df_c)
        df_c['Tobs'] = df_c[Tobs]
        return df_c
class laigu:
    def __init__(self, termius = False):
        self.climate = readfile('laigu/laigu4305_2022-7-12to2023-10-17.csv')
        self.climate['uwi'] = UWI_calcul(self.climate['uwi'], 115)
        self.h4305 = readfile('laigu/laigu4305_2022-7-12to2023-10-17.csv')
        self.h4383 = readfile(r'laigu\laigu4383_2021-11-14to2023-10-17.csv')
        self.h4456 = readfile('laigu/laigu4456_2021-11-14to2023-10-22.csv')

        self.h4442 = readfile('laigu/laigu4442_2021-11-14 to2023-10-21 .csv')

        self.h4299 = readfile('laigu/laigu4299_2023-08-27 to2023-10-21 .csv')
        self.h4258 = readfile('laigu/laigu4258_2023-08-27 to2023-10-21 .csv')
        self.h4340 = readfile('laigu/laigu4340_2023-08-27 to2023-10-21 .csv')

        self.altitude = [4319, 4367, 4379, 4391, 4305]
        self.altitude_new = [4148, 4191, 4258]

        self.fl = [19340, 18725, 18459, 18085, 21851]
        self.fl_new = [22360, 21548, 21013]
        self.fl_length = 28346

        if termius:
            self.altitude = self.altitude+self.altitude_new
            self.aw = self.aw+self.aw_new
            self.fl = self.fl+self.fl_new
        self.nomfl = [x / self.fl_length for x in self.fl]
        self.lapse = -0.0065
        self.p_tname = ['Tair', 'AirT', 'Ta_Avg', 'T2m', 'Ta']
        self.aws = []
        for aws in self.aw:
            for name in aws.columns:
                if name in self.p_tname:
                    Tobs_name = name
                    break
            aa = aws.loc[:, ['time', Tobs_name]]
            self.aws.append(aa)
        self.tantou = zip(self.aws, self.altitude, self.fl, self.nomfl)
    def compute_ts(self, df, altitude):

        dflaigu4383off = self.h4383

        dflapse = pd.DataFrame()
        dflapse['time'] = dflaigu4383off['time']
        dflapse['Test'] = dflaigu4383off['Tair'] + self.lapse*(altitude - 4383)

        for name in df.columns:
            if name in self.p_tname:
                Tobs = name
        df = df.loc[:, ['time', Tobs]]
        df_c = pd.merge(dflapse, df, how='inner', on='time')

        df_c = df_c.dropna(subset=[Tobs, 'Test'])
        df_c['bias'] = df_c['Test'] - df_c[Tobs]
        df_c = exception_remove(df_c)
        # mean = np.nanmean(df_c['bias'])
        # std = np.nanstd(df_c['bias'])
        w, b, r, p, std = stats.linregress(df_c['Test'], df_c[Tobs])
        return w

    def compute_bias(self, df, altitude):

        dflaigu4383off = self.h4383

        dflapse = pd.DataFrame()
        dflapse['time'] = dflaigu4383off['time']
        dflapse['Test'] = dflaigu4383off['Tair'] + self.lapse*(altitude - 4383)

        for name in df.columns:
            if name in self.p_tname:
                Tobs = name
        df = df.loc[:, ['time', Tobs]]
        df_c = pd.merge(dflapse, df, how='inner', on='time')

        df_c = df_c.dropna(subset=[Tobs, 'Test'])
        df_c['bias'] = df_c['Test'] - df_c[Tobs]
        df_c = exception_remove(df_c)

        df_c['Tobs'] = df_c[Tobs]
        return df_c


tspar4 = parlung4_orignal()
awspa4 = tspar4.tantou

tspar94 = parlung94()
aws94 = tspar94.tantou
aws94withoff = tspar94.tantou_withoff

tslaigu = laigu()
awslaigu = tslaigu.tantou


def mean_T():

    df4340 = tslaigu.h4340  #3
    df4258 = tslaigu.h4258  #1
    df4387 = readfile(r'C:\Users\admin\Desktop\code\ranwuT2m\laigu\laigu4387_2023-08-27 to2023-10-21 .csv')#4
    df4442 = tslaigu.h4442  #5
    df4456 = tslaigu.h4456  #6
    df4299 = tslaigu.h4299  #2

    df4383 = tslaigu.h4383
    df4305 = tslaigu.h4305
    df4305['Tair'] = df4305['Ta_Avg']

    altitude_yalong= [4258, 4299, 4340, 4387, 4442, 4456, 4305, 4383]
    tloggers_yalong = [df4258, df4299, df4340,df4387,df4442, df4456, df4305, df4383]
    flowline_yalong = [22360, 21548, 21013, 19869, 18725, 18459, 21851, 27346]
    flowline_yalong = [x/28346 for x in flowline_yalong]

    df4686 = tspar4.h4686
    df4745 = tspar4.h4745
    df4833 = tspar4.h4833
    df4674 = tspar4.h4674
    df4641 = tspar4.h4641
    altitude_p4 = [4686, 4745, 4833, 4674, 4641]
    tloggers_p4 = [df4686, df4745, df4833, df4674, df4641]
    flowline_p4 = [6885, 6303, 5214, 7655, 7760]
    flowline_p4 = [x/7615 for x in flowline_p4]

    df5098 = tspar94.h5098
    df5218 = tspar94.h5218
    df5261 = tspar94.h5261
    df5302 = tspar94.h5302
    df4948 = tspar94.h4948
    df5028 = tspar94.h5028
    df4990 = readfile(r'C:\Users\admin\Desktop\code\ranwuT2m\94\94par4990_2023-7-21to2023-10-20.csv')
    altitude_p94 = [5098, 5218, 5261, 5302, 5028, 4990, 4948]
    tloggers_p94 = [df5098, df5218, df5261, df5302, df5028, df4990, df4948]
    flowline_p94 = [1565, 995, 821, 738, 1885, 2450, 2550]
    flowline_p94 = [x/2272 for x in flowline_p94]

    off_altitude = [4305, 4674, 4641, 4990, 4948, 4383]
    off_station = [df4305, df4674, df4641, df4990, df4948, df4383]
    def time_filter(df):
        df = df[(df['time']>'2023-08-28')&(df['time']<'2023-09-20')]
        return df
    def afternoon_filter(df):
        df = df[(df['time'].dt.hour >= 12) & (df['time'].dt.hour < 17)]
        return df

    fig, axs = plt.subplots(2,2, figsize=(10,6))
    ax = axs.flatten()
    for i, v in enumerate(tloggers_yalong):
        df_202309 = time_filter(v)

        mean_tair = np.nanmean(df_202309['Tair'])

        df202309_high = afternoon_filter(df_202309)
        high_mean_tair = np.nanmean(df202309_high['Tair'])

        if i == 6 or i== 7:
            co = 'grey'
        else:
            co='royalblue'

        if i == 0:
            ax[0].scatter(altitude_yalong[i], mean_tair, color=co, label = 'Yalong', edgecolors= 'black', s=90, marker='^')
            ax[1].scatter(flowline_yalong[i], mean_tair, color=co, edgecolors= 'black', s=90, marker='^')
            ax[2].scatter(altitude_yalong[i], high_mean_tair, color=co, edgecolors= 'black', s=90, marker='^')
            ax[3].scatter(flowline_yalong[i], high_mean_tair, color=co, edgecolors= 'black', s=90, marker='^')
        else:
            ax[0].scatter(altitude_yalong[i], mean_tair, color=co, edgecolors= 'black', s=90, marker='^')
            ax[1].scatter(flowline_yalong[i], mean_tair, color=co, edgecolors= 'black', s=90, marker='^')
            ax[2].scatter(altitude_yalong[i], high_mean_tair, color=co, edgecolors= 'black', s=90, marker='^')
            ax[3].scatter(flowline_yalong[i], high_mean_tair, color=co, edgecolors= 'black', s=90, marker='^')


    for i, v in enumerate(tloggers_p4):
        df_202309 = time_filter(v)
        mean_tair = np.nanmean(df_202309['Tair'])
        df202309_high = afternoon_filter(df_202309)
        high_mean_tair = np.nanmean(df202309_high['Tair'])
        if i > 2:
            co = 'grey'
        else:
            co='green'
        if i ==0:
            ax[0].scatter(altitude_p4[i], mean_tair, color=co, label='Parlung 4', edgecolors= 'black', s=70)
            ax[1].scatter(flowline_p4[i], mean_tair, color=co, edgecolors= 'black', s=70)
            ax[2].scatter(altitude_p4[i], high_mean_tair, color=co, edgecolors= 'black', s=70)
            ax[3].scatter(flowline_p4[i], high_mean_tair, color=co, edgecolors= 'black', s=70)
        else:
            ax[0].scatter(altitude_p4[i], mean_tair, color=co, edgecolors= 'black', s=70)
            ax[1].scatter(flowline_p4[i], mean_tair, color=co, edgecolors= 'black', s=70)
            ax[2].scatter(altitude_p4[i], high_mean_tair, color=co, edgecolors= 'black', s=70)
            ax[3].scatter(flowline_p4[i], high_mean_tair, color=co, edgecolors= 'black', s=70)


    for i, v in enumerate(tloggers_p94):
        df_202309 = time_filter(v)
        mean_tair = np.nanmean(df_202309['Tair'])
        df202309_high = afternoon_filter(df_202309)
        high_mean_tair = np.nanmean(df202309_high['Tair'])
        if i > 4:
            co = 'grey'
            size1 = 120
        else:
            co='r'
            size1 = 120

        if i == 0:
            ax[0].scatter(altitude_p94[i], mean_tair, color=co, label='Parlung 94', edgecolors= 'black', s=size1, marker='*')
            ax[1].scatter(flowline_p94[i], mean_tair, color=co, edgecolors= 'black', s=size1, marker='*')
            ax[2].scatter(altitude_p94[i], high_mean_tair, color=co, edgecolors= 'black', s=size1, marker='*')
            ax[3].scatter(flowline_p94[i], high_mean_tair, color=co, edgecolors= 'black', s=size1, marker='*')
        else:
            ax[0].scatter(altitude_p94[i], mean_tair, color=co, edgecolors= 'black',s=size1, marker='*')
            ax[1].scatter(flowline_p94[i], mean_tair, color=co, edgecolors= 'black', s=size1, marker='*')
            ax[2].scatter(altitude_p94[i], high_mean_tair, color=co, edgecolors= 'black', s=size1, marker='*')
            ax[3].scatter(flowline_p94[i], high_mean_tair, color=co, edgecolors= 'black', s=size1, marker='*')

    ax[0].legend()

    high_meanT_off = []
    meanToff = []
    for i ,v in enumerate(off_station):
        TT = time_filter(v)
        meanoffT = np.nanmean(TT['Tair'])

        dfhighh = afternoon_filter(TT)
        dfhighhmean = np.nanmean(dfhighh['Tair'])
        meanToff.append(meanoffT)
        high_meanT_off.append(dfhighhmean)

    w1, b1 = slope_cal(off_altitude, meanToff)
    w2, b2 = slope_cal(off_altitude, high_meanT_off)
    x = np.arange(4000, 5600, 100)
    ax[0].plot(x, w1*x+b1, linestyle='--', c='black')
    ax[2].plot(x, w2 * x + b2, linestyle='--', c='black')
    ax[0].set_xlim(4100, 5400)
    ax[2].set_xlim(4100, 5400)
    # ax[1].axvline(x=tspar4.fl_length, color='green', linestyle='--')
    # ax[1].axvline(x=tspar94.fl_length, color='r', linestyle='--')
    # ax[1].axvline(x=tslaigu.fl_length, color='b', linestyle='--')
    # ax[3].axvline(x=tspar4.fl_length, color='green', linestyle='--')
    # ax[3].axvline(x=tspar94.fl_length, color='r', linestyle='--')
    # ax[3].axvline(x=tslaigu.fl_length, color='b', linestyle='--')

    ax[0].text(5200, 4, f'{round(w1, 4)}', c='r')
    ax[2].text(5200, 6, f'{round(w2, 4)}', c='r')

    for i in range(4):
        ax[i].grid(alpha=0.7)
        ax[i].set_ylim(0, 14)
    ax[0].set_ylabel('Mean Tair (℃)')
    # ax[0].set_title('Ta All')
    ax[2].set_ylabel('Mean Tair (℃)')

    ax[2].set_xlabel('Altitude (m)')
    ax[3].set_xlabel('Flowline Distance (m)')
    plt.show()
    # plt.savefig(r'C:\Users\admin\Desktop\冷却效应小论文\图\meanTair.jpg', dpi=300)




def uwi_ws_Tbias_Test():
    df4 = tspar4.climate
    dfrs = tspar4.h4600.loc[:, ['time', 'RS']]
    dfrs = dfrs[dfrs['RS'] > 10]

    df4 = df4.loc[:, ['time', 'uwi', 'ws','Tair']]
    df4['sen_heat'] = 1006 * 0.0038 *1.29 * (-1* df4['Tair']) * df4['ws']
    df4_bias = tspar4.compute_bias(df4, 4745, old=True)
    df4_bias = df4_bias.loc[:, ['time', 'bias']]
    df44 = df4.dropna()
    df4inn = df44[(df44['time']>='2009-6-20') & (df44['time']<='2009-9-1')]
    df4in = pd.merge(df4inn, df4_bias, how='inner', on= 'time')
    df4_ambient_T = tspar4.h4600.loc[:, ['time', 'Tair']]
    df4_ambient_T['Test'] = df4_ambient_T['Tair'] + (-0.0065)*(4745-4600)
    df4_ambient_T = df4_ambient_T[(df4_ambient_T['time'] >= '2009-6-20') & (df4_ambient_T['time'] <= '2009-9-1')]

    df94 = tspar94.climate.loc[:, ['time', 'WS_Avg', 'uwi', 'Ta_Avg','DR_Avg']]
    df94['sen_heat'] = 1006 * 0.0038 *1.29 * (-1* df94['Ta_Avg']) * df94['WS_Avg']
    df94_bias = tspar94.compute_bias(df94, 5218)
    df94_bias = df94_bias.loc[:, ['time', 'bias']]
    df944 = df94[(df94['time']>='2023-6-20') & (df94['time']<='2023-9-1')]
    df944 = df944.dropna()
    df94in = pd.merge(df944, df94_bias, how='inner', on= 'time')
    df94_ambient_T = tspar4.h4600.loc[:, ['time', 'Tair']]
    df94_ambient_T['Test'] = df94_ambient_T['Tair'] + (-0.0065) * (5218 - 4600)
    df94_ambient_T = df94_ambient_T[(df94_ambient_T['time'] >= '2023-6-20') & (df94_ambient_T['time'] <= '2023-9-1')]

    dflaigu = tslaigu.climate.loc[:, ['time', 'WS', 'uwi', 'Ta_Avg']]
    dflaigu['sen_heat'] = 1006 * 0.0038 *1.29 * (-1* dflaigu['Ta_Avg']) * dflaigu['WS']
    dflaiguice = tslaigu.h4322
    dflaiguice['time'] = hour_zero(dflaiguice['time'])
    dflaigu_bias = tslaigu.compute_bias(dflaiguice, 4322)
    dflaigu_bias = dflaigu_bias.loc[:, ['time', 'bias']]
    dflaiguu = dflaigu.dropna()
    dflaiguu = dflaiguu[(dflaiguu['time']>='2023-6-20') & (dflaiguu['time']<='2023-9-1')]
    dflaiguin = pd.merge(dflaiguu, dflaigu_bias, how='inner', on= 'time')
    dflaigu_off_T = tslaigu.h4383
    dflaigu_off_T['Test'] = dflaigu_off_T['Tair'] + (-0.0065) * (4322 - 4383)
    dflaigu_off_T = dflaigu_off_T[(dflaigu_off_T['time'] >= '2023-6-20') & (dflaigu_off_T['time'] <= '2023-9-1')]
    df4_rs = pd.merge(dfrs, df4in,on='time', how='inner')

    dflaigu_rs = pd.merge(dfrs, dflaiguin, on='time', how='inner')
    df94_rs = pd.merge(dfrs, df94in, on='time', how='inner')

    size4 = df4_rs['ws']*100 / np.max(df4_rs['ws'])

    sizelai = dflaigu_rs['WS']*100/np.max(df4_rs['ws'])
    size94 = df94_rs['WS_Avg']*100/np.max(df4_rs['ws'])
    fig, ax = plt.subplots(2,3, figsize=(10,6))
    ax = ax.flatten()
    # 定义颜色映射
    colors1 = [(0, 'red'),(0.3, 'white'), (1, 'blue')]
    cmap1 = LinearSegmentedColormap.from_list('custom_cmap', colors1)

    df4plot = pd.merge(df4_ambient_T, df4_rs, on='time', how='inner')  ## ws

    # sca = ax[1].scatter(df4plot['Test'], df4plot['uwi'], s = size1, edgecolor='black', c=df4plot['bias'],
    #                     cmap=cmap1, vmin=-3, vmax=10)

    df94plot = pd.merge(df94_ambient_T, df94_rs, on='time', how='inner')  ## WS_Avg

    df94low = df4plot[df94plot['uwi'] < -0.8]
    dflaiguplot = pd.merge(dflaigu_off_T, dflaigu_rs, on='time', how='inner')  ## WS

    # df4plot['T2'] = df4plot['Tair_x']
    # dflaiguplot['T2'] = dflaiguplot['Tair']
    # df94plot['T2'] = df94plot['Ta_Avg']
    #
    # dflaiguplot['ws'] = dflaiguplot['WS']
    # df94plot['ws'] = df94plot['WS_Avg']
    #
    # def df_bins(df11):
    #     # 创建分区（每 0.5 为一个区间）
    #     bins = np.arange(np.min(df11['Test']), np.max(df11['Test']), 0.5)  # 区间范围 [3, 13]，步长为 0.5
    #     df11['bin'] = pd.cut(df11['Test'], bins, right=False)
    #
    #     # 按分区计算统计量
    #     stats1 = df11.groupby('bin')['ws'].agg(
    #         mean='mean',
    #         std='std',
    #         # q25=lambda x: np.percentile(x, 25),
    #         # q75=lambda x: np.percentile(x, 75)
    #     ).reset_index()
    #     # 将区间转为字符串便于查看
    #     # 计算区间平均值
    #     stats1['bin_mean'] = stats1['bin'].apply(lambda x: (x.left + x.right) / 2)
    #     return stats1
    # dflaiguplot_mean = df_bins(dflaiguplot)
    # df4plot_mean = df_bins(df4plot)
    # df94plot_mean = df_bins(df94plot)
    #
    # plotdf = [dflaiguplot_mean, df4plot_mean, df94plot_mean]
    # coo = ['blue', 'green', 'red']
    # labell = ['Yalong', 'P4', 'P94']
    # for index, ii in enumerate(plotdf):
    #     ax.plot(ii['bin_mean'], ii['mean'], color=coo[index], label=labell[index])
    #     ax.fill_between(ii['bin_mean'], ii['mean'] - ii['std'],
    #                     ii['mean'] + ii['std'], alpha=0.2, color=coo[index])
    # ax.set_xlim(0, 17)
    # ax.legend()
    # ax.set_xlabel('Tair')
    # ax.set_ylabel('Wind speed')
    # plt.show()

    print(df94low.shape[0] / df94plot.shape[0])
    ax[2].scatter(df94plot['Test'], df94plot['uwi'], edgecolor='black', c=df94plot['bias'], cmap=cmap1,
                  vmin=-3, vmax=10)
    ax[1].scatter(df4plot['Test'], df4plot['uwi'],  edgecolor='black', c=df4plot['bias'], cmap=cmap1,
                  vmin=-3, vmax=10)
    ax[0].scatter(dflaiguplot['Test'], dflaiguplot['uwi'], edgecolor='black', c=dflaiguplot['bias'], cmap=cmap1,
                  vmin=-3, vmax=10)

    ax[5].scatter(df94plot['Test'], df94plot['WS_Avg'], edgecolor='black', c=df94plot['bias'], cmap=cmap1,
                  vmin=-3, vmax=10)
    ax[4].scatter(df4plot['Test'], df4plot['ws'],  edgecolor='black', c=df4plot['bias'], cmap=cmap1,
                  vmin=-3, vmax=10)
    ax[3].scatter(dflaiguplot['Test'], dflaiguplot['WS'], edgecolor='black', c=dflaiguplot['bias'], cmap=cmap1,
                  vmin=-3, vmax=10)

    # ax[2].scatter(dflaiguplot['Test'], dflaiguplot['uwi'], s=size3, edgecolor='black', c=dflaiguplot['bias'], cmap=cmap1,
    #               vmin=-3, vmax=10)
    ax[2].set_xticks((0,5,10))
    ax[2].set_yticks((-1, 0, 1))
    ax[1].set_yticks((-1, 0, 1))
    ax[0].set_yticks((-1, -0.4, 1))
    ax[0].set_ylim((-1, 1))
    ax[2].set_xticks((-1,5,10))
    ax[1].set_yticks((-1, 0, 1))
    ax[0].set_yticks((-1, -0.4, 1))

    # ax[0].set_yticks((-1, -0.9, -0.8))
    # ax[1].set_yticks((-1, -0.9, -0.8))
    # ax[2].set_yticks((-0.4, -0.44, -0.42))
    # ax[1].set_ylim((-1.01, -0.8))
    # ax[1].set_xlim((0, 15))
    # ax[0].set_ylim((-1.05, -0.66))
    # ax[0].set_xlim((0, 12))
    # ax[2].set_ylim((-0.441, -0.405))
    # ax[2].set_xlim((2, 20))

    ax[3].set_ylabel('Wind speed (m/s)', fontsize=12)
    ax[0].set_ylabel('UWI (℃)', fontsize=12)
    ax[1].set_title('Parlung4')
    ax[2].set_title('Parlung94')
    ax[0].set_title('Yalong')
    # # 添加整体的颜色条
    # fig.colorbar(sca, ax=ax, label='Cooling (℃)',  pad=2,orientation='horizontal', aspect=40)
    for i in range(6):
        ax[i].grid(True, alpha=0.6)
        if i >2:
            ax[i].set_xlabel(r'$T_{est}$ (℃)', fontsize=12)
    # plt.show()
    plt.savefig('./figure/uwit2.jpg', dpi=300)


