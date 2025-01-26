import netCDF4 as nc
from wrf import getvar, to_np, ll_to_xy
from utils import *
import matplotlib.pyplot as plt
df1 = readfile(r'C:\Users\admin\Desktop\wrfout_laiguon.csv')
df2 = readfile(r'C:\Users\admin\Desktop\laigu2023_6to7.csv')

def t_filter(df):
    dfo = df[(df['time']>='2023-6-27')&(df['time']<'2023-6-28')]
    return dfo

df1 = t_filter(df1)
df2 = t_filter(df2)

plt.scatter(df1['time'], df1['wd'], marker='*', c='r')
plt.scatter(df2['time'], df2['WD'], marker='o', c='b')
plt.show()