import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import *
import matplotlib.pyplot as plt

df4990 = readfileshangxiawu('4990_2307-2310.csv')
df4990.time = df4990.time + pd.Timedelta(hours=1)
df4948 = readfileshangxiawu('4948_2307-2310.csv')
df4948.time = df4948.time + pd.Timedelta(hours=1)

df = pd.merge(df4948, df4990, on='time', how='inner')
df['lapse'] = (df['Tair_y'] - df['Tair_x'])/(4990- 4948)
df['T5261'] = df['Tair_x'] + df['lapse']*(5261- 4948)
df5261 = readfileshangxiawu('5261.csv')
df = pd.merge(df, df5261, on='time', how='inner')
df['bias'] = df['T5261'] - df['Tair']

plt.scatter(df['T5261'], -1*df['bias'], s=0.4)
# plt.plot(np.arange(-5, 20, 1), np.arange(-5, 20, 1), c='r')
plt.xlim((-5, 15))
plt.ylim((-10, 5))
plt.show()