import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from wrfrun_util.compute_flowlineONarea import *
import rasterio
from rasterio.features import geometry_mask
from tqdm import tqdm
from utils import clip_vari
from scipy.optimize import curve_fit
from wrf import latlon_coords
### Find the relationship between windspeed and Tbias on Yanong
def grid_in_shp(grid, shape):
    # 计算 t2 数据的坐标网格
    lat, lon = latlon_coords(grid, as_np=True)

    # 构造空的掩膜数组，初始化为 False
    mask = np.full(grid.shape, False, dtype=bool)

    # 遍历每个格点
    for i in range(lat.shape[0]):
        for j in range(lat.shape[1]):
            point = gpd.points_from_xy([lon[i, j]], [lat[i, j]])

            # 检查点是否在 shapefile 边界内
            if shape.contains(point[0]).any():
                mask[i, j] = True

    # 将在掩膜范围内的格点保留，不在范围内的格点设置为 NaN
    grid_masked = xr.where(mask, grid, np.nan)
    return grid_masked

shp_file = 'glac_boundary/yalong_main.shp'
folder = './wrfout_file/laigu'
foldernoice = './wrfout_file/laigu_noice'
# 加载 shp 文件

ncfilesall = os.listdir(folder)
ncfiles = ncfilesall[48:61]

lonrange, latrange = (96.6999, 96.843), (29.278, 29.369)
gdf = gpd.read_file(shp_file)

for i, vv in enumerate(ncfiles):
    print('process' + f"{ncfiles[i]}")
    ncff = os.path.join(foldernoice, ncfiles[i])
    ncfice = os.path.join(folder, ncfiles[i])
    ds = nc.Dataset(ncff)
    dsice = nc.Dataset(ncfice)
    t2 = getvar(ds, 'T2', timeidx=-1)
    t2_c = clip_vari(t2, lonrange, latrange)
    t2ice = getvar(dsice, 'T2', timeidx=-1)
    t2ice_c = clip_vari(t2ice, lonrange, latrange)
    ws = getvar(dsice, 'wspd_wdir10', timeidx=-1)[0]
    ws_c = clip_vari(ws, lonrange,latrange)
    # 转换为 numpy 数组
    t2_np = to_np(t2_c)
    t2ice_np = to_np(t2ice_c)
    ws_np = to_np(ws_c)
    if i == 0:
        t2_timesum = np.zeros_like(t2_np)
        ws_timesum = np.zeros_like(ws_np)
    else:
        pass
    t2_bias = t2_np-t2ice_np
    t2_timesum = t2_timesum+ t2_bias
    ws_timesum = ws_timesum + ws_np

t2_timemean = t2_timesum/len(ncfiles)
ws_timemean = ws_timesum/len(ncfiles)
dem = getvar(dsice, 'ter')
dem = clip_vari(dem, lonrange, latrange)
slope = calculate_slope(dem, 240)
lat, lon = latlon_coords(t2_c, as_np=True)

# 构造空的掩膜数组，初始化为 False
mask = np.full(t2_timemean.shape, False, dtype=bool)

# 遍历每个格点
for i in range(lat.shape[0]):
    for j in range(lat.shape[1]):
        point = gpd.points_from_xy([lon[i, j]], [lat[i, j]])

        # 检查点是否在 shapefile 边界内
        if gdf.contains(point[0]).any():
            mask[i, j] = True

# 将在掩膜范围内的格点保留，不在范围内的格点设置为 NaN
ce_masked = xr.where(mask, t2_timemean, np.nan)
ws_masked = xr.where(mask, ws_timemean, np.nan)
slope_masked = xr.where(mask, slope, np.nan)

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制散点图，设置颜色映射和其他样式
scatter = ax.scatter(ws_masked,ce_masked, c=slope_masked, cmap='coolwarm', edgecolor='k', s=50, alpha=0.8)

# 添加颜色条，设置颜色条标签
cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Slope (°)', fontsize=14, labelpad=10)
cbar.ax.tick_params(labelsize=12)

# 拟合二次趋势线
xplot = np.ravel(ce_masked)
yplot = np.ravel(ws_masked)
slopemask1 = np.ravel(slope_masked)

nanmask = ~np.isnan(xplot) & ~np.isnan(yplot)
xplot1 = xplot[nanmask]
yplot1 = yplot[nanmask]
slopemask2 = slopemask1[nanmask]
slopemask = slopemask2 < 8
xplot11 = xplot1[slopemask]
yplot11 = yplot1[slopemask]
# coefficients = np.polyfit(yplot11,xplot11, 2)  # 二次多项式拟合
# trendline_x = np.linspace(min(yplot1), max(yplot1), 200)  # 用更高分辨率生成 x 轴值
# trendline_y = np.polyval(coefficients, trendline_x)  # 计算 y 轴值
def log_func(x, a, b):
    return a * np.log(x) + b

# 执行对数拟合
# 注意：如果 yplot11 中包含非正值，log(x) 可能无效
popt, pcov = curve_fit(log_func, yplot11, xplot11)

# 获取拟合参数
a, b = popt

# 生成趋势线
trendline_x = np.linspace(min(yplot11), max(yplot11), 200)
trendline_y = log_func(trendline_x, a, b)

# 绘制趋势线
ax.plot(trendline_x, trendline_y, color='red', linewidth=2, linestyle='--', label='Quadratic Fit')

# 设置坐标轴标签和标题
ax.set_ylabel('Cooling (°C)', fontsize=14)
ax.set_xlabel('Wind Speed (m/s)', fontsize=14)
# ax.set_title('Scatter Plot of T2 vs WS with Slope as Color', fontsize=16)

# 调整刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlim(0, 8)
ax.set_ylim(3, 12)
ax.grid(alpha=0.7)
# 显示图表
plt.tight_layout()
plt.savefig('Figure/ws_cooling_scatter.jpg', dpi=300)
# plt.show()