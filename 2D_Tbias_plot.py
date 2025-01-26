##24/10/5 plot Tbias from sensitivity experiment
import numpy as np
import xarray as xr
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 cartopy_xlim, cartopy_ylim, interpline, CoordPair)
import netCDF4 as nc
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas as gpd
import matplotlib.ticker as mticker
import os
from matplotlib.cm import get_cmap
import shapefile
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
plt.rc('font', family='Times New Roman')
def clip_vari(vari, lonrange, latrange):
    '''

    :param vari: wrfout中的变量
    :param lonrange: 类似(96.46, 96.85)
    :return:
    '''
    # 获取经纬度数据
    xlon = vari['XLONG'][:]
    xlat = vari['XLAT'][:]
    xlon = xlon.values
    xlat = xlat.values

    # 定义矩形区域的坐标范围
    x_max = lonrange[1]
    x_min = lonrange[0]
    y_min = latrange[0]
    y_max = latrange[1]

    ymax_index = np.unravel_index(np.argmin(np.abs(xlat - y_max)), np.abs(xlat - y_max).shape)[0]
    ymin_index = np.unravel_index(np.argmin(np.abs(xlat - y_min)), np.abs(xlat - y_min).shape)[0]
    xmax_index = np.unravel_index(np.argmin(np.abs(xlon - x_max)), np.abs(xlon - x_max).shape)[1]
    xmin_index = np.unravel_index(np.argmin(np.abs(xlon - x_min)), np.abs(xlon - x_min).shape)[1]

    # ws = ws.values
    out = vari[ymin_index:ymax_index, xmin_index:xmax_index]

    return out

def p4():
    # 加载 shp 文件
    shp_file = 'glac_boundary/p4.shp'  # 替换为你的文件路径
    gdf = gpd.read_file(shp_file)

    folder = './wrfout_file/laigu'
    x_extent = (96.895, 96.955)
    y_extent = (29.210, 29.286)
    ncfilesall = os.listdir(folder)
    ncfiles = ['wrfout_d01_2023-06-27_02_00_00','wrfout_d01_2023-06-27_04_00_00',
               'wrfout_d01_2023-06-27_06_00_00','wrfout_d01_2023-06-27_07_00_00',
               'wrfout_d01_2023-06-27_08_00_00','wrfout_d01_2023-06-27_09_00_00',
               'wrfout_d01_2023-06-27_10_00_00','wrfout_d01_2023-06-27_12_00_00']
    # ncfiles = ['wrfout_d01_2023-06-27_02_00_00',
    #            'wrfout_d01_2023-06-27_06_00_00',
    #            'wrfout_d01_2023-06-27_08_00_00',
    #            'wrfout_d01_2023-06-27_12_00_00',]

    ds1 = nc.Dataset(os.path.join(folder, ncfiles[0]), 'r+')
    t2 = getvar(ds1, 'T2', timeidx=-1)
    wrf_proj = get_cartopy(t2)


    times = ['June 27 '+str(int(x[-8:-6])+8)+':'+str(x[-5:-3]) for x in ncfiles]

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(11,10),subplot_kw={'projection': wrf_proj})
    # fig, axs = plt.subplots(nrows=2, ncols=4,subplot_kw={'projection': wrf_proj})

    for i, ax in enumerate(axs.flat):
        print('process' + f"{ncfiles[i]}")
        ncff = os.path.join(folder, ncfiles[i])
        ds = nc.Dataset(ncff, 'r+')

        u10 = getvar(ds, 'U10', timeidx=-1)
        v10 = getvar(ds, 'V10', timeidx=-1)
        t2 = getvar(ds, 'T2', timeidx=-1) - 273.15

        t2_clip = clip_vari(t2, lonrange=x_extent, latrange=y_extent)
        u10_clip = clip_vari(u10, lonrange=x_extent, latrange=y_extent)
        v10_clip = clip_vari(v10, lonrange=x_extent, latrange=y_extent)

        # 转换为 numpy 数组
        t2_np = to_np(t2_clip)
        u10_np = to_np(u10_clip)
        v10_np = to_np(v10_clip)
        gdf.plot(ax=ax, color='none', edgecolor='lime', linewidth=1.5, transform=ccrs.PlateCarree())
        # # Get the latitude and longitude points
        lats, lons = latlon_coords(t2_clip)
        aa= 2
        cws = ax.quiver(to_np(lons)[::aa, ::aa], to_np(lats)[::aa, ::aa], to_np(u10_clip[::aa, ::aa]), to_np(v10_clip[::aa, ::aa]),
                        transform=ccrs.PlateCarree(), scale=100)

        # shp2clip(cws, ax, shpfile='./glac_boundary/yalong.shp', proj=wrf_proj, vcplot=True)

        levels = np.arange(0, 10, 0.5)
        cs = ax.contourf(lons, lats, t2_np,
                           levels=levels,
                           transform=ccrs.PlateCarree(),
                           cmap=get_cmap("bwr"), # rianbow, BuGn
                           zorder=0, #图层顺序
                           extend='both')

        # 绘制 shp 文件

        ax.set_title(times[i], {"fontsize" : 14})

        # 使用 gridlines 添加刻度标签，但隐藏网格线
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.MaxNLocator(nbins=2)  # 例如，限制为最多 5 个经度刻度
        gl.ylocator = mticker.MaxNLocator(nbins=2)  # 例如，限制为最多 5 个纬度刻度
        #############
        gl.xlabels_top = False
        gl.xlines = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10, 'rotation': 0}  # 控制经度刻度字体大小
        gl.ylabel_style = {'size': 10, 'rotation': 90}  # 控制纬度刻度字体大小
        gl.xpadding = 10
        gl.ypadding = 10
        # 设置经纬度显示范围 (xmin, xmax, ymin, ymax)
        ax.set_extent([x_extent[0]+0.005, x_extent[1]-0.005, y_extent[0]+0.005, y_extent[1]-0.005], crs=ccrs.PlateCarree())

    cbar = plt.colorbar(cs, ax=axs, orientation="horizontal", pad=0.08, fraction=0.1, shrink=1, aspect = 50)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('2m Tair (℃)', fontsize=16)
    plt.savefig(r'C:\Users\admin\Desktop\冷却效应小论文\图\Smater\p4_240m_2D.jpg', dpi=300)
    # plt.show()
#
def p94():
    # 加载 shp 文件
    shp_file = 'glac_boundary/p94.shp'  # 替换为你的文件路径
    gdf = gpd.read_file(shp_file)

    folder = './wrfout_file/laigu'
    x_extent = (96.950, 97.004)
    y_extent = (29.364, 29.415)
    ncfilesall = os.listdir(folder)
    ncfiles = ['wrfout_d01_2023-06-27_02_00_00','wrfout_d01_2023-06-27_04_00_00',
    'wrfout_d01_2023-06-27_06_00_00','wrfout_d01_2023-06-27_07_00_00',
    'wrfout_d01_2023-06-27_08_00_00','wrfout_d01_2023-06-27_09_00_00',
    'wrfout_d01_2023-06-27_10_00_00','wrfout_d01_2023-06-27_12_00_00']
    # ncfiles = ['wrfout_d01_2023-06-27_02_00_00',
    #            'wrfout_d01_2023-06-27_06_00_00',
    #            'wrfout_d01_2023-06-27_08_00_00',
    #            'wrfout_d01_2023-06-27_12_00_00', ]

    ds1 = nc.Dataset(os.path.join(folder, ncfiles[0]), 'r+')
    t2 = getvar(ds1, 'T2', timeidx=-1)
    wrf_proj = get_cartopy(t2)

    times = ['June 27 ' + str(int(x[-8:-6]) + 8) + ':' + str(x[-5:-3]) for x in ncfiles]

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(11, 10), subplot_kw={'projection': wrf_proj})
    # fig, axs = plt.subplots(nrows=2, ncols=4,subplot_kw={'projection': wrf_proj})

    for i, ax in enumerate(axs.flat):
        print('process' + f"{ncfiles[i]}")
        ncff = os.path.join(folder, ncfiles[i])
        ds = nc.Dataset(ncff, 'r+')

        u10 = getvar(ds, 'U10', timeidx=-1)
        v10 = getvar(ds, 'V10', timeidx=-1)
        t2 = getvar(ds, 'T2', timeidx=-1) - 273.15

        t2_clip = clip_vari(t2, lonrange=x_extent, latrange=y_extent)
        u10_clip = clip_vari(u10, lonrange=x_extent, latrange=y_extent)
        v10_clip = clip_vari(v10, lonrange=x_extent, latrange=y_extent)

        # 转换为 numpy 数组
        t2_np = to_np(t2_clip)
        u10_np = to_np(u10_clip)
        v10_np = to_np(v10_clip)
        gdf.plot(ax=ax, color='none', edgecolor='lime', linewidth=1.5, transform=ccrs.PlateCarree())
        # # Get the latitude and longitude points
        lats, lons = latlon_coords(t2_clip)
        aa = 2
        cws = ax.quiver(to_np(lons)[::aa, ::aa], to_np(lats)[::aa, ::aa], to_np(u10_clip[::aa, ::aa]),
                        to_np(v10_clip[::aa, ::aa]),
                        transform=ccrs.PlateCarree(), scale=100)

        # shp2clip(cws, ax, shpfile='./glac_boundary/yalong.shp', proj=wrf_proj, vcplot=True)

        levels = np.arange(0, 10, 0.5)
        cs = ax.contourf(lons, lats, t2_np,
                         levels=levels,
                         transform=ccrs.PlateCarree(),
                         cmap=get_cmap("bwr"),  # rianbow, BuGn
                         zorder=0,  # 图层顺序
                         extend='both')

        # 绘制 shp 文件

        ax.set_title(times[i], {"fontsize": 14})

        # 使用 gridlines 添加刻度标签，但隐藏网格线
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.MaxNLocator(nbins=3)  # 例如，限制为最多 5 个经度刻度
        gl.ylocator = mticker.MaxNLocator(nbins=3)  # 例如，限制为最多 5 个纬度刻度
        #############
        gl.xlabels_top = False
        gl.xlines = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10, 'rotation': 0}  # 控制经度刻度字体大小
        gl.ylabel_style = {'size': 10, 'rotation': 90}  # 控制纬度刻度字体大小
        gl.xpadding = 10
        gl.ypadding = 10
        # 设置经纬度显示范围 (xmin, xmax, ymin, ymax)
        ax.set_extent([x_extent[0] + 0.005, x_extent[1] - 0.005, y_extent[0] + 0.005, y_extent[1] - 0.005],
                      crs=ccrs.PlateCarree())

    cbar = plt.colorbar(cs, ax=axs, orientation="horizontal", pad=0.08, fraction=0.1, shrink=1, aspect=50)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('2m Tair (℃)', fontsize=16)
    plt.savefig(r'C:\Users\admin\Desktop\冷却效应小论文\图\Smater\p94_240m_2D.jpg', dpi=300)
    # plt.show()

def icebias():
    colors = [(0, 'grey'), (0.5, "white"), (1, "blue")]  # 颜色在 0, 0.5, 1 的位置

    my_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    plt.rc('font', family='Times New Roman')



    # 加载 shp 文件
    folder = './wrfout_file/laigu'
    foldernoice = './wrfout_file/laigu_noice'
    shp_file = 'glac_boundary/p4.shp'  # 替换为你的文件路径
    gdf = gpd.read_file(shp_file)
    # 打开 NetCDF 文件
    ncfilesall = os.listdir(folder)
    ncfiles = ncfilesall[24:]
    ncfiles = [x for x in ncfiles if x[-5:-3] == '00' and (int(x[-8:-6]) + 2) % 2 == 0]
    ncfiles.insert(-1, ncfilesall[-5])
    ds1 = nc.Dataset(os.path.join(folder, ncfiles[0]), 'r+')
    t2 = getvar(ds1, 'T2', timeidx=-1)
    wrf_proj = get_cartopy(t2)

    times = ['June 27 ' + str(int(x[-8:-6]) + 8) + ':' + str(x[-5:-3]) for x in ncfiles]

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 16), subplot_kw={'projection': wrf_proj})

    # #p94
    # lonrange = (96.954, 96.995)
    # latrange = (29.376, 29.407)

    # # laigu local
    # lonrange = (96.526, 96.839)
    # latrange = (29.248, 29.425)

    # p4
    lonrange = (96.887, 96.948)
    latrange = (29.194, 29.278)
    levels = np.arange(0, 6, 0.25)

    for i, ax in enumerate(axes.flat):
        print('process' + f"{ncfiles[i]}")
        ncff = os.path.join(foldernoice, ncfiles[i])
        ncfice = os.path.join(folder, ncfiles[i])
        ds = nc.Dataset(ncff)
        dsice = nc.Dataset(ncfice)
        t2 = getvar(ds, 'T2', timeidx=-1)
        t2ice = getvar(dsice, 'T2', timeidx=-1)
        t2_clip = clip_vari(t2, (lonrange[0] - 0.02, lonrange[1] + 0.02), (latrange[0] - 0.02, latrange[1] + 0.02))
        t2ice_clip = clip_vari(t2ice, (lonrange[0] - 0.02, lonrange[1] + 0.02),
                               (latrange[0] - 0.02, latrange[1] + 0.02))

        # 转换为 numpy 数组
        t2_np = to_np(t2_clip)
        t2ice_np = to_np(t2ice_clip)
        t2_bias = t2_np - t2ice_np
        # # Get the latitude and longitude points
        lats, lons = latlon_coords(t2_clip)
        # aa= 3
        # cws = ax.quiver(to_np(lons)[::aa, ::aa], to_np(lats)[::aa, ::aa], to_np(u10[::aa, ::aa]), to_np(v10[::aa, ::aa]), transform=ccrs.PlateCarree(), scale=100)

        # shp2clip(cws, ax, shpfile='./glac_boundary/yalong.shp', proj=wrf_proj, vcplot=True)


        cs = ax.contourf(lons, lats, t2_bias,
                         levels=levels,
                         transform=ccrs.PlateCarree(),
                         cmap=my_cmap,  # rianbow, BuGn
                         zorder=0,  # 图层顺序
                         extend='both')

        # 绘制 shp 文件
        gdf.plot(ax=ax, color='none', edgecolor='lime', linewidth=0.5, transform=ccrs.PlateCarree())
        ax.set_title(times[i], {"fontsize": 18})

        # 使用 gridlines 添加刻度标签，但隐藏网格线
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.MaxNLocator(nbins=5)  # 例如，限制为最多 5 个经度刻度
        gl.ylocator = mticker.MaxNLocator(nbins=5)  # 例如，限制为最多 5 个纬度刻度
        #############
        gl.xlabels_top = False
        gl.xlines = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 14, 'rotation': 0, 'ha': 'right'}  # 控制经度刻度字体大小
        gl.ylabel_style = {'size': 14, 'rotation': 90}  # 控制纬度刻度字体大小
        gl.xpadding = 10
        gl.ypadding = 10
        # 设置经纬度显示范围 (xmin, xmax, ymin, ymax)
        ax.set_extent([lonrange[0], lonrange[1], latrange[0], latrange[1]], crs=ccrs.PlateCarree())

    cbar = plt.colorbar(cs, ax=axes, orientation="horizontal", pad=0.08, fraction=0.1, shrink=1, aspect=50)
    cbar.set_label('2m Tair Bias (℃)', fontsize=18)
    # 设置 colorbar 刻度的字体大小
    cbar.ax.tick_params(labelsize=16)  # 设置刻度字体大小
    plt.savefig(r'C:\Users\admin\Desktop\冷却效应小论文\图\Smater\4icebias240_2D.jpg', dpi=300)
    # plt.show()


test()