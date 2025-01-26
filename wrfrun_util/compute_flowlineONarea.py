import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from shapely.ops import nearest_points
import xarray as xr
import rasterio
import netCDF4 as nc
import matplotlib as mpl
from utils import *
import os
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 cartopy_xlim, cartopy_ylim, interpline, CoordPair)

mpl.rcParams['figure.dpi'] = 150
plt.rc('font', family='Times New Roman')


def gcl_point(gdf, point, start_p=0):
    '''
    compute_gclofarea
    :param gdf:
    :param point:
    :param start_p:
    :return:
    '''
    start_point = gdf.geometry.interpolate(start_p, normalized=True)  # 获取起点坐标

    point_geom = Point(point)
    point_gdf = gpd.GeoDataFrame(geometry=[point_geom], crs='EPSG:4326')  # 使用WGS 84坐标系
    point_gdf = point_gdf.to_crs(gdf.crs)

    # 找到点在线上的投影点
    nearest = gpd.GeoSeries(nearest_points(point_gdf.geometry.unary_union, gdf.unary_union)[1])

    # 计算投影点到start起点的沿着gdf的距离
    distance_along_line = gdf.project(nearest[0]) - gdf.project(start_point[0])
    distance_along_line = distance_along_line[0]
    # # 计算gdf总长
    # total_length = gdf.length.values[0]
    return round(float(distance_along_line), 2)

def compute_gclofarea(centerline_shp, dem, glacierboundary):
    '''
    dem是wrfout文件中getvar出来的，用别的dem坐标需要修改
    
    :param centerline_shp: glacier centerline
    :param dem: region dem
    :param glacierboundary: glacier boundary
    :return:
    '''
    # 将 shapefile 转换为一个多边形对象
    centerline_shp = centerline_shp.to_crs(epsg=32646)
    total_length = float(np.array(centerline_shp['length_m']))
    polygon = glacierboundary.unary_union  # 如果是多个面，合并为一个
    lons = dem['XLONG']
    lats = dem['XLAT']
    # 使用掩膜裁剪 DEM 数据
    # 创建一个布尔掩膜，确定哪些点在多边形内
    mask = [[Point(lon, lat).within(polygon) for lon, lat in zip(row_lons, row_lats)] for row_lons, row_lats in
            zip(lons, lats)]
    mask = xr.DataArray(mask, dims=dem.dims, coords=dem.coords)
    clipped_dem = dem.where(mask, other=np.nan)
    nan_dem1 = xr.full_like(clipped_dem, np.nan)
    nan_dem2 = xr.full_like(clipped_dem, np.nan)

    for i in range(clipped_dem.shape[0]):
        for j in range(clipped_dem.shape[1]):
            if np.isnan(clipped_dem[i, j]):
                continue
            else:
                latt = clipped_dem['XLAT'][i, j]
                lonn = clipped_dem['XLONG'][i, j]
                point = (to_np(lonn), to_np(latt))
                flowline_d = gcl_point(centerline_shp, point, start_p=0)
                nan_dem1[i, j] = flowline_d
                nan_dem2[i, j] = flowline_d / total_length
    # 提取并转换投影信息
    if 'projection' in nan_dem1.attrs:
        projection = nan_dem1.attrs['projection']
        # 将投影对象转换为字符串或其他可序列化的格式
        nan_dem1.attrs['projection'] = str(projection)  # 或者提取具体参数并保存
    # 提取并转换投影信息
    if 'projection' in nan_dem2.attrs:
        projection = nan_dem2.attrs['projection']
        # 将投影对象转换为字符串或其他可序列化的格式
        nan_dem2.attrs['projection'] = str(projection)  # 或者提取具体参数并保存
    return nan_dem1, nan_dem2

# # 加载 Shapefile
# shapefile_path = '../glac_boundary/yalong_main.shp'  # 替换为实际路径
# shapefile = gpd.read_file(shapefile_path)
# yalongfl = gpd.read_file('../glac_boundary/yalongmainfl.shp')
# # 将 yalongfl 重采样为 WGS84 UTM Zone 46N（EPSG:32646），适合然乌湖地区
# yalongfl_utm = yalongfl.to_crs(epsg=32646)
#
# folderice = '../wrfout_file/laigu'
# foldernoice = '../wrfout_file/laigu_noice'
# file = 'wrfout_d01_2023-06-27_08_00_00'
# ncfileice = nc.Dataset(os.path.join(folderice, file))
# ncfilenoice = nc.Dataset(os.path.join(foldernoice, file))
# dem = getvar(ncfileice, 'ter')
# f1, f2 = compute_gclofarea(yalongfl_utm, dem, shapefile)
# plotraster(f1)



# f1.to_netcdf(r'C:\Users\admin\Desktop\code\WRF\staticdata\laiguflow.nc')


# lons = dem['XLONG']
# lats = dem['XLAT']
# # 将 shapefile 转换为一个多边形对象
# polygon = shapefile.unary_union  # 如果是多个面，合并为一个
#
# # 创建一个布尔掩膜，确定哪些点在多边形内
# mask = [[Point(lon, lat).within(polygon) for lon, lat in zip(row_lons, row_lats)] for row_lons, row_lats in zip(lons, lats)]
# mask = xr.DataArray(mask, dims=dem.dims, coords=dem.coords)
#
# # 使用掩膜裁剪 DEM 数据
# clipped_dem = dem.where(mask, other=np.nan)
# plotraster(clipped_dem)
# # 创建一个和 DEM 结构一样但全为 NaN 的 DataArray
# nan_dem1 = xr.full_like(clipped_dem, np.nan)
# nan_dem2 = xr.full_like(clipped_dem, np.nan)
# ##判断流线哪端是起点
# centerline_path = '../glac_boundary/yanong_gcl.shp'
# gdf = gpd.read_file(centerline_path)
# gdf = gdf.to_crs(epsg=32645)  # 转换为 WGS84 坐标系
# # 计算每条线段的长度
# gdf['length'] = gdf.length
# # 计算总长度
# total_length = gdf['length'].sum()
# # 获取线段的起点和终点
# # 2. 获取线段的起点和终点
# start_point = gdf.geometry.interpolate(0, normalized=True)  # 获取起点坐标
# end_point = gdf.geometry.interpolate(1, normalized=True)      # 获取终点坐标
#
#
# for i in range(clipped_dem.shape[0]):
#     for j in range(clipped_dem.shape[1]):
#         if np.isnan(clipped_dem[i, j]):
#             continue
#         else:
#             latt = clipped_dem['XLAT'][i, j]
#             lonn = clipped_dem['XLONG'][i, j]
#             point = (to_np(lonn), to_np(latt))
#             flowline_d = compute_gclofPoint(gdf,start_point, point)
#             nan_dem1[i, j] = flowline_d
#             nan_dem2[i, j] = flowline_d/total_length
