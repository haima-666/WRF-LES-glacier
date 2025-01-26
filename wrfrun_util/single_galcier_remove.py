import wrf
import numpy as np
from netCDF4 import Dataset
import geopandas as gpd
from shapely.geometry import Point

# 读取 WRF 输入文件
ncfile = Dataset('../wrfinput_d01', 'r+')
lu_index = ncfile.variables['LU_INDEX'][0]
# 读取 shapefile
shapefile = '../glac_boundary/p4.shp'
gdf = gpd.read_file(shapefile)

# 确保 shapefile 使用 WGS84 坐标系
gdf = gdf.to_crs(epsg=4326)
# 获取 WRF 文件中的经纬度信息
lats = wrf.getvar(ncfile, 'XLAT').values
lons = wrf.getvar(ncfile, 'XLONG').values
# 创建与 LU_INDEX 相同维度的掩膜矩阵
mask = np.zeros_like(lu_index, dtype=bool)

# 遍历 LU_INDEX 矩阵，判断每个格点是否在 shapefile 面内部
for i in range(lu_index.shape[0]):
    for j in range(lu_index.shape[1]):
        # 获取格点的经纬度
        point = Point(lons[i, j], lats[i, j])

        # 判断该格点是否在 shapefile 面内
        if gdf.contains(point).any():
            mask[i, j] = True
# 将掩膜内的 LU_INDEX 值改为 23
lu_index[mask] = 23
# 将修改后的 LU_INDEX 写回原 WRF 文件
luout = np.reshape(lu_index, (1, lu_index.shape[0], lu_index.shape[1]))
ncfile.variables['LU_INDEX'][:] = luout

LANDUSEF = ncfile.variables['LANDUSEF']
landf22 = LANDUSEF[:, 22, :, :][0]
landf23 = LANDUSEF[:, 23, :, :][0]

landf22[mask] = 1
landf23[mask] = 0
landf22_out = np.reshape(landf22, (1, landf22.shape[0], landf22.shape[1]))
landf23_out = np.reshape(landf23, (1, landf22.shape[0], landf22.shape[1]))
ncfile.variables['LANDUSEF'][:, 22, :, :] = landf22_out
ncfile.variables['LANDUSEF'][:, 23, :, :] = landf23_out
ncfile.close()
