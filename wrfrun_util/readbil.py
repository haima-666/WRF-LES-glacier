import os
from osgeo import gdal
import numpy as np

def read_bil(bil_file):
    # 打开 BIL 文件
    dataset = gdal.Open(bil_file)
    if dataset is None:
        raise FileNotFoundError(f"Cannot open file {bil_file}")

    # 获取影像基本信息
    XSize = dataset.RasterXSize  # 影像列数
    YSize = dataset.RasterYSize  # 影像行数
    projection = dataset.GetProjection()  # 投影信息
    geotransform = dataset.GetGeoTransform()  # 仿射矩阵
    bandnum = dataset.RasterCount  # 波段数

    # 读取波段数据
    bands_data = []
    for i in range(1, bandnum + 1):
        band = dataset.GetRasterBand(i)
        nodata = band.GetNoDataValue()
        data = band.ReadAsArray(0, 0, XSize, YSize)
        bands_data.append(data)

    bands_data = np.array(bands_data)
    return XSize, YSize, projection, geotransform, bandnum, nodata, bands_data

def main(bil_file):
    XSize, YSize, projection, geotransform, bandnum, nodata, bands_data = read_bil(bil_file)
    print("XSize:", XSize)
    print("YSize:", YSize)
    print("Projection:", projection)
    print("Geotransform:", geotransform)
    print("Band Number:", bandnum)
    print("NoData Value:", nodata)
    print("Data Shape:", bands_data.shape)

bil_file = r'C:\Users\admin\Desktop\xxx2.bil'  # 输入 BIL 数据文件路径
main(bil_file)
