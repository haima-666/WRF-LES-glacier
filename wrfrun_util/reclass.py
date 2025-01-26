import os
from osgeo import gdal
import numpy as np

def read_tif(tifFile):
    tif = gdal.Open(tifFile)
    XSize = tif.RasterXSize  # 影像列数
    YSize = tif.RasterYSize
    projection = tif.GetProjection()  # 投影信息
    geotransform = tif.GetGeoTransform()  # 仿射矩阵
    bandnum = tif.RasterCount
    data = tif.GetRasterBand(1)
    nodata = data.GetNoDataValue()
    tifdata = data.ReadAsArray(0, 0, XSize, YSize).astype(float)
    return XSize, YSize, projection, geotransform, bandnum, nodata, tifdata

def reclass(data, nodata_value=0):
    # 将值为 0 的像元设置为 NoData
    # data = np.where(data == 0 ,14 , data)
    return data

# 输出tif
def creat_tif(DriverName, out_np, XSize, YSize, Bandnum, datatype, geotransform, projection, nodata, data):
    driver = gdal.GetDriverByName(DriverName)
    options = ['COMPRESS=LZW']  # 使用 LZW 压缩
    new_dataset = driver.Create(out_np, XSize, YSize, Bandnum, datatype, options=options)
    new_dataset.SetGeoTransform(geotransform)
    new_dataset.SetProjection(projection)
    band_out = new_dataset.GetRasterBand(1)
    band_out.WriteArray(data)
    band_out.SetNoDataValue(nodata)  # 设置 NoData 值
    band_out.ComputeStatistics(True)
    del new_dataset

def main(intifFile, outpath, outname):
    XSize, YSize, projection, geotransform, bandnum, nodata, tifdata = read_tif(intifFile)
    newdata = reclass(tifdata, nodata_value=128)  # 将值为 0 的像元设置为 255 (8 位整数的 NoData 值)
    datatype = gdal.GDT_Byte  # 使用 8 位整数
    DriverName = "GTiff"
    outpn = os.path.join(outpath, outname)
    creat_tif(DriverName, outpn, XSize, YSize, bandnum, datatype, geotransform, projection, 255, newdata.astype(np.uint8))

intifFile = r'C:\Users\admin\Desktop\新建文件夹\reclassnonglacier.tif'  # 输入 tif 数据
outpath = r'C:\Users\admin\Desktop\code\WRF\staticdata'  # 输出路径
outname = 'CLCDnoiceout2.tif'
main(intifFile, outpath, outname)
