import os
from osgeo import gdal
import numpy as np
def read_tif(tifFile):
    tif = gdal.Open(tifFile)
    XSize = tif.RasterXSize  # 影像列数
    YSize = tif.RasterYSize
    projection = tif.GetProjection()  # 投影信息
    geotransform = tif.GetGeoTransform()  # 仿射矩阵
    bandnum=tif.RasterCount
    data = tif.GetRasterBand(1)
    nodata = data.GetNoDataValue()
    tifdata = data.ReadAsArray(0, 0, XSize, YSize).astype(float)
    return XSize,YSize,projection,geotransform,bandnum,nodata,tifdata


def reclass_CLCD2USGS(data):
    data = np.where((data == 1), 3, data)
    data = np.where((data == 2), 15, data)
    data = np.where((data == 3) , 8, data)
    data = np.where((data == 4) , 7, data)
    data = np.where((data == 5), 16, data)
    data = np.where((data == 6), 23, data)
    data = np.where((data == 7) , 23, data)
    data = np.where((data == 8) , 1, data)
    data = np.where((data == 9), 17, data)

    return data

def reclass_CLCD2IGBP20(data):
    data = np.where((data == 1), 12, data)
    data = np.where((data == 2), 5, data)
    data = np.where((data == 3) , 6, data)
    data = np.where((data == 4) , 10, data)
    data = np.where((data == 5), 17, data)
    data = np.where((data == 6), 15, data)
    data = np.where((data == 7) , 16, data)
    data = np.where((data == 8) , 13, data)
    data = np.where((data == 9), 11, data)
    return data


# 输出tif
def creattif(DriverName, out_np, XSize, YSize, Bandnum, datatype, geotransform, projection, nodata, data):
    driver = gdal.GetDriverByName(DriverName)
    options = ['COMPRESS=LZW']  # 添加LZW压缩选项
    # dst_ds = driver.Create( dst_filename, 512, 512, 1, gdal.GDT_Byte )这句就创建了一个图像。宽度是512*512,单波段，数据类型是Byte这里要注意，它少了源数据，因为这里用不到源数据。它创建了一个空的数据集。要使它有东西，需要用其他步骤往里头塞东西。
    new_dataset = driver.Create(out_np, XSize, YSize, Bandnum, datatype, options=options)  # band[str(i)]
    new_dataset.SetGeoTransform(geotransform)  # 写入仿射变换参数
    new_dataset.SetProjection(projection)
    band_out = new_dataset.GetRasterBand(1)

    data_format = np.uint8(data)
    band_out.WriteArray(data_format)  # 写入数组数据
    band_out.SetNoDataValue(nodata)
    if DriverName == "GTiff":
        band_out.ComputeStatistics(True)
    del new_dataset
def main(intifFile,outpath,outname):
    XSize,YSize,projection,geotransform,bandnum,nodata,tifdata=read_tif(intifFile)
    newdata=reclass_CLCD2USGS(tifdata)
    datatype = gdal.GDT_Byte
    DriverName = "GTiff"
    outpn=os.path.join(outpath, outname)
    creattif(DriverName, outpn, XSize, YSize, bandnum, datatype, geotransform, projection, nodata, newdata)

intifFile = './staticdata/CLCD2021_clipp.tif'  #输入tif数据
outpath= './staticdata'  #输出路径
outname='CLCD2021_reclass_noice.tif'
main(intifFile,outpath,outname)