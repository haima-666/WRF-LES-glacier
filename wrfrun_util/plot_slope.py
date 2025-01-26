import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

def calculate_slope(array, resolution):
    dzdx,dzdy = np.gradient(array, resolution)
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))*(180/np.pi)
    return slope

def plotraster(data):
    # 可视化栅格数据
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar(label='HGT_M values')
    plt.title('Visualization of HGT_M')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

# User input for file name
# ncfile = input("Enter the netCDF file name (including path if not in the current directory): ")
# resolution = float(input("Enter res(m):"))
ncfile = './d05.nc'
resolution = 48
ncf = nc.Dataset(ncfile, 'r+')
dem = ncf.variables['HGT_M'][:][0]
dem = np.array(dem)
# slope = calculate_slope(dem, resolution)
plotraster(dem)