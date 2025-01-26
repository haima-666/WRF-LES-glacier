import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

def one_two_one_smooth(array, npass):
    row = array.shape[0]
    column = array.shape[1]
    smootharray = np.copy(array)
    for n in range(npass):
        for i in range(row):
            for j in range(0, column, 1):

                if j == 0:
                    smootharray[i, j] = array[i, j]*0.6+0.4*array[i, j+1]
                elif j == column-1:
                    smootharray[i, j] = array[i, j]*0.6+0.4*array[i, j-1]
                else:
                    smootharray[i, j] = array[i, j] * 0.5 + 0.25 * (array[i, j - 1] + array[i, j + 1])

        for j in range(column):
            for i in range(0, row, 1):
                if i == 0:
                    array[i, j] = smootharray[i, j]*0.6+0.4*smootharray[i+1, j]
                elif i == row-1:
                    array[i, j] = smootharray[i, j] * 0.6 + 0.4 * smootharray[i -1, j]
                else:
                    array[i,j] = smootharray[i, j]*0.5 + 0.25*(smootharray[i-1, j]+ smootharray[i+1, j])

    return array

def calculate_slope(array, resolution):
    dzdx,dzdy = np.gradient(array, resolution)
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))*(180/np.pi)
    return slope

def gaussian_kernel(size, sigma=1):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            - ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)
        ),
        (size, size)
                    )
    return kernel / np.sum(kernel)


def gaosi_smooth(image, kernel_size, resolution):
    kernel = gaussian_kernel(kernel_size)
    slope = calculate_slope(image, resolution)
    kernel_h, kernel_w = kernel.shape

    pad_height = (kernel_h-1) // 2
    pad_width = (kernel_w - 1) // 2
    padded_image = np.pad(image, ((pad_height, pad_height),(pad_width,pad_width)), mode='constant', constant_values=0)
    output = np.copy(image)
    indices = np.argwhere(slope > 20)
    # 打乱索引的顺序
    np.random.shuffle(indices)
    for index in indices:
        i, j = index
        region = padded_image[i: i+kernel_h, j:j+kernel_w]
        if np.any(region == 0):
            non_zerovalue = region[region != 0]
            newregion = np.where(region == 0,np.mean(non_zerovalue), region)
            output[i, j] = np.sum(newregion*kernel)
        else:
            output[i, j] = np.sum(region*kernel)
    return output


def one_two_one_highslope(array, resolution, npass):
    row, column = array.shape[0], array.shape[1]
    smootharray = np.copy(array)
    slope = calculate_slope(array, resolution)
    for n in range(npass):
        for i in range(row):
            for j in range(2, column-2, 1):
                if slope[i, j] > 25:
                    smootharray[i, j] = (0.1* array[i,j] + 0.25* (array[i, j-1] + array[i, j+1])
                                         + 0.2*(array[i, j-2] + array[i, j+2]) )
                elif 20 < slope[i ,j] < 25:
                    smootharray[i, j] = (0.8* array[i,j])  + 0.1*(array[i, j-1] + array[i, j+1])
                elif 15 <slope[i, j] < 20:
                    smootharray[i, j] = 0.8 * array[i, j] + 0.1 * (array[i, j - 1] + array[i, j + 1])
                else:
                    smootharray[i, j] = 0.8 * array[i, j] + 0.1 * (array[i, j - 1] + array[i, j + 1])
            smootharray[i, 1] = array[i, 1] * 0.6 + (array[i, 0] + array[i, 2]) * 0.2
            smootharray[i, 0] = array[i, 0]*0.6 + array[i, 1]*0.4
            smootharray[i, column - 2] = array[i, column - 2] * 0.6 + (
                        array[i, column - 1] + array[i, column - 3]) * 0.2
            smootharray[i, column-1] = array[i, column-1] * 0.6 + array[i, column-2] * 0.4


        for j in range(column):
            for i in range(2, row -2, 1):
                if slope[i, j] > 25:
                    array[i, j] = (0.1* smootharray[i,j] + 0.25* (smootharray[i-1, j] + smootharray[i+1, j])
                                         + 0.2*(smootharray[i-2, j] + smootharray[i+2, j]) )
                elif 20 < slope[i ,j] < 25:
                    array[i, j] = 0.8* smootharray[i,j] + 0.1* (smootharray[i-1, j] + smootharray[i+1, j])

                elif  15 < slope[i, j] < 20:
                    array[i, j] = 0.8* smootharray[i, j] + 0.1 * (smootharray[i-1, j] + smootharray[i+1, j])

                else:
                    array[i, j] = 0.8 * smootharray[i, j] + 0.1 * (smootharray[i - 1, j] + smootharray[i + 1, j])
            array[1, j] = smootharray[1, j]*0.6 + (smootharray[0, j] + smootharray[2, j])*0.2
            array[0, j] = smootharray[0, j]*0.6 + smootharray[1, j]*0.4
            array[row-2, j] = smootharray[row-2, j]*0.6 + (smootharray[row-1, j] + smootharray[row-3, j])*0.2
            array[row-1, j] = smootharray[row-1, j]*0.6 + smootharray[row-2, j]*0.4

    return array



# User input for file name
ncfile = input("Enter the netCDF file name (including path if not in the current directory): ")

resolution = float(input("Enter res(m):"))


ncf = nc.Dataset(ncfile, 'r+')
dem = ncf.variables['HGT_M'][:][0]
dem = np.array(dem)
demout1 = np.copy(dem)

demout1 = one_two_one_smooth(demout1,1)
for i in range(3):
    demout1 = gaosi_smooth(demout1, 5, resolution)



demout = np.reshape(demout1, (1, demout1.shape[0], demout1.shape[1]))

# Write the result back to the netCDF file
ncf.variables['HGT_M'][:] = demout

# Close the netCDF file
ncf.close()





