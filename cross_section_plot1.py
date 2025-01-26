import os.path
from math import acos, degrees, pi, cos, sin
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import numpy as np

import netCDF4 as nc
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 cartopy_xlim, cartopy_ylim, interpline, CoordPair, ll_to_xy)
from utils import *
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

####this script is to produce figure2 and figure3


plt.rc('font', family='Times New Roman')
def rotate_ua_va_vert_cross(ua_vertcross, va_vertcross):
    '''
    Takes u and v wind component vertcross, rotates them to align with
    transect meteorological direction, from start_point to end_point.
    '''

    coord_pairs_1 = to_np(ua_vertcross.coords["xy_loc"])
    coord_pairs_2 = to_np(va_vertcross.coords["xy_loc"])
    if (any(coord_pairs_1 != coord_pairs_2)):
        print("u-component and v component does not match")
        return
    coord_pairs = coord_pairs_1
    main_lat = [(x.lat) for x in coord_pairs]
    main_lon = [(x.lon) for x in coord_pairs]
    # Create an emptry transect
    met_dir_transect = []

    point_a = main_lat[0], main_lon[0]
    point_b = main_lat[1], main_lon[1]
    point_c = main_lat[0], main_lon[1]
    A = geodesic(point_a, point_b).km
    B = geodesic(point_b, point_c).km
    C = geodesic(point_c, point_a).km
    degrees_A_C = 90 + degrees(acos((A * A + C * C - B * B)/(2.0 * A * C)))
    met_dir_transect.append(degrees_A_C)

    for point in range(len(main_lat)):
        if point == 0:
            continue
        point_a = main_lat[point-1], main_lon[point-1]
        point_b = main_lat[point], main_lon[point]
        point_c = main_lat[point-1], main_lon[point]
        A = geodesic(point_a, point_b).km
        B = geodesic(point_b, point_c).km
        C = geodesic(point_c, point_a).km
        degrees_A_B = 180 - \
            degrees(acos((A * A + B * B - C * C)/(2.0 * A * B)))
        met_dir_transect.append(degrees_A_B)

    met_dir_transect_2 = np.array(met_dir_transect, ndmin=1)
    # if deg == True:
    a = met_dir_transect_2/180
    # else:
    #    a = met_dir_transect_2/pi
    c = [cos(pi*X) for X in a]
    s = [sin(pi*X) for X in a]
    c_tile = np.tile(c, (len(ua_vertcross.vertical), 1))
    s_tile = np.tile(s, (len(ua_vertcross.vertical), 1))
    # if clockwise == True:
    un = ua_vertcross * c_tile - va_vertcross * s_tile
    vn = ua_vertcross * s_tile + va_vertcross * c_tile
    # else:
    #    un = ua_cross * c_tile + va_cross * s_tile
    #    vn = va_cross * c_tile - ua_cross * s_tile
    return (vn, un)
def plot_process(x_cross):
    # x_cross1 = 10.0 * np.log10(x_cross)

    x_cross_filled = np.ma.copy(to_np(x_cross))

    for i in range(x_cross_filled.shape[-1]):
        column = x_cross_filled[:, i]
        first_idx = int(np.transpose((column > -200).nonzero())[0])
        x_cross_filled[0: first_idx, i] = x_cross_filled[first_idx, i]
    return x_cross_filled

colors = [(0, "white"), (1, "blue")]  # 颜色在 0, 0.5, 1 的位置

my_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

folder = './wrfout_file/laigu'
foldernocie = './wrfout_file/laigu_remove3'
# 加载 shp 文件

ncfilesall = os.listdir(folder)
ncfiles = ncfilesall[48:65]

ds1 = nc.Dataset(os.path.join(folder, ncfiles[0]), 'r+')
t2 = getvar(ds1, 'T2', timeidx=-1)
wrf_proj = get_cartopy(t2)

times = ['June 27 '+str(int(x[-8:-6])+8)+':'+str(x[-5:-3]) for x in ncfiles]

# 1.创建画布实例
fig3 = plt.figure(figsize=(16, 8))

# 2.创建“区域规划图”实例
spec2 = gridspec.GridSpec(1, 6, figure=fig3, wspace=1.7, hspace=0.5)  # 设置三行四列,并设置子图间预留的宽、高度量

# 3.根据给定的“区域规划图”，创建对应的坐标系实例
ax1 = fig3.add_subplot(spec2[0, 0:4])
ax2 = fig3.add_subplot(spec2[0, 4:6])
ax = ax2
axc = ax1
ax.set_ylim(0, 150)
ax.set_xlim(0, 3)


def p4t():
    points4 = [(20, 29.246, 96.929), (14, 29.258, 96.935), (26, 29.234, 96.923)]
    name4 = ['p1', 'p2', 'p3']

    co = ['orange', 'red', 'blue']


    temp_timemean = []
    temp_timemean_nofilled = []
    uwind = []
    wwind = []
    for i, file in enumerate(ncfiles):

        ncfile = nc.Dataset(os.path.join(folder, file))
        ncfile_noice = nc.Dataset(os.path.join(foldernocie, file))
        ht = getvar(ncfile, "z", timeidx=-1)

        # Define the cross section start and end points
        cross_start = CoordPair(lat=29.2288, lon=96.9210)
        cross_end = CoordPair(lat=29.2871, lon=96.9480)
        ter = getvar(ncfile, 'ter', timeidx=-1)


        w = getvar(ncfile, 'wa', timeidx=-1)
        u = getvar(ncfile, 'ua', timeidx=-1)
        v = getvar(ncfile, 'va', timeidx=-1)
        pt = getvar(ncfile, 'tc', timeidx=-1)
        pt_noice = getvar(ncfile_noice, 'tc', timeidx=-1)

        w_cross = vertcross(w, ht, wrfin=ncfile, start_point=cross_end,
                            end_point=cross_start, latlon=True, meta=True, autolevels=1000)
        u_cross = vertcross(u, ht, wrfin=ncfile, start_point=cross_end,
                            end_point=cross_start, latlon=True, meta=True, autolevels=1000)
        v_cross = vertcross(v, ht, wrfin=ncfile, start_point=cross_end,
                            end_point=cross_start, latlon=True, meta=True, autolevels=1000)

        pt_cross = vertcross(pt, ht, wrfin=ncfile, start_point=cross_end,
                             end_point=cross_start, latlon=True, meta=True, autolevels=10000)
        pt_cross_noice = vertcross(pt_noice, ht, wrfin=ncfile_noice, start_point=cross_end,
                                   end_point=cross_start, latlon=True, meta=True, autolevels=10000)
        pt_cross_filled = plot_process(pt_cross)
        pt_plot = pt_cross_noice - pt_cross
        pt_plot_filled = plot_process(pt_plot)

        h_wind_c, p_wind_c = rotate_ua_va_vert_cross(u_cross, v_cross)

        wind_ucross = to_np(p_wind_c)
        wind_wcross = to_np(w_cross)

        temp_timemean.append(to_np(pt_plot_filled))
        temp_timemean_nofilled.append(to_np(pt_plot))
        uwind.append(to_np(wind_ucross))
        wwind.append(to_np(wind_wcross))

    temp_mean = np.mean(temp_timemean, axis=0)
    temp_mean_nofilled = np.mean(temp_timemean_nofilled, axis=0)
    uwind_mean = np.mean(uwind, axis=0)
    wwind_mean = np.mean(wwind, axis=0)

    ter_line = interpline(ter, wrfin=ncfile, start_point=cross_end,
                          end_point=cross_start, latlon=True)

    w_levels = np.arange(0, 3.5, 0.25)
    xs = np.arange(0, pt_cross_filled.shape[-1], 1)
    ys = to_np(pt_cross.coords["vertical"])
    w_contours = axc.contourf(xs,
                                      ys,
                                      temp_mean,
                                      levels=w_levels,
                                      cmap=my_cmap,
                                      extend="both")

    filterr = 3
    xswind = np.arange(0, pt_cross_filled.shape[-1], 1)
    yswind = to_np(w_cross.coords['vertical'])
    axc.quiver(xswind[::filterr], yswind[::filterr], uwind_mean[::filterr, ::filterr],
                       wwind_mean[::filterr, ::filterr], scale=100)

    coord_pairs = to_np(u_cross.coords["xy_loc"])
    x_ticks = np.arange(coord_pairs.shape[0])
    x_labels = [f"({pair.lat:.3f}, {pair.lon:.3f})" for ii, pair in enumerate(to_np(coord_pairs))]

    num_ticks = 3
    thin = int((len(x_ticks) / num_ticks) + .5)
    axc.set_xticks(x_ticks[::thin])
    axc.set_xticklabels(x_labels[::thin], rotation=0, fontsize=10)

    ## create ice field
    a = to_np(ter_line.coords['xy_loc'])
    main_lat = [(x.lat) for x in a]

    icefield1 = (96.9365, 29.2623)

    mainlat1 = np.argmin(np.abs(np.array(main_lat) - icefield1[1]))
    ht_xs1 = np.arange(0, ter_line.shape[-1], 1)
    ht_fill = axc.fill_between(ht_xs1, 0, to_np(ter_line), facecolor="grey")
    ht_xs2 = np.arange(mainlat1, max(ht_xs1) + 1, 1)
    ht_fill1 = axc.fill_between(ht_xs2, 0, to_np(ter_line)[mainlat1: max(ht_xs1) + 1],
                                        facecolor="skyblue")

    # axc.set_title(f"{times[i]}", {"fontsize": 14})
    axc.set_ylim((4500, 5600))

    plotx = [xx[0] for xx in points4]
    for k, iii in enumerate(plotx):
        axc.axvline(x=iii, color=f'{co[k]}', linestyle='--', label='x=1')

    for ii in range(3, 6):
        axc.set_xlabel("Latitude, Longitude", fontsize=14)
    ax1.set_ylabel("Altitude (m)", fontsize=14)


    # cbar = fig.colorbar(w_contours, ax=axc, orientation="horizontal", pad=0.12, fraction=0.1, shrink=1, aspect=35)
    # cbar.ax.tick_params(labelsize=12)
    # cbar.set_label('Cooling (℃)', rotation=0, fontsize=14)

    ##高度图
    for jj, po in enumerate(points4):

        t_bias = temp_mean_nofilled[:, po[0]]
        ver = pt_cross.vertical
        ter1 = ter_line[po[0]]
        height = ver - ter1
        ax.plot(t_bias, height, c=co[jj], label=f'{name4[jj]}')
        # ax.plot(Temp, height, c=co[jj], label=f'{name4[jj]}')
        # ax.plot(temp_noice, height, c=co[jj], linestyle='--')


    ax.grid(alpha=0.7)
    ax.tick_params(axis='both', labelsize=12)
    axc.tick_params(axis='both', labelsize=12)

    axc.set_xlabel("Latitude, Longitude", fontsize=14)
    ax.set_xlabel("Cooling (℃)", fontsize=14)
    ax.set_ylabel("Height (m)", fontsize=14)
    axc.set_ylabel("Altitude (m)", fontsize=14)
    # ax.set_xticks((0, 5, 9))

    plt.savefig('./Figure/cross_height4mean.jpg', dpi =300)
    # plt.show()

def laigut():

    points = [(38, 29.296, 96.800), (43, 29.291, 96.809), (8, 29.329, 96.734), (24, 29.312, 96.768)]
    name = ['p1', 'p2', 'p3', 'p4']
    co = ['orange', 'red', 'blue', 'green']
    cross_start = CoordPair(lat=29.3380, lon=96.7180)
    cross_center = CoordPair(lat=29.2921, lon=96.8091)
    cross_end = CoordPair(lat=29.3021, lon=96.8079)
    temp_timemean = []
    temp_timemean_nofilled = []
    uwind = []
    wwind = []
    for i, file in enumerate(ncfiles):
        ncfile = nc.Dataset(os.path.join(folder, file))
        ncnoice = nc.Dataset(os.path.join(foldernocie, file))
        ht = getvar(ncfile, "z", timeidx=-1)
        w = getvar(ncfile, 'wa', timeidx=-1)
        u = getvar(ncfile, 'ua', timeidx=-1)
        v = getvar(ncfile, 'va', timeidx=-1)
        pt = getvar(ncfile, 'tc', timeidx=-1)
        pt_noice = getvar(ncnoice, 'tc', timeidx=-1)

        w_cross = vertcross(w, ht, wrfin=ncfile, start_point=cross_start,
                            end_point=cross_center, latlon=True, meta=True, autolevels=1000)
        u_cross = vertcross(u, ht, wrfin=ncfile, start_point=cross_start,
                            end_point=cross_center, latlon=True, meta=True, autolevels=1000)
        v_cross = vertcross(v, ht, wrfin=ncfile, start_point=cross_start,
                            end_point=cross_center, latlon=True, meta=True, autolevels=1000)

        w_cross2 = vertcross(w, ht, wrfin=ncfile, start_point=cross_center,
                             end_point=cross_end, latlon=True, meta=True, autolevels=1000)
        u_cross2 = vertcross(u, ht, wrfin=ncfile, start_point=cross_center,
                             end_point=cross_end, latlon=True, meta=True, autolevels=1000)
        v_cross2 = vertcross(v, ht, wrfin=ncfile, start_point=cross_center,
                             end_point=cross_end, latlon=True, meta=True, autolevels=1000)

        pt_cross = vertcross(pt, ht, wrfin=ncfile, start_point=cross_start,
                             end_point=cross_center, latlon=True, meta=True, autolevels=10000)
        pt_cross2 = vertcross(pt, ht, wrfin=ncfile, start_point=cross_center,
                              end_point=cross_end, latlon=True, meta=True, autolevels=10000)

        pt_cross_noice = vertcross(pt_noice, ht, wrfin=ncnoice, start_point=cross_start,
                                   end_point=cross_center, latlon=True, meta=True, autolevels=10000)
        pt_cross2_noice = vertcross(pt_noice, ht, wrfin=ncnoice, start_point=cross_center,
                                    end_point=cross_end, latlon=True, meta=True, autolevels=10000)

        pt_cross_filled = plot_process(pt_cross)
        pt_cross_filled2 = plot_process(pt_cross2)
        pt_cross_filled_noice = plot_process(pt_cross_noice)
        pt_cross_filled2_noice = plot_process(pt_cross2_noice)
        pt_cross_concat = np.hstack((to_np(pt_cross_filled), to_np(pt_cross_filled2)))
        pt_cross_concat_nocie = np.hstack((to_np(pt_cross_filled_noice), to_np(pt_cross_filled2_noice)))
        pt_plot = pt_cross_concat_nocie - pt_cross_concat

        pt_cross_concat_nofill = np.hstack((to_np(pt_cross), to_np(pt_cross2)))
        pt_cross_concat_nocie_nofill = np.hstack((to_np(pt_cross_noice), to_np(pt_cross2_noice)))
        pt_plot_nofill = pt_cross_concat_nocie_nofill - pt_cross_concat_nofill

        h_wind_c, p_wind_c = rotate_ua_va_vert_cross(u_cross, v_cross)
        h_wind_c2, p_wind_c2 = rotate_ua_va_vert_cross(u_cross2, v_cross2)
        wind_ucross = np.hstack((to_np(p_wind_c), to_np(p_wind_c2)))
        wind_wcross = np.hstack((to_np(w_cross), to_np(w_cross2)))


        temp_timemean.append(to_np(pt_plot))
        temp_timemean_nofilled.append(to_np(pt_plot_nofill))
        uwind.append(to_np(wind_ucross))
        wwind.append(to_np(wind_wcross))

    temp_mean = np.mean(temp_timemean, axis=0)
    temp_mean_nofilled = np.mean(temp_timemean_nofilled, axis=0)
    uwind_mean = np.mean(uwind, axis=0)
    wwind_mean = np.mean(wwind, axis=0)

    coord_pairs = to_np(pt_cross.coords["xy_loc"])
    x_ticks = np.arange(coord_pairs.shape[0])
    x_labels = [f"({pair.lat:.2f}, {pair.lon:.2f})" for i, pair in enumerate(to_np(coord_pairs))]

    w_levels = np.arange(0, 12, 1)
    xs = np.arange(0, pt_cross_concat.shape[-1], 1)
    ys = to_np(pt_cross.coords["vertical"])
    w_contours = axc.contourf(xs,
                              ys,
                              to_np(pt_plot),
                              levels=w_levels,
                              cmap=my_cmap,
                              extend="both")

    filterr = 3
    yswind = to_np(u_cross.coords['vertical'])
    axc.quiver(xs[::filterr], yswind[::filterr], uwind_mean[::filterr, ::filterr],
               wwind_mean[::filterr, ::filterr], scale=20)

    num_ticks = 4
    thin = int((len(x_ticks) / num_ticks) + .5)
    axc.set_xticks(x_ticks[::thin])
    axc.set_xticklabels(x_labels[::thin], rotation=0, fontsize=10)

    ter = getvar(ncfile, 'ter', timeidx=-1)
    ter_line = interpline(ter, wrfin=ncfile, start_point=cross_start,
                          end_point=cross_center)
    ter_line2 = interpline(ter, wrfin=ncfile, start_point=cross_center,
                           end_point=cross_end)
    ter_line_last = to_np(ter_line)[-1]
    ter_line2_ = np.insert(to_np(ter_line2), 0, ter_line_last)
    ht_xs1 = np.arange(0, ter_line.shape[-1], 1)
    ht_xs2 = np.arange(ht_xs1[-1], ht_xs1[-1] + ter_line2.shape[-1] + 1, 1)
    ht_fill = axc.fill_between(ht_xs1, 0, to_np(ter_line), facecolor="skyblue")
    ht_fill2 = axc.fill_between(ht_xs2, 0, to_np(ter_line2_), facecolor="grey")

    axc.set_title(f'{times[i]}', {"fontsize": 16})
    axc.set_ylim((3900, 4800))

    plotx = [xx[0] for xx in points]
    for k, iii in enumerate(plotx):
        axc.axvline(x=iii, color=f'{co[k]}', linestyle='--', label='x=1')

    for jj, po in enumerate(points):

        temp_bias = temp_mean_nofilled[:, po[0]]
        ver = pt_cross_noice.vertical
        ter1 = ter_line[po[0]]
        height = ver - ter1
        ax.plot(temp_bias, height, c=co[jj], label=f'{name[jj]}')

    axc.set_xlabel("Latitude, Longitude", fontsize=14)
    ax.set_xlabel("Cooling (℃)", fontsize=14)
    ax.set_ylabel("Height (m)", fontsize=14)
    axc.set_ylabel("Altitude (m)", fontsize=14)
    ax.set_xticks((0, 5, 9))
    # ax.legend()
    ax.grid(alpha=0.7)
    ax.tick_params(axis='both', labelsize=12)
    axc.tick_params(axis='both', labelsize=12)

    plt.savefig('Figure/cross_heightlaigumean.jpg', dpi=300)
    # plt.show()

def p94t():
    points94 = [(9, 29.393, 96.972), (12, 29.387, 96.975)]
    co = ['red', 'forestgreen']


    temp_timemean = []
    temp_timemean_nofilled = []
    uwind = []
    wwind = []
    for i, file in enumerate(ncfiles):

        ncfile = nc.Dataset(os.path.join(folder, file))
        ncfile_noice = nc.Dataset(os.path.join(foldernocie, file))
        ht = getvar(ncfile, "z", timeidx=-1)

        # Define the cross section start and end points
        cross_start = CoordPair(lat=29.378, lon=96.979)

        cross_end = CoordPair(lat=29.413, lon=96.966)

        ter = getvar(ncfile, 'ter', timeidx=-1)


        w = getvar(ncfile, 'wa', timeidx=-1)
        u = getvar(ncfile, 'ua', timeidx=-1)
        v = getvar(ncfile, 'va', timeidx=-1)
        pt = getvar(ncfile, 'tc', timeidx=-1)
        pt_noice = getvar(ncfile_noice, 'tc', timeidx=-1)

        w_cross = vertcross(w, ht, wrfin=ncfile, start_point=cross_end,
                            end_point=cross_start, latlon=True, meta=True, autolevels=1000)
        u_cross = vertcross(u, ht, wrfin=ncfile, start_point=cross_end,
                            end_point=cross_start, latlon=True, meta=True, autolevels=1000)
        v_cross = vertcross(v, ht, wrfin=ncfile, start_point=cross_end,
                            end_point=cross_start, latlon=True, meta=True, autolevels=1000)

        pt_cross = vertcross(pt, ht, wrfin=ncfile, start_point=cross_end,
                             end_point=cross_start, latlon=True, meta=True, autolevels=10000)
        pt_cross_noice = vertcross(pt_noice, ht, wrfin=ncfile_noice, start_point=cross_end,
                                   end_point=cross_start, latlon=True, meta=True, autolevels=10000)
        pt_cross_filled = plot_process(pt_cross)
        pt_plot = pt_cross_noice - pt_cross
        pt_plot_filled = plot_process(pt_plot)

        h_wind_c, p_wind_c = rotate_ua_va_vert_cross(u_cross, v_cross)

        wind_ucross = to_np(p_wind_c)
        wind_wcross = to_np(w_cross)

        temp_timemean.append(to_np(pt_plot_filled))
        temp_timemean_nofilled.append(to_np(pt_plot))
        uwind.append(to_np(wind_ucross))
        wwind.append(to_np(wind_wcross))

    temp_mean = np.mean(temp_timemean, axis=0)
    temp_mean_nofilled = np.mean(temp_timemean_nofilled, axis=0)
    uwind_mean = np.mean(uwind, axis=0)
    wwind_mean = np.mean(wwind, axis=0)

    ter_line = interpline(ter, wrfin=ncfile, start_point=cross_end,
                          end_point=cross_start, latlon=True)

    w_levels = np.arange(0, 2, 0.25)
    xs = np.arange(0, pt_cross_filled.shape[-1], 1)
    ys = to_np(pt_cross.coords["vertical"])
    w_contours = axc.contourf(xs,
                                      ys,
                                      temp_mean,
                                      levels=w_levels,
                                      cmap=my_cmap,
                                      extend="both")

    filterr = 3
    xswind = np.arange(0, pt_cross_filled.shape[-1], 1)
    yswind = to_np(w_cross.coords['vertical'])
    axc.quiver(xswind[::filterr], yswind[::filterr], uwind_mean[::filterr, ::filterr],
                       wwind_mean[::filterr, ::filterr], scale=100)

    coord_pairs = to_np(u_cross.coords["xy_loc"])
    x_ticks = np.arange(coord_pairs.shape[0])
    x_labels = [f"({pair.lat:.3f}, {pair.lon:.3f})" for ii, pair in enumerate(to_np(coord_pairs))]

    num_ticks = 3
    thin = int((len(x_ticks) / num_ticks) + .5)
    axc.set_xticks(x_ticks[::thin])
    axc.set_xticklabels(x_labels[::thin], rotation=0, fontsize=10)

    ## create ice field
    a = to_np(ter_line.coords['xy_loc'])
    main_lat = [(x.lat) for x in a]

    icefield1 = (96.972, 29.398)

    mainlat1 = np.argmin(np.abs(np.array(main_lat) - icefield1[1]))
    ht_xs1 = np.arange(0, ter_line.shape[-1], 1)
    ht_fill = axc.fill_between(ht_xs1, 0, to_np(ter_line), facecolor="grey")
    ht_xs2 = np.arange(mainlat1, max(ht_xs1) + 1, 1)
    ht_fill1 = axc.fill_between(ht_xs2, 0, to_np(ter_line)[mainlat1: max(ht_xs1) + 1],
                                        facecolor="skyblue")

    # axc.set_title(f"{times[i]}", {"fontsize": 14})
    axc.set_ylim((5000, 5800))

    plotx = [xx[0] for xx in points94]
    for k, iii in enumerate(plotx):
        axc.axvline(x=iii, color=f'{co[k]}', linestyle='--', label='x=1')

    for ii in range(3, 6):
        axc.set_xlabel("Latitude, Longitude", fontsize=14)
    ax1.set_ylabel("Altitude (m)", fontsize=14)


    # cbar = fig.colorbar(w_contours, ax=axc, orientation="horizontal", pad=0.12, fraction=0.1, shrink=1, aspect=35)
    # cbar.ax.tick_params(labelsize=12)
    # cbar.set_label('Cooling (℃)', rotation=0, fontsize=14)

    ##高度图
    for jj, po in enumerate(points94):

        t_bias = temp_mean_nofilled[:, po[0]]
        ver = pt_cross.vertical
        ter1 = ter_line[po[0]]
        height = ver - ter1
        ax.plot(t_bias, height, c=co[jj])
        # ax.plot(Temp, height, c=co[jj], label=f'{name4[jj]}')
        # ax.plot(temp_noice, height, c=co[jj], linestyle='--')



p94t()
ax.grid(alpha=0.7)
ax.tick_params(axis='both', labelsize=16)
axc.tick_params(axis='both', labelsize=16)

axc.set_xlabel("Latitude, Longitude", fontsize=18)
ax.set_xlabel("Cooling (℃)", fontsize=18)
ax.set_ylabel("Height (m)", fontsize=18)
axc.set_ylabel("Altitude (m)", fontsize=18)

# plt.savefig('./Figure/cross_height94mean.jpg', dpi=300)
plt.show()