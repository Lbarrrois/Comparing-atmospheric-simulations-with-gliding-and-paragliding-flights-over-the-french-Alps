#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import skgstat
from reader import Reader
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from datetime import datetime
from datetime import date as date_creator
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from numba import njit
import pandas as pd
import geopandas as gpd
import os
from numba import jit,njit
from tqdm import tqdm
import math
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import xarray as xr
import string
from scipy.interpolate import RegularGridInterpolator
from shapely import plotting

get_ipython().run_line_magic('matplotlib', 'inline')



# Extents
extent_son = (5.499792, 7.032229, 44.104446, 45.18888)
extent_father = ()
img_extent = (4.7942, 8.1545, 43.3545, 46.6707)



#Save paths
savetraces_path = 'T:/C2H/STAGES/Wiki_glide/Nparray/traces/'
savejson_path = 'T:/C2H/STAGES/Wiki_glide/Json/'
savefig_path = 'T:/C2H/STAGES/Wiki_glide/Figures/'
savenet_path = 'T:/C2H/STAGES/Wiki_glide/Netcdf/'
savenpa_path = 'T:/C2H/STAGES/Wiki_glide/Nparray/'
savecsv_path = 'T:/C2H/STAGES/Wiki_glide/Csv/'
savegif_path = 'T:/C2H/STAGES/Wiki_glide/Figures/GIF_TH/'
saveflights_path = 'T:/C2H/STAGES/Wiki_glide/Vols/'
savesta_path = 'T:/C2H/STAGES/Wiki_glide/Stations/'
savesat_path = 'T:/C2H/STAGES/Wiki_glide/Satellites/Images_MODIS_VIIRS/'



#Plotting
size = 25
params = {'legend.fontsize': 15,
#          'figure.figsize': (20,8),
          'axes.labelsize': size*0.8,
          'axes.titlesize': size*0.8,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'axes.titlesize': 25.0,
          'axes.titlepad': 25}
plt.rcParams.update(params)



#Selecting indisde list/array/xarray
def liste_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(path + f)]
    
@njit()
def merge_to_max(values1,values2):
    n,m = np.shape(values1)
    values = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if values1[i,j]>= values2[i,j] :
                values[i,j] =  values1[i,j]
            else :
                values[i,j] =  values2[i,j]
    return values

@njit
def get_cell_coordinates(lons, lats,longitude, latitude):
    cell_longitude = int((longitude-lons[0]) / (lons[1] - lons[0]))
    cell_latitude = int((latitude-lats[0]) / (lats[1] - lats[0]))
    
    return cell_latitude,cell_longitude

@njit
def find_closest_neighbour_inside_tab(lat_l, lon_l, lat, lon):
    h,k = np.shape(lat_l)
    dist = np.ones((h,k))
    for i in range(h):
        for j in range(k):
            dist[i,j] = (lat_l[i,j] - lat)**2 + (lon_l[i,j] - lon)**2
    return np.argmin(dist)//h,np.argmin(dist)%k

@njit
def find_closest_neighbour_inside_list(lat_l, lon_l, lat, lon):
    dist = np.arange(len(lat_l))
    for i in range(len(lat_l)):
        dist[i] = (lat_l[i] - lat)**2 + (lon_l[i] - lon)**2
    return np.argmin(dist)

def select_inside_xarray(xarray,img_extent,time_step,iteration,type):
    
    if type == 'simu' :
        selected_events = xarray.where(
            (xarray['lat'] >= img_extent[2]) & (xarray['lat'] <= img_extent[3]) &
            (xarray['lon'] >= img_extent[0]) & (xarray['lon'] <= img_extent[1]) &
            (xarray['iteration'] >= iteration),
            drop=True
        )
        
    elif type == 'traces' :
        selected_events = xarray.where(
            (xarray['latitude_inf'] >= img_extent[2]) & (xarray['latitude_inf'] <= img_extent[3]) &
            (xarray['latitude_sup'] >= img_extent[2]) & (xarray['latitude_sup'] <= img_extent[3]) &
            (xarray['longitude_sup'] >= img_extent[0]) & (xarray['longitude_sup'] <= img_extent[1]) &
            (xarray['longitude_inf'] >= img_extent[0]) & (xarray['longitude_inf'] <= img_extent[1]) &
            (xarray['time_stemp'] == time_step),
            drop=True
        )       

    elif type == 'comp' :
        selected_events = xarray.where(
            (xarray['iteration'] == iteration),
            drop=True
        )

    elif type == 'comp_topo' :
        selected_events = xarray.where(
            (xarray['iteration_i'] == iteration),
            drop=True
        ) 

    elif type == 'comp_T':
        selected_events = xarray.where(
            (xarray['iteration_T'] == iteration),
            drop=True
        ) 

    elif type == 'speed' :
        selected_events = xarray.where(
            (xarray['latitude'] >= img_extent[2]) & (xarray['latitude'] <= img_extent[3]) &
            (xarray['longitude'] >= img_extent[0]) & (xarray['longitude'] <= img_extent[1]) &
            (xarray['hours'] == time_step),
            drop=True)
            
    return selected_events

def select_inside_pdframe_2(df,iteration,extent,type):
    if type == 'net':
        selected_pd = df[(df['lat'] >=extent[2]) & (df['lat'] <= extent[3]) &
              (df['lon'] >= extent[0]) & (df['lon'] <= extent[1]) &
            (df['rawtime'] == 10*60+6*60*60+iteration*10*60)]

    elif type == 'mf':
        selected_pd = df[(df['lat'] >=extent[2]) & (df['lat'] <= extent[3]) &
              (df['lon'] >= extent[0]) & (df['lon'] <= extent[1]) &
            (df['rawtime'] == 10*60+6*60*60+iteration*10*60)]

    return selected_pd

def select_inside_pdframe(df,iteration,extent,type):
    if type == 'net':
        selected_pd = df[(df['lat'] >=extent[2]) & (df['lat'] <= extent[3]) &
              (df['lon'] >= extent[0]) & (df['lon'] <= extent[1]) &
            (df['rawtime'] == iteration*3600)]

    elif type == 'mf':
        selected_pd = df[(df['lat'] >=extent[2]) & (df['lat'] <= extent[3]) &
              (df['lon'] >= extent[0]) & (df['lon'] <= extent[1]) &
            (df['rawtime'] == iteration*3600)]

    return selected_pd

@njit()
def reducing(lon_simu,lat_simu,extent_i,Simu_array):
    ind = np.argwhere((lon_simu > extent_i[0]) & (lon_simu < extent_i[1]) & (lat_simu > extent_i[2]) & (lat_simu < extent_i[3]))
    return Simu_array[:,ind[:,0]]

@njit
def find_indice_min_dist(liste_lat,liste_lon,latitude,longitude):
    n = len(liste_lat)
    res = np.zeros((1,n))[0]
    for i in range(n):
            res[i] = np.sqrt((liste_lat[i]-latitude)**2 + (liste_lon[i]-longitude)**2)
    return np.argmin(res)

def pre_max_height_traces_carto(values,all,nlat,nlon,lons,lats):
    all_lon = all[:,0]
    all_lat = all[:,1]
    all_alti = all[:,2] 
    for j,k,l in zip(all_lon,all_lat,all_alti) :
        if np.min(lons) <= j <= np.max(lons) and np.min(lats) <= k <= np.max(lats) :
            cell_coordinates = get_cell_coordinates(lons, lats,j, k)
            if values[cell_coordinates[0],cell_coordinates[1]] == 0 :
                values[cell_coordinates[0],cell_coordinates[1]] = l
            else :
                if l > values[cell_coordinates[0],cell_coordinates[1]] :
                    values[cell_coordinates[0],cell_coordinates[1]] = l
    return values




#Diurn cycle meteorological variables
def compute_diurn_cycle(var_simu, var_sta,  percent,Comp_simu_rvuv_19):
    mediane_sta = [[] for i in range(24)]
    mediane_simu = [[] for i in range(24)]
    
    for i in range(24) :

        comp_it_simu = Comp_simu_rvuv_19.where(
        (Comp_simu_rvuv_19['heure'] == i),
            drop = True)
        
        mediane_sta[i] = np.nanpercentile(comp_it_simu[var_sta],percent)
        mediane_simu[i] = np.nanpercentile(comp_it_simu[var_simu],percent)

    return mediane_sta,mediane_simu
    
def medians_rvuv(Comp_simu_rvuv_19):
    #    return mediane_net,mediane_simu_net,mediane_mf,mediane_simu_mf
    var = ['T_simu','T_sta','U_moy_simu','U_moy_sta','Rv_simu','Rv_sta','P_simu','P_sta']
    
    medians = np.zeros((8,24))
    lows = np.zeros((8,24))
    highs = np.zeros((8,24))
    for i in range(4) :
        medians[2*i,:] = compute_diurn_cycle(var[2*i], var[2*i+1],50,Comp_simu_rvuv_19)[0]
        medians[2*i+1,:] = compute_diurn_cycle(var[2*i], var[2*i+1],50,Comp_simu_rvuv_19)[1]
    
        highs[2*i,:] = compute_diurn_cycle(var[2*i], var[2*i+1],75,Comp_simu_rvuv_19)[0]
        highs[2*i+1,:] = compute_diurn_cycle(var[2*i], var[2*i+1],75,Comp_simu_rvuv_19)[1]
    
        lows[2*i,:] = compute_diurn_cycle(var[2*i], var[2*i+1],25,Comp_simu_rvuv_19)[0]
        lows[2*i+1,:] = compute_diurn_cycle(var[2*i], var[2*i+1],25,Comp_simu_rvuv_19)[1]

    return lows,medians,highs

def all_traces_diurn_rvuv(Ffather_19,Ffather_20,Sson_19,Sson_20,AaROME_19,AaROME_20):

    lows_19_f,medians_19_f,highs_19_f = medians_rvuv(Ffather_19)
    lows_20_f,medians_20_f,highs_20_f = medians_rvuv(Ffather_20)

    lows_f = np.concatenate((lows_19_f,lows_20_f), axis = 1)
    medians_f = np.concatenate((medians_19_f,medians_20_f), axis = 1)
    highs_f = np.concatenate((highs_19_f,highs_20_f), axis = 1)

    lows_19_s,medians_19_s,highs_19_s = medians_rvuv(Sson_19)
    lows_20_s,medians_20_s,highs_20_s = medians_rvuv(Sson_20)

    lows_s = np.concatenate((lows_19_s,lows_20_s), axis = 1)
    medians_s = np.concatenate((medians_19_s,medians_20_s), axis = 1)
    highs_s = np.concatenate((highs_19_s,highs_20_s), axis = 1)

    lows_19_A,medians_19_A,highs_19_A = medians_rvuv(AaROME_19)
    lows_20_A,medians_20_A,highs_20_A = medians_rvuv(AaROME_20)

    lows_A = np.concatenate((lows_19_A,lows_20_A), axis = 1)
    medians_A = np.concatenate((medians_19_A,medians_20_A), axis = 1)
    highs_A = np.concatenate((highs_19_A,highs_20_A), axis = 1)

    return lows_f,medians_f,highs_f,lows_s,medians_s,highs_s,lows_A,medians_A,highs_A





#Diurn cycle thermalling ascents
def new_diurn_cycle_th(thermals_daily,simu_daily):
    res =  np.zeros((6,24))
    for i in range(23):
        
        reduced_thermal_19_it = thermals_daily.where(
            (thermals_daily['rawtime'] >= i*60*60) & (thermals_daily['rawtime'] <= (i+1)*60*60),
            drop = True)

        if len(reduced_thermal_19_it['type']) != 0 :
            reduced_simu_it = simu_daily.where(
                (simu_daily['heure'] == i),
                drop = True)

            extent_son = (5.499792, 7.032229, 44.104446, 45.18888)
            ngridcell = 100 
            max_speed_th = new_scatter_th(reduced_thermal_19_it,reduced_simu_it,extent_son,ngridcell)

            res[0,i] = np.nanpercentile(np.ravel(max_speed_th[0]),25)
            res[1,i] = np.nanpercentile(np.ravel(max_speed_th[0]),50)
            res[2,i] = np.nanpercentile(np.ravel(max_speed_th[0]),75)
            res[3,i] = np.nanpercentile(np.ravel(max_speed_th[1]),25)
            res[4,i] = np.nanpercentile(np.ravel(max_speed_th[1]),50)
            res[5,i] = np.nanpercentile(np.ravel(max_speed_th[1]),75)

    for i in range(23):
        for j in range(6):
            if res[j,i] == 0 :
                res[j,i] = np.nan

    return res




#Diurn cycle ABLH
def all_traces_diurn_BHL_2(reduced_BHL_father_19,reduced_BHL_father_20,reduced_BHL_son_19,reduced_BHL_son_20,reduced_BHL_AROME_19,reduced_BHL_AROME_20):

    lows_19_f,medians_19_f,highs_19_f = fill_medians(reduced_BHL_father_19)
    lows_20_f,medians_20_f,highs_20_f = fill_medians(reduced_BHL_father_20)

    lows_f = np.concatenate((lows_19_f,lows_20_f), axis = 1)
    medians_f = np.concatenate((medians_19_f,medians_20_f), axis = 1)
    highs_f = np.concatenate((highs_19_f,highs_20_f), axis = 1)

    lows_19_s,medians_19_s,highs_19_s = fill_medians(reduced_BHL_son_19)
    lows_20_s,medians_20_s,highs_20_s = fill_medians(reduced_BHL_son_20)

    lows_s = np.concatenate((lows_19_s,lows_20_s), axis = 1)
    medians_s = np.concatenate((medians_19_s,medians_20_s), axis = 1)
    highs_s = np.concatenate((highs_19_s,highs_20_s), axis = 1)

    lows_19_A,medians_19_A,highs_19_A = fill_medians(reduced_BHL_AROME_19)
    lows_20_A,medians_20_A,highs_20_A = fill_medians(reduced_BHL_AROME_20)

    lows_A = np.concatenate((lows_19_A,lows_20_A), axis = 1)
    medians_A = np.concatenate((medians_19_A,medians_20_A), axis = 1)
    highs_A = np.concatenate((highs_19_A,highs_20_A), axis = 1)

    return lows_f,medians_f,highs_f,lows_s,medians_s,highs_s,lows_A,medians_A,highs_A





# Meteorological variables
def potential_temp(temp_i,press_i,alt_i,type):
    gamma = 2/7
    rho = 1
    g = 9.81
    p0 = 100000
    
    if type == 'net':
        if np.isnan(temp_i) == False and np.isnan(alt_i) == False :
            press_net = p0-rho*g*alt_i
            t_pot=(temp_i+273.15)*(press_net/p0)**gamma - 273.15
        else :
            t_pot = np.nan
    
    elif type == 'mf':
        
        if np.isnan(temp_i) == False :
            if np.isnan(press_i) == False :
                t_pot = temp_i*(press_i/p0)**gamma - 273.15
            elif np.isnan(alt_i) == False :
                press_net = p0-rho*g*alt_i
                t_pot=(temp_i)*(press_net/p0)**gamma - 273.15
            else :
                t_pot = np.nan
        else :
            t_pot = np.nan 
    return t_pot

@njit()
def compute_potential_T(press_liste, T_list):
    p0 = 100000
    gamma = 2/7
    return T_list*(p0/press_liste)**gamma

@njit
def compute_press(alt_i):
    rho = 1
    g = 9.81
    p0 = 100000
    press_net = p0-rho*g*alt_i
    return press_net

def compute_rv_from_rh(liste_P,liste_T_paspot,liste_Rh):
    rv_sta = mixing_ratio_from_relative_humidity(liste_P*units.Pa, liste_T_paspot*units.degK, liste_Rh/100)
    return rv_sta

@njit()
def filled_press_values(liste_press,liste_alt_1,liste_alt_2):
    rho = 1
    g = 9.81
    p0 = 100000
    res = np.zeros((1,len(liste_press)))[0]
    for i in range(len(liste_press)):
        if np.isnan(liste_press[i]) == True :
            if np.isnan(liste_alt_1[i]) == False :
                res[i] = p0-rho*g*liste_alt_1[i]
            elif np.isnan(liste_alt_2[i]) == False :
                res[i] = p0-rho*g*liste_alt_2[i]
            else :
                res[i] = np.nan
        else :
            res[i] = liste_press[i]
    return res

@njit()
def p_sat(liste_T):
    P0 = 100000
    M = 0.018
    Lv = 2.26*10**6
    T0 = 373.15
    R = 8.314
    return P0*np.exp(-(M*Lv/R)*((1/liste_T) - (1/T0)))

@njit()
def compute_rv_from_hu(liste_hu,liste_press,liste_T):
    Mvap, Mair = 0.018, 29.0 
    Psat = p_sat(liste_T)
    res = (Mvap/Mair)/((liste_press/Psat)*(100/liste_hu)-1)
    return res
        




# Interpolation inverse distance weighting
def simple_idw(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi

def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])

    return np.hypot(d0, d1)





# Comparisons
@njit()
def compare_ground_W_max(lon_inf,lon_sup,lat_inf,lat_sup,Glide_array,lon_simu,lat_simu,Simu_array,type):
    
    res = np.zeros((1,len(lon_inf)))[0]
    res_trace = np.zeros((1,len(lon_inf)))[0]
    res_simu = np.zeros((1,len(lon_inf)))[0]

    for i in range(len(lon_inf)):
        
        extent_i = (lon_inf[i],lon_sup[i],lat_inf[i],lat_sup[i])
        reduced_array = Simu_array[(lon_simu<extent_i[1]) & (lon_simu>extent_i[0]) & (lat_simu<extent_i[3]) & (lat_simu>extent_i[2])]

        if len(reduced_array) != 0 : 
            BLH_simu_array = np.max(Simu_array[(lon_simu<extent_i[1]) & (lon_simu>extent_i[0]) & (lat_simu<extent_i[3]) & (lat_simu>extent_i[2])])
            res[i] = Glide_array[i] - BLH_simu_array
            res_trace[i] = Glide_array[i]
            res_simu[i] = BLH_simu_array
            
        else :
            res[i] = np.nan
            res_trace[i] = np.nan
            res_simu[i] = np.nan
            
    return res

def compare_ground_rvuv(lon_inf,lon_sup,lat_inf,lat_sup,Glide_array,lon_simu,lat_simu,Simu_array):

    dlon, dlat = 0.01,0.01

    res = np.zeros((6,len(lon_inf)))
    res_trace = np.zeros((6,len(lon_inf)))
    res_simu = np.zeros((6,len(lon_inf)))

    for i in range(len(lon_inf)):
        
        extent_i = (lon_inf[i],lon_sup[i],lat_inf[i],lat_sup[i])
        reduced_array = reducing(lon_simu,lat_simu,extent_i,Simu_array)
        
        if len(reduced_array[0]) != 0 :
            lons = lon_simu[(lon_simu > extent_i[0]) & (lon_simu < extent_i[1]) & 
            (lat_simu > extent_i[2]) & (lat_simu < extent_i[3])]
            lats = lat_simu[(lon_simu > extent_i[0]) & (lon_simu < extent_i[1]) & 
            (lat_simu > extent_i[2]) & (lat_simu < extent_i[3])]

            x,y = lons, lats
            lon, lat = lon_inf[i] + dlon, lat_inf[i] + dlat
            for var in range(6):
                z = reduced_array[var,:]
                if np.shape(x) == np.shape(y) and np.shape(y) == np.shape(z) :
                    interpolate = simple_idw(x, y, z, lon, lat)[0]
                    res[var,i] = Glide_array[var,i]-interpolate
                    res_trace[var,i] = Glide_array[var,i]
                    res_simu[var,i] = interpolate
                else :
                    res[var,i] = np.nan
                    res_trace[var,i] = np.nan
                    res_simu[var,i] = np.nan

        else :
            res[:,i] = np.nan
            res_trace[:,i] = np.nan
            res_simu[:,i] = np.nan
        
    return res,res_trace,res_simu

@njit()
def compare_ground_W_max(lon_inf,lon_sup,lat_inf,lat_sup,Glide_array,lon_simu,lat_simu,Simu_array,type):
    
    res = np.zeros((1,len(lon_inf)))[0]
    res_trace = np.zeros((1,len(lon_inf)))[0]
    res_simu = np.zeros((1,len(lon_inf)))[0]

    for i in range(len(lon_inf)):
        
        extent_i = (lon_inf[i],lon_sup[i],lat_inf[i],lat_sup[i])
        reduced_array = Simu_array[(lon_simu<extent_i[1]) & (lon_simu>extent_i[0]) & (lat_simu<extent_i[3]) & (lat_simu>extent_i[2])]

        if len(reduced_array) != 0 : 
            BLH_simu_array = np.max(Simu_array[(lon_simu<extent_i[1]) & (lon_simu>extent_i[0]) & (lat_simu<extent_i[3]) & (lat_simu>extent_i[2])])
            res[i] = Glide_array[i] - BLH_simu_array
            res_trace[i] = Glide_array[i]
            res_simu[i] = BLH_simu_array
            
        else :
            res[i] = np.nan
            res_trace[i] = np.nan
            res_simu[i] = np.nan
            
    return res

@njit()
def compare_ground_topo(lon_inf,lon_sup,lat_inf,lat_sup,lon_simu,lat_simu,topo_array):

    mean_ = np.zeros((1,len(lon_inf)))[0]
    hquart_ = np.zeros((1,len(lon_inf)))[0]
    std_ = np.zeros((1,len(lon_inf)))[0]

    for i in range(len(lon_inf)):
        extent_i = (lon_inf[i],lon_sup[i],lat_inf[i],lat_sup[i])
        elevation_i = topo_array[(lon_simu<extent_i[1]) & (lon_simu>extent_i[0]) & (lat_simu<extent_i[3]) & (lat_simu>extent_i[2])]
       
        if len(elevation_i) != 0 :
            mean_[i] = np.mean(elevation_i)
            hquart_[i] = np.percentile(elevation_i,90)
            std_[i] = np.std(elevation_i)

        else :
            mean_[i] = np.nan
            hquart_[i] = np.nan
            std_[i] = np.nan
        
    return mean_,hquart_,std_

def fill_traces(Comp_simu_rvuv_19):
    #    return mediane_net,mediane_simu_net,mediane_mf,mediane_simu_mf
    var = ['T_simu','T_sta','U_moy_simu','U_moy_sta','Rv_simu','Rv_sta','P_simu','P_sta']
    
    medians = np.zeros((8,24))
    lows = np.zeros((8,24))
    highs = np.zeros((8,24))
    for i in range(4) :
        medians[2*i,:] = compute_diurn_cycle(var[2*i], var[2*i+1],50,Comp_simu_rvuv_19)[0]
        medians[2*i+1,:] = compute_diurn_cycle(var[2*i], var[2*i+1],50,Comp_simu_rvuv_19)[1]
    
        highs[2*i,:] = compute_diurn_cycle(var[2*i], var[2*i+1],75,Comp_simu_rvuv_19)[0]
        highs[2*i+1,:] = compute_diurn_cycle(var[2*i], var[2*i+1],75,Comp_simu_rvuv_19)[1]
    
        lows[2*i,:] = compute_diurn_cycle(var[2*i], var[2*i+1],25,Comp_simu_rvuv_19)[0]
        lows[2*i+1,:] = compute_diurn_cycle(var[2*i], var[2*i+1],25,Comp_simu_rvuv_19)[1]

    return lows,medians,highs




# Traces
def compute_dist(lat1,lon1,lat2,lon2,rad=True):
    if not(rad):
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)
    # approximate radius of earth in m
    R = 6373_000.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c
    
def smooth(x,window_len=11,window='hanning'):
        if x.ndim != 1:
                raise ValueError("smooth only accepts 1 dimension arrays.")
        if x.size < window_len:
                raise ValueError("Input vector needs to be bigger than window size.")
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]

def remove_zero_from_alti(alti):
    if alti[0] < 10 :
        alti[0] = alti[1]
    if alti[-1] < 10 :
        alti[-1] = alti[-2]
    for i,alt in enumerate(alti):
        if alt < 10 :
            if i >= len(alti)-1 :
                alti[i] = (alti[i-1]+alti[i])/2
            else :  
                alti[i] = (alti[i-1]+alti[i+1])/2
    return alti

def read_igc(file):
    with open(file, 'r') as f:
        parsed_igc_file = Reader().read(f)
    previous_lat = 0
    previous_lon = 0
    
    all_lon = np.zeros(len(parsed_igc_file['fix_records'][1]))
    all_lat = np.zeros(len(parsed_igc_file['fix_records'][1]))
    all_speed=np.zeros(len(parsed_igc_file['fix_records'][1]))
    all_vz=np.zeros(len(parsed_igc_file['fix_records'][1]))
    all_alti=np.zeros(len(parsed_igc_file['fix_records'][1]))
    all_time=[0 for _ in range(len(parsed_igc_file['fix_records'][1]))]

    for i,record in enumerate(parsed_igc_file['fix_records'][1]):
        record['time'] = record['time'].replace(hour=record['time'].hour + 1)
        all_lat[i] = record['lat']
        all_lon[i] = record['lon']
        all_time[i] = record['time']        
    return all_lon,all_lat,all_alti,all_time

def get_date_time_dif(start_time,stop_time):
    date = date_creator(1, 1, 1)
    datetime1 = datetime.combine(date, start_time)
    datetime2 = datetime.combine(date, stop_time)
    time_elapsed = datetime1 - datetime2
    return time_elapsed.total_seconds()

def reshape_array(arr,time_vid):
    nb_img_by_sec = 24
    
    t_true = np.linspace(time_vid[0], time_vid[-1], num=len(time_vid), endpoint=True)
    t_inter = np.linspace(time_vid[0], time_vid[-1], num=int(len(time_vid)*nb_img_by_sec/speed_acc), endpoint=True)
    f = interp1d(t_true, arr, kind='cubic')

    return f(t_inter)

def smooth_igc_output(L_all):
    all_ret = []
    for l_val in L_all:
        l_val[0]=l_val[1]#=np.mean(l_val[:int(len(l_val)/10)])
        smoothed = smooth(l_val,50,'hanning')
        all_ret.append(smoothed)
    return all_ret

def plot_smooth_non_smooth(smooth,non_smooth):
    plt.figure(figsize=(18,9))
    plt.plot(non_smooth)
    plt.plot(smooth)
    plt.show()

def get_last_date_of_all_raw_file(path_raw_file):
    delta_time_writing = 20
    all_ending_time = []
    for file in os.listdir(path_raw_file):
        if "_11_" in file :
            time_end = os.path.getmtime(path_raw_file+'\\'+file)
            all_ending_time.append(datetime.fromtimestamp(time_end-delta_time_writing).time())
    return all_ending_time

def convert_time_to_sec(all_time):
    for i in range(len(all_time)):
        all_time[i] = all_time[i].hour*3600 + all_time[i].minute*60 + all_time[i].second
    return np.array(all_time,dtype=np.float32)

def compute_v(lon_,lat_,alt_,raw_t_):
    
    n = len(lon_)
    speed_ = np.zeros((n-1,4))
    
    for i in range(n-1):
        dt = raw_t_[i+1] - raw_t_[i]
        if dt > 0 :
            d_lat = compute_dist(lat_[i],lon_[i],lat_[i+1],lon_[i],rad=False)
            d_lon = compute_dist(lat_[i],lon_[i],lat_[i],lon_[i+1],rad=False)
            d_alt = alt_[i+1]-alt_[i]
            
            speed_[i,0] = d_lon/dt
            speed_[i,1] = d_lat/dt
            speed_[i,2] = d_alt/dt
            speed_[i,3] = raw_t_[i]

    return speed_

@njit
def bearing_rate(speed):
    
    n = np.shape(speed)[0]
    bearting_rate_ = np.zeros((1,n-1))[0]
    
    for i  in range(n-2):
        
        dt = speed[i+1,3] - speed[i,3]
        norm1 = np.sqrt(speed[i+1,0]**2 + speed[i+1,1]**2)
        norm2 = np.sqrt(speed[i,0]**2 + speed[i,1]**2)
        
        if dt > 0 and norm1 != 0 and norm2 != 0 :
            
            dotp = np.dot(speed[i+1,:2], speed[i,:2])
            bearing = dotp/(norm1*norm2)
            bearting_rate_[i] = np.arccos(bearing)/dt*(180/np.pi)
            
    return bearting_rate_

def find_consecutive(list,value):
    n = len(list)
    k=0
    indices = []
    while k < len(list):
        i = 0
        if list[k] == value :
            takeoff = k
            while i+k < len(list) and (list[i+k] == value or np.isnan(list[i+k]) == True) == True :
                i = i + 1
            if i < 25 :
                landing = k+i
                indices.append((takeoff,landing))
        k = k + i + 1
    return indices

@njit()
def build_mask(indices,len_ind):
    res = np.zeros((1,len_ind))[0]
    for i in indices:
        for j in range(i[1]-i[0]):
            res[i[0]+j] = -10
    return res

@njit()
def build_mask_2(indices,len_ind):
    res = np.zeros((1,len_ind))[0]
    for i in range(len(indices)-1):
#        if indices[i+1][0] - indices[i][-1] < 30 : 
        for j in range(indices[i+1][0] - indices[i][-1]):
            res[indices[i][-1]+j] = -10
    return res