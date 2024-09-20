# Title: CAT Risk Modelling Sandbox
# Author: Arnaud Mignan
# Date: 17.05.2024
# Description: Functions associated to the notebook CATRiskModellingSandbox_tutorial.ipynb to model risks in a virtual environment.
# License: MIT
# Version: 1.1
# Dependencies: numpy, pandas, matplotlib, json, networkx, os, scipy, re, imageio, skimage
# Contact: arnaud@mignanriskanalytics.com
# Citation: Mignan, A. (2025), Introduction to Catastrophe Risk Modelling – A Physics-based Approach. Cambridge University Press, DOI: 10.1017/9781009437370

#Copyright 2024 A. Mignan
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), 
#to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
#and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
#IN THE SOFTWARE.


# Equation and section numbers provided below are from the textbook.

import json
import matplotlib.pyplot as plt
import matplotlib.colors as plt_col
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import networkx as netx
import numpy as np
import os
import pandas as pd
import scipy
import re
import imageio
from skimage import measure

wd = os.getcwd()

##################
## INPUT/OUTPUT ##
##################
def load_json2dict(filename):
    file = open(filename, 'rb')
    data = json.load(file)
    return data


###############
## UTILITIES ##
###############
def partitioning(IDs, w, n):
    '''
    Return a 1D array of length n of IDs based on their weights w
    '''
    indsort = np.argsort(w)
    cumPr = np.cumsum(w[indsort])
    midpt = np.arange(1/n, 1+1/n, 1/n) - 1/n/2
    vec_IDs = np.zeros(n).astype(int)
    for i in range(n):
        vec_IDs[i] = IDs[indsort][np.argwhere(cumPr > midpt[i])[0]]
    return np.sort(vec_IDs)

def fetch_A0(region):
    R_earth = 6371.                    # [km]
    A_earth = 4 * np.pi * R_earth**2   # [km^2]
    A_CONUS = 8080464.                 # [km^2]
    regions = ['global', 'CONUS']
    areas = [A_earth, A_CONUS]
    return areas[region == regions]

def zero_boundary_2d(arr, nx, ny):
    arr[:nx,:] = 0
    arr[-nx:,:] = 0
    arr[:,:ny] = 0
    arr[:,-ny:] = 0
    return arr


#############################
## ENVIRONMENTAL FUNCTIONS ##
#############################
class RasterGrid:
    """Define the coordinates (x,y) of the square-pixels of a 2D raster grid.
    
    Notes:
        If x0, xbuffer, ybuffer and/or lat_deg are not provided by the user, 
        they are fixed to xmin, 0, 0 and 45, respectively.

    Attributes:
        par (dict): Input, dictionary with keys ['w', 'xmin', 'x0' (opt.),
                    'xmax', 'ymin', 'ymax', 'xbuffer' (opt.), 'ybuffer' (opt.)]
        w (float): Pixel width in km
        xmin (float): Minimum abcissa of buffer box
        xmax (float): Maximum abcissa of buffer box
        ymin (float): Minimum ordinate of buffer box
        ymax (float): Maximum ordinate of buffer box
        xbuffer (float): Buffer width in the x direction (default is 0.)
        ybuffer (float): Buffer width in the y direction (default is 0.)
        lat_deg (float): Latitude at center of the grid (default is 45.)
        x0 (float): Abscissa of reference N-S coastline (default is xmin)
        x (ndarray(dtype=float, ndim=1)): 1D array of unique abscissas
        y (ndarray(dtype=float, ndim=1)): 1D array of unique ordinates
        xx (ndarray(dtype=float, ndim=2)): 2D array of grid abscissas
        yy (ndarray(dtype=float, ndim=2)): 2D array of grid ordinates
        nx (int): Length of x
        ny (int): Length of y

    Returns: 
        class instance: A new instance of class RasterGrid
    
    Example:
        Create a grid

            >>> grid = RasterGrid({'w': 1, 'xmin': 0, 'xmax': 2, 'ymin': 0, 'ymax': 3})
            >>> grid.x
            array([0., 1., 2.])
            >>> grid.y
            array([0., 1., 2., 3.])
            >>> grid.xx
            array([[0., 0., 0., 0.],
               [1., 1., 1., 1.],
               [2., 2., 2., 2.]])
            >>> grid.yy
            array([[0., 1., 2., 3.],
               [0., 1., 2., 3.],
               [0., 1., 2., 3.]])
    """
    
    def __init__(self, par):
        """
        Initialize RasterGrid
        
        Args:
            par (dict): Dictionary of input parameters with the following keys:
                w (float): Pixel width in km
                xmin (float): Minimum abcissa of buffer box
                xmax (float): Maximum abcissa of buffer box
                ymin (float): Minimum ordinate of buffer box
                ymax (float): Maximum ordinate of buffer box
                xbuffer (float, optional): Buffer width in the x direction (default is 0)
                ybuffer (float, optional): Buffer width in the y direction (default is 0)
                x0 (float, optional): Abscissa of reference N-S coastline (default is xmin)
        """
        self.par = par
        self.w = par['w']
        self.xmin = par['xmin']
        self.xmax = par['xmax']
        self.ymin = par['ymin']
        self.ymax = par['ymax']
        if 'xbuffer' in par.keys():
            self.xbuffer = par['xbuffer']
        else:
            self.xbuffer = 0.
        if 'ybuffer' in par.keys():
            self.ybuffer = par['ybuffer']
        else:
            self.ybuffer = 0.
        if 'x0' in par.keys():
            self.x0 = par['x0']
        else:
            self.x0 = self.xmin
        if 'lat_deg' in par.keys():
            self.lat_deg = par['lat_deg']
        else:
            self.lat_deg = 45.
        self.x = np.arange(self.xmin - self.w/2, self.xmax + self.w/2, self.w) + self.w/2
        self.y = np.arange(self.ymin - self.w/2, self.ymax + self.w/2, self.w) + self.w/2
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')
        self.nx = len(self.x)
        self.ny = len(self.y)

    def __repr__(self):
        return 'RasterGrid({})'.format(self.par)
    

def calc_coord_river_dampedsine(grid, par, z = ''):
    '''
    Calculate the (x,y,z) coordinates of the river(s) defined from a damped sine wave.
    '''
    nriv = len(par['riv_y0'])
    river_xi = np.array([])
    river_id = np.array([])
    river_yi = np.array([])
    river_zi = np.array([])
    for riv in range(nriv):
        expdecay = par['riv_A_km'][riv] * np.exp(-par['riv_lbd'][riv] * grid.x)
        yrv_0 = expdecay * np.cos(par['riv_ome'][riv] * grid.x) + par['riv_y0'][riv]
        indy = np.where(grid.y > par['riv_y0'][riv] - 1e-6)[0][0]
        if len(z) != 0:
            zrv_0 = z[:,indy]
            indland = np.where(zrv_0 >= 0)
        else:
            zrv_0 = np.zeros(grid.nx)
            indland = np.arange(grid.nx)
        river_xi = np.append(river_xi, grid.x[indland])
        river_yi = np.append(river_yi, yrv_0[indland])
        river_zi = np.append(river_zi, zrv_0[indland])
        river_id = np.append(river_id, np.repeat(riv, grid.nx)[indland])
    return river_xi, river_yi, river_zi, river_id


def gen_rdmcoord_tracks(N, grid, npt, max_deviation):
    '''
    Return coordinates of N storm tracks, defined as straight lines
    subject to random deviation (below max_deviation) along y at npt points.
    '''
    ID = np.repeat(np.arange(N)+1, npt)
    x = np.tile(np.linspace(grid.xmin, grid.xmax, npt), N)
    ystart = grid.ymin + np.random.random(N) * (grid.ymax - grid.ymin)
    yend = grid.ymin + np.random.random(N) * (grid.ymax - grid.ymin)
    y = np.linspace(ystart, yend, npt, axis = 1).flatten()
    deviation = np.random.uniform(-max_deviation, max_deviation, size = N*npt)
    y += deviation
    return x, y, ID


######################
## HAZARD FUNCTIONS ##
######################

## SOURCE DEF. & EVENT RATE ##
def incrementing(xmin, xmax, xbin, scale):
    '''
    Return evenly spaced values within a given interval in linear or log scale
    '''
    if scale == 'linear':
        xi = np.arange(xmin, xmax + xbin, xbin)
    if scale == 'log':
        xi = 10**np.arange(np.log10(xmin), np.log10(xmax) + xbin, xbin)
    return xi

def get_peril_evID(evID):
    '''
    Return the peril identifiers for an array of event identifiers
    '''
    return np.array([evID[k][0:2] for k in range(len(evID))])

def calc_Lbd_powerlaw(S, a, b):
    '''
    Calculate the cumulative rate Lbd according to a power-law (Eq. 2.38) given event size S
    '''
    Lbd = 10**(a - b * np.log10(S))
    return Lbd

def calc_Lbd_exponential(S, a, b):
    '''
    Calculate the cumulative rate Lbd according to an exponential law (Eq. 2.39) given event size S
    '''
    Lbd = 10**(a - b * S)
    return Lbd

def calc_Lbd_GPD(S, mu, xi, sigma, Lbdmin):
    '''
    Calculate the cumulative rate Lbd according to the Generalised Pareto Distribution (Eq. 2.50) given event size S
    '''
    Lbd = Lbdmin * (1 + xi * (S - mu) / sigma)**(-1 / xi)
    return Lbd


def transform_cum2noncum(S, par):
    '''
    Transform the rate from cumulative (Lbd) to non-cumulative (lbd) (e.g., Eq. 2.65)
    '''
    if par['Sscale'] == 'linear':
        S_lo = S - par['Sbin']/2
        S_hi = S + par['Sbin']/2
    elif par['Sscale'] == 'log':
        S_lo = 10**(np.log10(S) - par['Sbin']/2)
        S_hi = 10**(np.log10(S) + par['Sbin']/2)
    if par['distr'] == 'powerlaw':
        Lbd_lo = calc_Lbd_powerlaw(S_lo, par['a'], par['b'])
        Lbd_hi = calc_Lbd_powerlaw(S_hi, par['a'], par['b'])
    if par['distr'] == 'exponential':
        Lbd_lo = calc_Lbd_exponential(S_lo, par['a'], par['b'])
        Lbd_hi = calc_Lbd_exponential(S_hi, par['a'], par['b'])
    if par['distr'] == 'GPD':
        Lbd_lo = calc_Lbd_GPD(S_lo, par['mu'], par['xi'], par['sigma'], par['Lbdmin'])
        Lbd_hi = calc_Lbd_GPD(S_hi, par['mu'], par['xi'], par['sigma'], par['Lbdmin'])
    lbd = Lbd_lo - Lbd_hi
    return lbd


## EQ CASE ##
def get_char_srcLine(par):
    '''
    Calculate coordinates of fault sources based on their extrema and the provided 
    resolution, as well as lenghts and strikes of faults and fault segments.        
    '''
    src_xi = np.array([])
    src_yi = np.array([])
    src_id = np.array([])
    src_L = np.array([])
    seg_id = np.array([])
    seg_strike = np.array([])
    seg_L = np.array([])
    seg = 0
    for src_i in range(len(par['x'])):
        Lsum = 0
        for seg_i in range(len(par['x'][src_i]) - 1):
            dx = par['x'][src_i][seg_i+1] - par['x'][src_i][seg_i]
            dy = par['y'][src_i][seg_i+1] - par['y'][src_i][seg_i]
            sign1 = dx / np.abs(dx)
            sign2 = dy / np.abs(dy)
            L = np.sqrt(dx**2 + dy**2)
            strike = np.arctan(dx/dy) * 180 / np.pi
            sign3 = np.sin(strike * np.pi / 180) / np.abs(np.sin(strike * np.pi / 180))
            npt = int(np.round(L / par['bin_km']))
            seg_xi = np.zeros(npt)
            seg_yi = np.zeros(npt)
            seg_xi[0] = par['x'][src_i][seg_i]
            seg_yi[0] = par['y'][src_i][seg_i]
            for k in range(1, npt):
                seg_xi[k] = seg_xi[k-1] + sign1 * sign3 * par['bin_km'] * np.sin(strike * np.pi / 180)
                seg_yi[k] = seg_yi[k-1] + sign2 * par['bin_km'] * np.cos(strike * np.pi / 180)
            src_xi = np.append(src_xi, np.append(seg_xi, par['x'][src_i][seg_i+1]))
            src_yi = np.append(src_yi, np.append(seg_yi, par['y'][src_i][seg_i+1]))
            src_id = np.append(src_id, np.repeat(src_i, len(seg_xi)+1))
            seg_id = np.append(seg_id, np.repeat(seg, len(seg_xi)+1))
            seg_strike = np.append(seg_strike, strike)
            seg_L = np.append(seg_L, L)
            seg += 1
            Lsum += L
        src_L = np.append(src_L, Lsum)
    return {'x': src_xi, 'y': src_yi, 'srcID': src_id, 'srcL': src_L, 'segID': seg_id, 'strike': seg_strike, 'segL': seg_L}

def calc_EQ_length2magnitude(L):
    '''
    Given the earthquake rupture L [km], calculate the magnitude M
    '''
    c1, c2 = [5., 1.22]     # reverse case, Fig. 2.6b, Wells and Coppersmith (1994)
    M = c1 + c2 * np.log10(L)
    return np.round(M, decimals = 1)

def calc_EQ_magnitude2length(M):
    '''
    Given the earthquake magnitude M, calculate the rupture length L [km]
    (for floating rupture computations)
    '''
    c1, c2 = [5., 1.22]     # reverse case, Fig. 2.6b, Wells and Coppersmith (1994)
    L = 10**((M - c1)/c2)
    return L

def pop_EQ_floatingRupture(evIDi, Si, src, srcEQ_char):
    '''
    '''
    nRup = len(Si)
    li = calc_EQ_magnitude2length(Si)
    flt_x = srcEQ_char['x']
    flt_y = srcEQ_char['y']
    flt_L = srcEQ_char['srcL']
    flt_id = srcEQ_char['srcID']
    indflt = partitioning(np.arange(len(flt_L)), flt_L / np.sum(flt_L), nRup)  # longer faults visited more often
    Rup_loc = np.zeros(nRup, dtype = object)
    Rup_coord = pd.DataFrame(columns = ['evID', 'x', 'y', 'loc'])
    i = 0
    while i < nRup:
        flt_target = np.random.choice(indflt, 1)
        indID = flt_id == flt_target
        src_x = flt_x[indID]
        src_y = flt_y[indID]
        src_L = flt_L[flt_target]
        init = np.floor((src_L - li[i]) / src['EQ']['bin_km'])
        if src_L >= li[i]:
            u = np.ceil(np.random.random(1) * init).astype(int)[0]         # random rupture start loc
            Rup_x = src_x[u:(u + li[i] / src['EQ']['bin_km']).astype(int)]
            Rup_y = src_y[u:(u + li[i] / src['EQ']['bin_km']).astype(int)]
            Rup_loc[i] = src['EQ']['object'] + str(flt_target[0] + 1)
            Rup_coord = pd.concat([Rup_coord, pd.DataFrame(data = {'evID': np.repeat(evIDi[i], len(Rup_x)), \
                                                              'x': Rup_x, 'y': Rup_y, \
                                                              'loc': np.repeat(Rup_loc[i], len(Rup_x))})], ignore_index=True)
            i += 1
            
    return Rup_coord


## TC CASE ##
def get_TCtrack_highres(x0, y0, id0, bin_km):
    x_hires = np.array([])
    y_hires = np.array([])
    id_hires = np.array([])
    evID = np.unique(id0)
    for i in range(len(evID)):
        indev = np.where(id0 == evID[i])[0]
        x_ev = x0[indev]
        y_ev = y0[indev]
        for seg in range(len(x_ev) - 1):
            dx = x_ev[seg + 1] - x_ev[seg]
            dy = y_ev[seg + 1] - y_ev[seg]
            sign1 = dx / np.abs(dx)
            sign2 = dy / np.abs(dy)
            L = np.sqrt(dx**2 + dy**2)
            strike = np.arctan(dx/dy) * 180 / np.pi
            npt = int(np.round(L / bin_km))
            seg_xi = np.zeros(npt)
            seg_yi = np.zeros(npt)
            seg_xi[0] = x_ev[seg]
            seg_yi[0] = y_ev[seg]
            for k in range(1, npt):
                seg_xi[k] = seg_xi[k-1] + sign1 * sign2 * bin_km * np.sin(strike * np.pi / 180)
                seg_yi[k] = seg_yi[k-1] + sign1 * sign2 * bin_km * np.cos(strike * np.pi / 180)
            x_hires = np.append(x_hires, np.append(seg_xi, x_ev[seg + 1]))
            y_hires = np.append(y_hires, np.append(seg_yi, y_ev[seg + 1]))
            id_hires = np.append(id_hires, np.repeat(evID[i], len(seg_xi)+1))
    
    Track_coord = pd.DataFrame({'x': x_hires, 'y': y_hires, 'ID': id_hires})
    return Track_coord

def calc_S_track(stochset, src, Track_coord):    
    indperil = np.where(stochset['ID'] == 'TC')[0]
    evIDs = stochset['evID'][indperil].values
    vmax_start = stochset['S'][indperil].values
    S_alongtrack = {}
    for i in range(src['TC']['N']):
        indtrack = np.where(Track_coord['ID'] == i+1)[0]
        track_x = Track_coord['x'][indtrack].values
        track_y = Track_coord['y'][indtrack].values

        npt = len(indtrack)
        track_vmax = np.repeat(vmax_start[i], npt)  # track over ocean at vmax_start

        # find inland section & reduce vmax
        d = [np.min(np.sqrt((track_x[j] - src['SS']['x'])**2 + \
                            (track_y[j] - src['SS']['y'])**2)) for j in range(npt)]
        indcoast = np.where(d == np.min(d))[0]
        d2coast = track_x[indcoast[0]:] - track_x[indcoast[0]]
        # ad-hoc decay relationship:
        track_vmax[indcoast[0]:] = vmax_start[i] * np.exp(-.1 / src['TC']['vforward_m/s'] * d2coast)
        
        S_alongtrack[evIDs[i]] = track_vmax
    return S_alongtrack


## STOCHASTIC EVENT SET ##
def gen_eventset(src, sizeDistr):
    ev_stoch = pd.DataFrame(columns = ['ID', 'evID', 'S', 'lbd'])
    srcEQ_char = get_char_srcLine(src['EQ'])
    Mmax2 = calc_EQ_length2magnitude(srcEQ_char['srcL'][1])  # smaller of 2 hardcoded faults in src

    for ID in src['perils']:
        if ID in sizeDistr['primary']:
            # event ID definition #
            evID = [ID + str(i+1) for i in range(src[ID]['N'])]

            # size incrementation #
            Si = incrementing(sizeDistr[ID]['Smin'], sizeDistr[ID]['Smax'], sizeDistr[ID]['Sbin'], sizeDistr[ID]['Sscale'])

            # weighting how Si size distributed over N event sources
            Si_n = len(Si)
            Si_ind = np.arange(Si_n)
            if ID == 'EQ':
                # smaller events more often to test more spatial combinations on fault segments
                qi = np.linspace(1,11,Si_n)
                qi /= np.sum(qi)
                qi = np.sort(qi)[::-1]                
            else:
                # equal weight
                qi = np.repeat(1./Si_n, Si_n)
            Si_ind_vec = partitioning(Si_ind, qi, src[ID]['N'])  # distribute Si sizes into N event sources
            Si_vec = Si[Si_ind_vec]
            wi = 1 / np.array([np.count_nonzero(Si_ind_vec == i) for i in Si_ind])
            wi_vec = [wi[Si_ind == i][0] for i in Si_ind_vec]    # weight of rate(Si) at each of N locations

            # rate calculation #
            # calibrate event productivity
            if sizeDistr[ID]['distr'] == 'powerlaw':
                if 'a' not in sizeDistr[ID].keys():
                    rescaled = src['grid_A_km2'] / fetch_A0(sizeDistr[ID]['region'])
                    sizeDistr[ID]['a'] = sizeDistr[ID]['a0'] + np.log10(rescaled)
            if sizeDistr[ID]['distr'] == 'GPD':
                if 'Lbdmin' not in sizeDistr[ID].keys():
                    rescaled = src['grid_A_km2'] / fetch_A0(sizeDistr[ID]['region'])
                    sizeDistr[ID]['Lbdmin'] = sizeDistr[ID]['Lbdmin0'] * rescaled
            # calculate event rate (weighted)
            lbdi = transform_cum2noncum(Si_vec, sizeDistr[ID])
            lbdi = lbdi * wi_vec
            ev_stoch = pd.concat([ev_stoch, pd.DataFrame(data = {'ID': np.repeat(ID, src[ID]['N']), 'evID': evID, 'S': Si_vec, 'lbd': lbdi})], ignore_index=True)

        if ID in sizeDistr['secondary']:
            trigger = sizeDistr[ID]['trigger']
            evID = [ID + '_from' + trigger + str(i+1) for i in range(src[trigger]['N'])]
            Si_vec = np.repeat(np.nan, src[trigger]['N'])
            lbdi = np.repeat(np.nan, src[trigger]['N'])
            ev_stoch = pd.concat([ev_stoch, pd.DataFrame(data = {'ID': np.repeat(ID, src[trigger]['N']), 'evID': evID, 'S': Si_vec, 'lbd': lbdi})], ignore_index=True)

    return ev_stoch.reset_index(drop = True)


## HAZARD FOOTPRINT MODELS ##
# analytical
def calc_I_shaking_ms2(S, r):
    PGA_g = 10**(-1.34 + .23*S - np.log10(r))     # size = magnitude
    g_earth = 9.81                   # [m/s^2]
    PGA_ms2 = PGA_g * g_earth
    return PGA_ms2

def calc_I_blast_kPa(S, r):
    Z = r * 1e3 / (S * 1e6)**(1/3)                # size = energy in kton TNT
    p_kPa = (1772/Z**3 - 114/Z**2 + 108/Z)
    return p_kPa

def calc_I_ash_m(S, r):
    # assumes h0 proportional to V - e.g h0 = 1e-3 km for V=3 km3 (1980 Mt. St. Helens)
    h0 = 1e-3 /3 * S                                                # size = volume in km3 
    r_half = np.sqrt(S * np.log(2)**2 / (2* np.pi * h0) )
    h_m = ( h0 * np.exp (-np.log(2) * r / r_half) ) * 1e3   # m
    return h_m

def calc_I_v_ms(S, r, par):
    '''
    Eq. 2.24
    '''
    rho_atm = 1.15                                       # air density [kg/m3]
    Omega = 7.2921e-5                                    # [rad/s]
    f = 2 * Omega * np.sin(par['lat_deg'] * np.pi/180)   # Coriolis parameter
    
    pn = par['pn_mbar'] * 100                            # [Pa]
    B = par['B_Holland']
    
    R = 51.6 * np.exp(-.0223 * S + .0281 * par['lat_deg'])   # see caption of Fig. 2.19
    pc = pn - 1 / B * (rho_atm * np.exp(1) * S**2)
    
    v_ms = ( B * R**B * (pn - pc) * np.exp(-(R/r)**B) / (rho_atm * r**B) + r**2 * f**2 / 4 )**.5 - r*f/2
    return v_ms

def add_v_forward(vf, vtan, track_x, track_y, grid, t_i):
    # components of forward motion vector
    if t_i < len(track_x)-1:
        dx = track_x[t_i+1]-track_x[t_i]
        dy = track_y[t_i+1]-track_y[t_i]
        if dx == 0 and dy == 0:
            dx = track_x[t_i]-track_x[t_i-1]
            dy = track_y[t_i]-track_y[t_i-1]
    else:
        # assumes same future direction
        dx = track_x[t_i]-track_x[t_i-1]
        dy = track_y[t_i]-track_y[t_i-1]


    beta = np.arctan(dy/dx)
    if dx > 0:
        vf_x = vf * np.cos(beta)
        vf_y = vf * np.sin(beta)
    else:
        vf_x = -vf * np.cos(beta)
        vf_y = -vf * np.sin(beta)
        
    # components of gradient-based azimuthal wind vector
    dx = grid.xx - track_x[t_i]
    dy = grid.yy - track_y[t_i]
    alpha = np.arctan(dy/dx)
    # if x > x0
    vtan_x = -vtan * np.sin(alpha)
    vtan_y = vtan * np.cos(alpha)
    # if x < x0
    indneg = np.where(grid.xx < track_x[t_i])
    vtan_x[indneg] = vtan[indneg] * np.sin(alpha[indneg])
    vtan_y[indneg] = -vtan[indneg] * np.cos(alpha[indneg])

    vtot_x = vtan_x + vf_x
    vtot_y = vtan_y + vf_y
    vtot = np.sqrt(vtot_x**2 + vtot_y**2)
    return vtot, vtot_x, vtot_y, vtan_x, vtan_y


# threshold model
def calc_S_TC2SS(v_max, relationship = 'generic'):
    '''
    Empirical relationships according to the Saffir-Simpson scale (generic) or 
    from Lin et al. (2010) (New York harbor).
    vmax: max wind speed [m/s] during storm passage
    S_SS: storm surge size at the source (coastline) 
    '''
    if relationship == 'generic':
        S_SS = .0011 * v_max**2 
    if relationship == 'New York harbor':
        S_SS = .031641 * v_max - .00075537 * v_max**2 + 3.1941e-5 * v_max**3
    return np.round(S_SS, decimals = 3)

def model_SS_Bathtub(I_trigger, src_SS, grid, topo_z):
        vmax_coastline = np.zeros(grid.ny)
        for j in range(grid.ny):
            indx = np.where(grid.x > src_SS['x'][j]-1e-6)[0][0]
            vmax_coastline[j] = I_trigger[indx,j]
        S_SS = calc_S_TC2SS(vmax_coastline, src_SS['bathy'])
        I_SS = np.zeros((grid.nx, grid.ny))
        for j in range(grid.ny):
            I_alongx = S_SS[j] - topo_z[:,j]
            I_alongx[I_alongx < 0] = 0
            I_alongx[grid.x < src_SS['x'][j]] = 0
            I_SS[:,j] = I_alongx
        return I_SS


def gen_hazFootprints(stochset, src, grid, topo_z):
    catalog_hazFootprints = {}
    print('generating footprints for:')
    for ID in src['perils']:
        indperil = np.where(stochset['ID'] == ID)[0]
        Nev_peril = len(indperil)

        if ID == 'AI':
            print(ID)
            for i in range(Nev_peril):
                evID = stochset['evID'][indperil].values[i]
                S = stochset['S'][indperil].values[i]
                r = np.sqrt((grid.xx - src['AI']['x'][i])**2 + (grid.yy - src['AI']['y'][i])**2)   # point source
                catalog_hazFootprints[evID] = calc_I_ash_m(S, r)

        if ID == 'VE':
            print(ID)
            for i in range(Nev_peril):
                evID = stochset['evID'][indperil].values[i]
                S = stochset['S'][indperil].values[i]
                r = np.sqrt((grid.xx - src['VE']['x'][0])**2 + (grid.yy - src['VE']['y'][0])**2)   # point source
                catalog_hazFootprints[evID] = calc_I_blast_kPa(S, r)

        if ID == 'EQ':
            print(ID)
            EQ_coords = src['EQ']['rup_coords']
            for i in range(Nev_peril):
                evID = stochset['evID'][indperil].values[i]
                S = stochset['S'][indperil].values[i]
                evID_coords = EQ_coords[EQ_coords['evID'] == evID]
                npt = len(evID_coords)
                d2rupt = np.zeros((grid.nx, grid.ny, npt))
                for k in range(npt):
                    d2rupt[:,:,k] = np.sqrt((grid.xx - evID_coords['x'].values[k])**2 + (grid.yy - evID_coords['y'].values[k])**2)
                dmin = d2rupt.min(axis = 2)
                z = src['EQ']['z_km'][int(evID_coords['loc'].iloc[0][-1])-1]
                r = np.sqrt(dmin**2 + z**2)                                                        # line source
                catalog_hazFootprints[evID] = calc_I_shaking_ms2(S, r)

        if ID == 'TC':
            print(ID)
            Track_coord = get_TCtrack_highres(src['TC']['x'], src['TC']['y'], src['TC']['ID'], src['TC']['bin_km'])
            S_alongtrack = calc_S_track(stochset, src, Track_coord) # ad-hoc inland decay of windspeed
            for i in range(Nev_peril):
                evID = stochset['evID'][indperil].values[i]
                indtrack = np.where(Track_coord['ID'] == i+1)[0]
                track_x = Track_coord['x'][indtrack].values
                track_y = Track_coord['y'][indtrack].values
                track_S = S_alongtrack[evID]
                npt = len(indtrack)
                I_t = np.zeros((grid.nx, grid.ny, npt))
                for j in range(npt):
                    r = np.sqrt((grid.xx - track_x[j])**2 + (grid.yy - track_y[j])**2)             # point source at time t
                    I_sym_t = calc_I_v_ms(track_S[j], r, src['TC'])
                    I_t[:,:,j], _, _, _, _ = \
                        add_v_forward(src['TC']['vforward_m/s'], I_sym_t, track_x, track_y, grid, j)
                catalog_hazFootprints[evID] = np.nanmax(I_t, axis = 2)                                # track source

        if ID == 'SS':
            print(ID)
            pattern = re.compile(r'TC(\d+)')  # match "TC" followed by numbers
            for i in range(Nev_peril):
                evID = stochset['evID'][indperil].values[i]
                evID_trigger = re.search(pattern, evID).group()
                I_trigger = catalog_hazFootprints[evID_trigger]
                catalog_hazFootprints[evID] = model_SS_Bathtub(I_trigger, src['SS'], grid, topo_z)

    print('... catalogue completed')
    return catalog_hazFootprints


###########################
## dynamic hazard models ##
###########################

## LANDSLIDE CASE ##
def calc_topo_attributes(z, w):
    z = np.pad(z*1e-3, 1, 'edge')   # from m to km
    # 3x3 kernel method to get dz/dx, dz/dy
    dz_dy, dz_dx = np.gradient(z)
    dz_dx = dz_dx[1:-1,1:-1] / w
    dz_dy = (dz_dy[1:-1,1:-1] / w) * (-1)
    tan_slope = np.sqrt(dz_dx**2 + dz_dy**2)
    slope = np.arctan(tan_slope) * 180 / np.pi
    aspect = 180 - np.arctan(dz_dy/dz_dx)*180/np.pi + 90 * (dz_dx + 1e-6) / (np.abs(dz_dx) + 1e-6)
    return tan_slope, aspect

def calc_FS(slope, h, w, par):
    '''
    Calculates the factor of safety using Eq. 3 of Pack et al. (1998).
        
    Reference:
        Pack RT, Tarboton DG, Goodwin CN (1998), The SINMAP Approach to Terrain Stability Mapping. 
        Proceedings of the 8th Congress of the International Association of Engineering Geology, Vancouver, BC, 
        Canada, 21 September 1998
    '''
    g_earth = 9.81    # [m/s^2]
    rho_wat = 1000.   # [kg/m^3]
    FS = (par['Ceff_Pa'] / (par['rho_kg/m3'] * g_earth * h) + np.cos(slope * np.pi/180) * \
         (1 - w * rho_wat / par['rho_kg/m3']) * np.tan(par['phieff_deg'] * np.pi/180)) / \
         np.sin(slope * np.pi/180)
    return FS

def get_neighborhood_ind(i, j, grid_shape, r_v, method):
    '''
    Get the indices of the neighboring cells, depending on method and radius of vision
    '''
    nx, ny = grid_shape
    # rv_box neighborhood
    indx = range(i - r_v, i + r_v + 1)
    indy = range(j - r_v, j + r_v + 1)
    # cut at grid borders
    indx_k = np.array([np.nan if (k < 0 or k > (nx - 1)) else k for k in indx])
    indy_k = np.array([np.nan if (k < 0 or k > (ny - 1)) else k for k in indy])
    indx_cut = ~np.isnan(indx_k)
    indy_cut = ~np.isnan(indy_k)
    ik, jk = [indx_k[indx_cut].astype('int'), indy_k[indy_cut].astype('int')]
    # mask
    mask = np.ones((2*r_v + 1, 2*r_v + 1), dtype = bool)
    nx_mask, ny_mask = mask.shape
    i0 = int(np.floor(nx_mask/2))
    j0 = int(np.floor(ny_mask/2))
    mask[i0,j0] = 0
    if method == 'Moore':
        mask_cut = mask[np.ix_(indx_cut, indy_cut)]
    if method == 'vonNeumann':
        mask = np.zeros((nx_mask, ny_mask), dtype = bool)
        mask[i0,:] = 1
        mask[:,j0] = 1
        mask[i0,j0] = 0
        mask_cut = mask[np.ix_(indx_cut, indy_cut)]
    return [np.meshgrid(ik,jk)[i].flatten()[mask_cut.flatten()] for i in range(2)]

def get_ind_aspect2moore(ind_old):
    '''
    Return Moore indices from indices defined from the aspect angle.
    
    Note:
        The aspect angle directs towards index np.round(aspect*8/360).astype('int').
        It therefore takes the form:  765
                                      0 4
                                      123
        while Moore indices take the form: 012 (see get_neighborhood_ind() function).
                                           3 4
                                           567
    '''
    ind_new = np.array([3,5,6,7,4,2,1,0,3])
    return ind_new[ind_old]

def calc_stableSlope(h, w, par):
    slope_i = np.arange(1, 50, .1)
    FS_i = calc_FS(slope_i, h, w, par)
    slope_stable = slope_i[FS_i > 1.5][-1]
    return slope_stable

def run_CellularAutomaton_LS(LSfootprint, hs, topo_z, grid, par, w, movie):
    h0 = np.copy(hs)  # original soil depth
    h = np.copy(hs)   # updating soil depth
    z = np.copy(topo_z)  # altitude changes over time

    if movie['create'] and not os.path.exists(movie['path']):
        os.makedirs(movie['path'])

    LSfootprint_hmax = np.zeros((grid.nx, grid.ny))
    kmax = 20
    k = 1
    while k <= kmax:
        print('iteration', k, '/', kmax)
        indmov = np.where(np.logical_and(LSfootprint == 1, h > 0))
        for kk in range(len(indmov[0])):
            i, j = [indmov[0][kk], indmov[1][kk]]
            z_pad = np.pad(z, 1, 'edge')
            tan_slope, aspect = calc_topo_attributes(z_pad[i:i+3, j:j+3], grid.w)
            slope = np.arctan(tan_slope[1,1]) * 180 / np.pi
            steepestdir = np.round(aspect[1,1] * 7 / 360).astype('int') 
            slope_stable = calc_stableSlope(h[i,j], w[i,j], par)
            if slope > slope_stable:
                i_nbor, j_nbor = get_neighborhood_ind(i, j, (grid.nx, grid.ny), 1, method = 'Moore')
                steepestdir_rfmt = get_ind_aspect2moore(steepestdir)
                i1, j1 = [i_nbor[steepestdir_rfmt], j_nbor[steepestdir_rfmt]]
                if steepestdir % 2 == 0: # perpendicular
                    dh_stable = grid.w*1e3 * np.tan(slope_stable * np.pi/180)
                    dz = (grid.w*1e3 * np.tan(slope*np.pi/180) - dh_stable)/2
                else:                    # diagonal
                    dh_stable = grid.w*1e3 * np.sqrt(2) * np.tan(slope_stable * np.pi/180)
                    dz = (grid.w*1e3 * np.sqrt(2) * np.tan(slope*np.pi/180) - dh_stable)/2
                if dz > h[i,j]:
                    dz = h[i,j]
                z[i,j] = z[i,j] - dz
                z[i1,j1] = z[i1,j1] + dz
                h[i,j] = h[i,j] - dz
                h[i1,j1] = h[i1,j1] + dz
                LSfootprint[i1,j1] = 1

        LSfootprint_hmax = np.maximum(LSfootprint_hmax, h - h0)        
        
        # plot
        if movie['create']:
            plt.rcParams['font.size'] = '20'
            fig, ax = plt.subplots(1, 1, figsize=(10,10), facecolor = 'white')
            h_plot = h_code(h, h0)
            ax.contourf(grid.xx, grid.yy, h_plot, cmap = h_col, vmin = 1, vmax = 6)
            ax.contourf(grid.xx, grid.yy, ls.hillshade(topo_z, vert_exag=.1), cmap='gray', alpha = .1)
            ax.set_xlabel('x [km]')
            ax.set_ylabel('y [km]')
            ax.set_title('LS iteration'+str(k), pad = 10)
            ax.set_aspect(1)
            ax.set_xlim(movie['xmin'], movie['xmax'])
            ax.set_ylim(movie['ymin'], movie['ymax'])
            if k < 10:
                k_str = '0' + str(k)
            else:
                k_str = str(k)
            plt.savefig(movie['path']+'iter' + k_str + '.png', dpi = 300, bbox_inches='tight')
            plt.close()

        k += 1
        
    if movie['create']:
        fd = movie['path']
        img = []
        filenames = [filename for filename in os.listdir(fd) if filename.startswith('iter')]
        filenames.sort()
        for filename in filenames:
            img.append(imageio.imread(fd + filename))
        imageio.mimsave(wd + '/figures/movie_LS.gif', img, duration = 500, loop = 0)
            
    return LSfootprint_hmax


def gen_hazFootprints_LS(stochset, src, grid, topo_z, soil_hs):
    catalog_hazFootprints_LS = {}
    ID = 'LS'
    pattern = re.compile(r'RS(\d+)')
    indperil = np.where(stochset['ID'] == ID)[0]
    Nev_peril = len(indperil)
    tan_slope, _ = calc_topo_attributes(topo_z, grid.w)
    slope = np.arctan(tan_slope) * 180 / np.pi
    nx, ny = int(grid.xbuffer/grid.w), int(grid.ybuffer/grid.w)

    for i in range(Nev_peril):
        evID = stochset['evID'][indperil].values[i]
        print(evID)
        evID_trigger = re.search(pattern, evID).group()
        FS_state = np.zeros((grid.nx,grid.ny))
        LSfootprint_seed = np.zeros((grid.nx, grid.ny))
        S_trigger = stochset['S'][stochset['evID'] == evID_trigger].values
        hw = S_trigger * 1e-3 * src['RS']['duration']        # water column [m]
        wetness = hw / soil_hs
        wetness[wetness > 1] = 1              # max possible saturation
        wetness[soil_hs == 0] = 0             # no soil case
        FS = calc_FS(slope, soil_hs, wetness, src['LS'])
        FS_state = get_FS_state(FS)
        LSfootprint_seed[FS_state == 2] = 1      # initiates landslide where slope is unstable
        LSfootprint_seed = zero_boundary_2d(LSfootprint_seed, nx, ny)   # no landslide in buffer zone

        movie = {'create': False}
        catalog_hazFootprints_LS[evID] = run_CellularAutomaton_LS(LSfootprint_seed, soil_hs, topo_z, grid, src['LS'], wetness, movie)
    print('... catalogue completed')
    return catalog_hazFootprints_LS


## WILDFIRE CASE ##
def run_CellularAutomaton_WF(landuseLayer_state, src, grid, topoLayer_z):
    landuseLayer_state4WF = np.copy(landuseLayer_state)
    indForest = np.where(landuseLayer_state.flatten() == 1)[0]
    indForest2Grass = np.random.choice(indForest, size = int(len(indForest) * src['WF']['ratio_grass']), replace=False)
    landuseLayer_state4WF = landuseLayer_state4WF.flatten()
    landuseLayer_state4WF[indForest2Grass] = 0
    landuseLayer_state4WF = landuseLayer_state4WF.reshape(landuseLayer_state.shape)

    if not os.path.exists(src['WF']['path']):
        os.makedirs(src['WF']['path'])

    S = np.zeros(landuseLayer_state4WF.shape)  # 0: no tree
    S[landuseLayer_state4WF == 1] = 1          # 1: tree
    for i in range(src['WF']['nsim']):
        if i%1000 == 0:
            print(i, '/', src['WF']['nsim'])

        # LOADING (long-term tree growth)
        indForest_notree = indForest[np.where(S.flatten()[indForest] == 0)[0]]
        new_tree_xy = np.random.choice(indForest_notree, size = src['WF']['rate_newtrees'])
        S_flat = S.flatten()
        S_flat[new_tree_xy] = 1
        landuseLayer_state4WF_flat = landuseLayer_state4WF.flatten()
        landuseLayer_state4WF_flat[new_tree_xy] = 1

        if np.random.random(1) <= src['WF']['p_lightning']:
            # TRIGGER (lightning)
            lightning_xy = np.random.choice(indForest, size=1)
            WF_fp = np.zeros(S.shape)
            WF_fp[:,:] = np.nan
            if S_flat[lightning_xy] == 1:
                S = S_flat.reshape(S.shape)
                landuseLayer_state4WF = landuseLayer_state4WF_flat.reshape(S.shape)
                S_clumps = measure.label(S, connectivity = 1)   # von Neumann neighbourhood
                clump_WF = S_clumps.flatten()[lightning_xy]
                indWF = S_clumps == clump_WF
                WF_fp[indWF] = 5                 # S = 5 for wildfire footprint in land use classes
                S[indWF] = 0                     # burned = no tree
                landuseLayer_state4WF[indWF] = 0

                if np.sum(WF_fp == 5) >= src['WF']['Smin_plot']:
                    plt.rcParams['font.size'] = '14'
                    fig, ax = plt.subplots(1, 1, figsize=(7,7))
                    ax.contourf(grid.xx, grid.yy, ls.hillshade(topoLayer_z, vert_exag=.1), cmap='gray', alpha = .1)
                    ax.pcolormesh(grid.xx, grid.yy, landuseLayer_state4WF, cmap = col_S, vmin=-1, vmax=5, alpha = .5)
                    ax.pcolormesh(grid.xx, grid.yy, WF_fp, cmap = col_S, vmin=-1, vmax=5)
                    plt.savefig(src['WF']['path'] + 'iter' + str(i) + '.jpg', dpi = 300)
                    plt.close()
                    WF_fp_saved = WF_fp

    return landuseLayer_state4WF, WF_fp_saved


## FLUVIAL FLOOD CASE ##
def run_CellularAutomaton_FF(I_RS, src, grid, topoLayer_z):
    A_catchment = (100e3 * 100e3)                # [m2] ad-hoc area east of the virtual region
    Qp = I_RS * A_catchment                      # [m3/s]
    river_xi, river_yi, _, _ = calc_coord_river_dampedsine(grid, src['FF'])
    src_indx = np.where(grid.x > river_xi[-1] - 1e-6)[0][0]
    src_indy = np.where(grid.y > river_yi[-1] - 1e-6)[0][0]
    
    l_src_max = Qp / (2 *(grid.w * 1e3)**2)     # [m/s]
    tmax = src['RS']['duration'] * 3600         # [s]
    
    # make offshore mask
    mask_offshore = np.zeros((grid.nx, grid.ny), dtype = bool)
    for j in range(grid.ny):
        mask_offshore[grid.x >= src['SS']['x'][j], j] = True
    
    FFfootprint_t = np.zeros((grid.nx,grid.ny))    # flood footprint at time t
    FFfootprint = np.zeros((grid.nx,grid.ny))      # static flood footprint I(x,y) = max_t I(x,y,t)
    c = .5
    for t in range(tmax):
        if t%3600 == 0:
            print(t/3600, 'hr. /', tmax/3600)
    
        # water flowing at source grid cells (channel composed of two cells)
        FFfootprint_t[src_indx,src_indy] = l_src_max      # should consider a hydrograph (with gradual increase to hW_src_max instead)
        FFfootprint_t[src_indx,src_indy-1] = l_src_max
    
        l = topoLayer_z + FFfootprint_t
    
        dl_a = np.pad((l[:,1:] - l[:,:-1]), [(0, 0), (1, 0)]) # left     see fig. 4.5 of the textbook
        dl_b = np.pad((l[:-1,:] - l[1:,:]), [(0, 1), (0, 0)]) # bottom
        dl_c = np.pad((l[:,:-1] - l[:,1:]), [(0, 0), (0, 1)]) # right
        dl_d = np.pad((l[1:,:] - l[:-1,:]), [(1, 0), (0, 0)]) # top
    
        dl_all = np.stack((dl_a, dl_b, dl_c, dl_d))
        dl_all[dl_all < 0] = 0                        # water only going down, 0 for sum below
        dl_sum = np.sum(dl_all, axis = 0)
        dl_sum[dl_sum == 0] = np.inf                  # inf for 0 weight below
        weight_dl = dl_all / dl_sum
    
        lmax = np.minimum(FFfootprint_t, np.amax(c * dl_all, axis = 0))
        lmov_all = weight_dl * lmax
    
        lIN_a = np.pad(lmov_all[0,:,1:], [(0, 0), (0, 1)])
        lIN_b = np.pad(lmov_all[1,:-1,:], [(1, 0), (0, 0)])
        lIN_c = np.pad(lmov_all[2,:,:-1], [(0, 0), (1, 0)])
        lIN_d = np.pad(lmov_all[3,1:,:], [(0, 1), (0, 0)])
        lOUT_0 = np.sum(lmov_all, axis = 0)
    
        FFfootprint_t = FFfootprint_t + (lIN_a + lIN_b + lIN_c + lIN_d) - lOUT_0    # simplified eq. 4.6 of textbook
        FFfootprint_t = FFfootprint_t * mask_offshore
        FFfootprint = np.maximum(FFfootprint, FFfootprint_t)
        
    return FFfootprint


####################
## RISK FUNCTIONS ##
####################
def f_D(I, peril):
    if peril == 'AI' or peril == 'Ex':   # I = overpressure [kPa]
        mu = np.log(20)
        sig = .4
        MDR = .5 * (1 + scipy.special.erf((np.log(I) - mu)/(sig * np.sqrt(2))))
    if peril == 'EQ':                    # I = peak ground acceleration [m/s2]
        mu = np.log(6)
        sig = .6
        MDR = .5 * (1 + scipy.special.erf((np.log(I) - mu)/(sig * np.sqrt(2))))
    if peril == 'FF' or peril == 'SS':   # I = inundation depth [m]
        c = .45
        MDR = c * np.sqrt(I)
        MDR[MDR > 1] = 1
    if peril == 'LS':                    # I = landslide thickness [m]
        c1 = -1.671
        c2 = 3.189
        c3 = 1.746
        MDR = 1 - np.exp(c1*((I+c2)/c2 - 1)**c3)
    if peril == 'VE':                    # I = ash thickness [m]
        g_earth = 9.81                   # [m/s^2]
        rho_ash = 900                    # [kg/m3]  (dry ash)
        I_kPa = rho_ash * g_earth * I * 1e-3
        mu = 1.6
        sig = .4
        MDR = .5 * (1 + scipy.special.erf((np.log(I_kPa) - mu)/(sig * np.sqrt(2))))
    if peril == 'WS' or peril == 'TC':   # I = wind speed [m/s]
        v_thresh = 25.7 # 50 kts
        v_half = 74.7
        vn = (I - v_thresh) / (v_half - v_thresh)
        vn[vn < 0] = 0
        MDR = vn**3 / (1+vn**3)
    return MDR


def gen_lossFootprints(catalog_hazFootprints, expoFootprint, stochset):
    ELT = pd.DataFrame(columns = ['ID', 'evID', 'lbd', 'L'])
    catalog_MDR = {}
    catalog_lossFootprints = {}
    n_fp = len(catalog_hazFootprints)
    evIDs = np.array(list(catalog_hazFootprints.keys()))
    perils = get_peril_evID(evIDs)
    print('generating footprints for:')
    peril_check = ''
    delete_TC = False
    for i in range(n_fp):
        evID = evIDs[i]
        peril = perils[i]
        if peril != peril_check:
            print(peril)
            peril_check = peril

        hazFootprint = catalog_hazFootprints[evID]
        if evIDs[i][2] != '_':
            # primary events
            evID_trigger = evID
            MDR = f_D(hazFootprint, peril)
            lossFootprint = MDR * expoFootprint
            Ltot = np.nansum(lossFootprint)
        else:
            # secondary events
            if peril == 'LS':
                MDR = f_D(hazFootprint, peril)
                evID_trigger = evID[7:]               # according to secondary ID format ID_fromIDx
                peril = 'RS+LS'
                evID = evID_trigger + '+LS'
                lossFootprint = MDR * expoFootprint   # no direct RS loss (i.e., invisible event)
                Ltot = np.nansum(lossFootprint)
            if peril == 'SS':
                MDR = f_D(hazFootprint, peril)
                # combine TC+SS
                evID_trigger = evID[7:]               # according to secondary ID format ID_fromIDx
                peril = 'TC+SS'
                evID = evID_trigger + '+SS'
                lossFootprint_TC = catalog_lossFootprints[evID_trigger]
                lossFootprint = lossFootprint_TC + MDR * (expoFootprint - lossFootprint_TC)  # assumes SS only damages what is not yet damaged by TC
                Ltot = np.nansum(lossFootprint)
                delete_TC = True                      # delete TC from ELT since now combined in TC+SS

        catalog_MDR[evID] = MDR
        catalog_lossFootprints[evID] = lossFootprint
        ELT = pd.concat([ELT, pd.DataFrame({'ID': peril, 'evID': evID, 'lbd': stochset[stochset['evID'] == evID_trigger]['lbd'], 'L': Ltot})])

    if delete_TC:
        ELT = ELT[ELT['ID'] != 'TC']

    return catalog_lossFootprints, ELT


def gen_YLT(ELT, Nsim):
    # 1. calculate the overall ELT frequency
    lbd = np.sum(ELT['lbd'])
    
    # 2. simulate the number of events each year
    k = np.random.poisson(lbd, int(Nsim))
    
    # 3. define a simulation ID for each simulated event
    simIDs = []
    simID = 1
    for val in k:
        simIDs.append(np.repeat(simID, val))
        simID += 1
    simIDs = np.concatenate(simIDs)
    
    # 4. sample events according to the ELT rates
    n = np.sum(k)                       # tot. number of events
    u = np.random.random(n)             # random numbers for sampling
    EF_norm = ELT['EF'] / lbd           # normalised exceedance frequency
    IDs = [ELT['evID'][EF_norm > u[i]].iloc[0] for i in range(n)]

    # 5. use ELT as lookup table to add losses
    YLT = pd.DataFrame(data = {'simID': simIDs, 'evID': IDs})
    YLT = YLT.merge(ELT[['evID', 'L']], on='evID', how='left')
    
    return YLT


def calc_EP(lbd):
    nev = len(lbd)
    EFi = np.zeros(nev)
    for i in range(nev):
        EFi[i] = np.sum(lbd[0:i+1])
    EPi = 1 - np.exp(- EFi)            # Eq. 3.22
    return EFi, EPi

def calc_riskmetrics_fromELT(ELT, q_VAR):
    AAL = np.sum(ELT['lbd'] * ELT['L'])                   # Eq. 3.18
    ELT = ELT.sort_values(by = 'L', ascending = False)    # losses in descending order
    EFi, EPi = calc_EP(ELT['lbd'].values)
    ELT['EF'], ELT['EP'] = [EFi, EPi]
    # VaR_q and TVaR_q
    p = 1 - q_VAR
    ELT_asc = ELT.sort_values(by = 'L')                    # losses in ascending order
    VaRq = ELT_asc['L'][ELT_asc['EP'] < p].iloc[0]         # Eq. 3.23
    TVaRq = np.sum(ELT_asc['L'][ELT_asc['L'] > VaRq]) / len(ELT_asc['L'][ELT_asc['L'] > VaRq])   # derived from Eq. 3.24

    L_hires = 10**np.linspace(np.log10(ELT_asc['L'].min()+1e-6), np.log10(ELT_asc['L'].max()), num = 1000)
    EP_hires = np.interp(L_hires, ELT_asc['L'], ELT_asc['EP'])
    VaRq_interp = L_hires[EP_hires < p][0]
    TVaRq_interp = np.sum(L_hires[L_hires > VaRq_interp]) / len(L_hires[L_hires > VaRq_interp])

    return ELT, AAL, VaRq_interp, TVaRq_interp, VaRq, TVaRq

def calc_riskmetrics_fromYLT(YLT, Nsim, q_VAR):
    AAL = np.sum(YLT['L']) / Nsim                                                   # Eq. 3.19
    YLT_asc = YLT.sort_values(by = 'L')
    # VaR_q and TVaR_q
    n = len(YLT)
    VaRq = YLT_asc['L'].iloc[int(q_VAR*n+1)]                                        # Sec. 3.3.2.3
    TVaRq = 1/(n - (q_VAR*n+1) + 1) * np.sum(YLT_asc['L'].iloc[int(q_VAR*n+1):])    # Eq. 3.25
    return AAL, VaRq, TVaRq

def calc_EPfromYLT(YLT, Nsim):
    '''
    '''
    EPi = (Nsim - np.arange(Nsim)) / Nsim
    simIDs = np.unique(YLT['simID'])
    n = len(simIDs)
    Lmax = [np.max(YLT['L'][YLT['simID'] == simID]) for simID in simIDs]
    Lagg = [np.sum(YLT['L'][YLT['simID'] == simID]) for simID in simIDs]
    Li_max = np.concatenate((np.zeros(int(Nsim) - n), np.sort(Lmax)))
    Li_agg = np.concatenate((np.zeros(int(Nsim) - n), np.sort(Lagg)))
    return [Li_max, Li_agg, EPi]


##############
## PLOTTING ##
##############
def col_peril(peril):
    col_peril_extra = '#663399'    # Rebeccapurple
    col_peril_geophys = "#CD853F"  # Peru
    col_peril_hydro = "#20B2AA"    # MediumSeaGreen
    col_peril_meteo = "#4169E1"    # RoyalBlue
    col_peril_clim = '#8B0000'     # DarkRed
    col_peril_tech = '#708090'     # SlateGrey
    if peril == 'AI':
        col = col_peril_extra
    if peril == 'EQ' or peril == 'LS' or peril == 'VE':
        col = col_peril_geophys
    if peril == 'FF' or peril == 'SS':
        col = col_peril_hydro
    if peril == 'RS' or peril == 'WS' or peril == 'TC':
        col = col_peril_meteo
    if peril == 'WF':
        col = col_peril_clim
    if peril == 'Ex':
        col = col_peril_tech
    return col

ls = plt_col.LightSource(azdeg = 45, altdeg = 45)

# terrain color scheme
n_water, n_land = [50,200]
col_water = plt.cm.terrain(np.linspace(0, 0.17, n_water))
col_land = plt.cm.terrain(np.linspace(0.25, 1, n_land))
col_terrain = np.vstack((col_water, col_land))
cmap_z = plt_col.LinearSegmentedColormap.from_list('cmap_z', col_terrain)
class norm_z(plt_col.Normalize):
    # from https://stackoverflow.com/questions/40895021/python-equivalent-for-matlabs-demcmap-elevation-appropriate-colormap
    # col_val = n_water/(n_water+n_land)
    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val = 0.2, clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent the sealevel
        self.col_val = col_val
        plt_col.Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# land use state color scheme
colors = [(0/255.,127/255.,191/255.),        # -1 - water mask
          (236/255., 235/255., 189/255.),    # 0 - grassland (fall green)
          (34/255.,139/255.,34/255.),        # 1 - forest
          (131/255.,137/255.,150/255.),      # 2 - built, residential
          (10/255.,10/255.,10/255.),         # 3 - built, industry
          (230/255.,230/255.,230/255.),      # 4 - built, commercial
          (255/255.,0/255.,0/255.)]           # 5 - wildfire
col_S = plt_col.LinearSegmentedColormap.from_list('col_S', colors, N = 7)

# soil color scheme
def col_state_h(h, h0):
    h_plot = np.copy(h)
    h_plot[h == 0] = 0                               # erosion +++ (scarp)
    h_plot[h == h0] = 1                              # intact
    h_plot[np.logical_and(h > 0, h <= h0/2)] = 2     # erosion ++
    h_plot[np.logical_and(h > h0/2, h < h0)] = 3     # erosion +
    h_plot[np.logical_and(h > h0, h <= 2*h0)] = 4    # landslide +
    h_plot[h > 2*h0] = 5                             # landslide ++
    return h_plot
colors = [(105/255,105/255,105/255),        # 0 - scarp / erosion +++ (dimgrey)
          (236/255,235/255,189/255),        # 1 - intact (fall green)
          (195/255,176/255,145/255),        # 2 - erosion ++ (khaki)
          (186/255,135/255,89/255),         # 3 - erosion + (deer)
          (155/255,118/255,83/255),         # 4 - landslide + (dirt)
          (131/255,105/255,83/255)]         # 5 - landslide ++ (pastel brown)
col_h = plt_col.LinearSegmentedColormap.from_list('col_h', colors, N = 6)


def plot_envLayers(grid, src, topoLayer_z, soilLayer_hs, landuseLayer_S, roadNetwork):
    plt.rcParams['font.size'] = '18'
    fig, ax = plt.subplots(2, 2, figsize=(20, 16))
    ax[0,0].contourf(grid.xx, grid.yy, ls.hillshade(topoLayer_z, vert_exag=.1), cmap='gray', alpha = .1)
    if 'EQ' in src['perils']:
        for src_i in range(len(src['EQ']['x'])):
            if src_i == 1:
                ax[0,0].plot(src['EQ']['x'][src_i], src['EQ']['y'][src_i], color = col_peril('EQ'), label = 'faults (EQ)')
            else:
                ax[0,0].plot(src['EQ']['x'][src_i], src['EQ']['y'][src_i], color = col_peril('EQ'))
    if 'FF' in src['perils']:
        river_xi, river_yi, _, river_id = calc_coord_river_dampedsine(grid, src['FF'])
        ax[0,0].plot(river_xi, river_yi, color = col_peril('FF'), label = 'river bed')
        ax[0,0].scatter(np.max(river_xi), src['FF']['riv_y0'][0], s=100, marker = 's', color = col_peril('FF'), label = 'upstream point (FF)')
    if 'VE' in src['perils']:
        ax[0,0].scatter(src['VE']['x'], src['VE']['x'], color = col_peril('VE'), s=100, marker = '^', label = 'volcano (VE)')
    ax[0,0].plot([grid.xmin + grid.xbuffer, grid.xmax - grid.xbuffer, grid.xmax - grid.xbuffer, \
              grid.xmin + grid.xbuffer, grid.xmin + grid.xbuffer],
             [grid.ymin + grid.ybuffer, grid.ymin + grid.ybuffer, grid.ymax - grid.ybuffer, \
              grid.ymax - grid.ybuffer, grid.ymin + grid.ybuffer], linestyle = 'dotted', color = 'black')
    ax[0,0].set_xlim(grid.xmin, grid.xmax)
    ax[0,0].set_ylim(grid.ymin, grid.ymax)
    ax[0,0].set_xlabel('x [km]')
    ax[0,0].set_ylabel('y [km]')
    ax[0,0].set_title('Peril source coordinates', pad = 20)
    ax[0,0].legend(loc = 'upper left', fontsize = 16)
    ax[0,0].set_aspect(1)

    plt_zmin_m = -500
    plt_zmax_m = 4500
    ax[0,1].contourf(grid.xx, grid.yy, ls.hillshade(topoLayer_z, vert_exag=.1), cmap='gray')
    z_plot = np.copy(topoLayer_z)
    z_plot[z_plot < plt_zmin_m] = plt_zmin_m
    z_plot[z_plot > plt_zmax_m] = plt_zmax_m
    img = ax[0,1].contourf(grid.xx, grid.yy, z_plot, norm = norm_z(sealevel = 0, vmax = plt_zmax_m), \
                        cmap = cmap_z, levels = np.arange(plt_zmin_m, plt_zmax_m+100, 100), alpha = .8)
    fig.colorbar(img, ax = ax[0,1], fraction = .04, pad = .04, label = 'z [m]')
    ax[0,1].plot([grid.xmin + grid.xbuffer, grid.xmax - grid.xbuffer, grid.xmax - grid.xbuffer, \
              grid.xmin + grid.xbuffer, grid.xmin + grid.xbuffer],
             [grid.ymin + grid.ybuffer, grid.ymin + grid.ybuffer, grid.ymax - grid.ybuffer, \
              grid.ymax - grid.ybuffer, grid.ymin + grid.ybuffer], linestyle = 'dotted', color = 'black')
    ax[0,1].set_xlim(grid.xmin, grid.xmax)
    ax[0,1].set_ylim(grid.ymin, grid.ymax)
    ax[0,1].set_xlabel('x [km]')
    ax[0,1].set_ylabel('y [km]')
    ax[0,1].set_title('Topography', pad = 20)
    ax[0,1].set_aspect(1)

    legend_h = [Patch(facecolor=(105/255,105/255,105/255, .5), edgecolor='black', label='h=0 (scarp)'),
                Patch(facecolor=(216/255,228/255,188/255, .5), edgecolor='black', label='h=$h_0$ (soil)')]
    h0_m = 10
    h_state = col_state_h(soilLayer_hs, h0_m)

    ax[1,0].contourf(grid.xx, grid.yy, ls.hillshade(topoLayer_z, vert_exag=.1), cmap='gray')
    ax[1,0].pcolormesh(grid.xx, grid.yy, h_state, cmap = col_h, vmin=0, vmax=5, alpha = .5)
    ax[1,0].plot([grid.xmin + grid.xbuffer, grid.xmax - grid.xbuffer, grid.xmax - grid.xbuffer, \
              grid.xmin + grid.xbuffer, grid.xmin + grid.xbuffer],
             [grid.ymin + grid.ybuffer, grid.ymin + grid.ybuffer, grid.ymax - grid.ybuffer, \
              grid.ymax - grid.ybuffer, grid.ymin + grid.ybuffer], linestyle = 'dotted', color = 'black')
    ax[1,0].set_xlim(grid.xmin, grid.xmax)
    ax[1,0].set_ylim(grid.ymin, grid.ymax)
    ax[1,0].set_xlabel('x [km]')
    ax[1,0].set_ylabel('y [km]')
    ax[1,0].set_title('Soil depth', pad = 20)
    ax[1,0].set_aspect(1)
    ax[1,0].legend(handles=legend_h, loc='upper left', fontsize = 16)

    legend_S = [Patch(facecolor=(0/255.,127/255.,191/255.,.5), edgecolor='black', label='water ($\mathcal{S}$ = -1)'),
                Patch(facecolor=(236/255., 235/255., 189/255.,.5), edgecolor='black', label='grassland ($\mathcal{S}$ = 0)'),
                Patch(facecolor=(34/255.,139/255.,34/255.,.5), edgecolor='black', label='forest ($\mathcal{S}$ = 1)'),
                Patch(facecolor=(131/255.,137/255.,150/255.,.5), edgecolor='black', label='residential ($\mathcal{S}$ = 2)'),
                Patch(facecolor=(10/255.,10/255.,10/255.,.5), edgecolor='black', label='industrial ($\mathcal{S}$ = 3)'),
                Patch(facecolor=(230/255.,230/255.,230/255.,.5), edgecolor='black', label='commercial ($\mathcal{S}$ = 4)'),
                Line2D([0], [0], color='yellow', linewidth=1, label='road network')]
    ax[1,1].contourf(grid.xx, grid.yy, ls.hillshade(topoLayer_z, vert_exag=.1), cmap='gray')
    ax[1,1].pcolormesh(grid.xx, grid.yy, landuseLayer_S, cmap = col_S, vmin=-1, vmax=5, alpha = .5)
    ax[1,1].plot(roadNetwork[0], roadNetwork[1], color='yellow', lw = .5)
    ax[1,1].plot([grid.xmin + grid.xbuffer, grid.xmax - grid.xbuffer, grid.xmax - grid.xbuffer, \
              grid.xmin + grid.xbuffer, grid.xmin + grid.xbuffer],
             [grid.ymin + grid.ybuffer, grid.ymin + grid.ybuffer, grid.ymax - grid.ybuffer, \
              grid.ymax - grid.ybuffer, grid.ymin + grid.ybuffer], linestyle = 'dotted', color = 'black')
    ax[1,1].set_xlim(grid.xmin, grid.xmax)
    ax[1,1].set_ylim(grid.ymin, grid.ymax)
    ax[1,1].set_xlabel('x [km]')
    ax[1,1].set_ylabel('y [km]')
    ax[1,1].set_title('Land use', pad = 20)
    ax[1,1].set_aspect(1)
    ax[1,1].legend(handles=legend_S, loc='upper right', fontsize = 16)
    plt.tight_layout()
    plt.savefig(wd + '/figures/envLayers.jpg', dpi = 300)
    plt.pause(1)
    plt.show()


def plot_hazFootprints(catalog_hazFootprints, grid, src, topoLayer_z, plot_Imax, nstoch = 5):
    evIDs = np.array(list(catalog_hazFootprints.keys()))
    ev_peril = get_peril_evID(evIDs)
    perils = np.unique(ev_peril)
    nperil = len(perils)

    plt.rcParams['font.size'] = '18'
    fig, ax = plt.subplots(nperil, nstoch, figsize=(20, nperil*20/nstoch))

    for i in range(nperil):
        indperil = np.where(ev_peril == perils[i])[0]
        nev = len(indperil)
        nplot = np.min([nstoch, nev])
        evID_shuffled = evIDs[indperil]
        if nev > nplot:
            np.random.shuffle(evID_shuffled)
        Imax = plot_Imax[perils[i]]
        for j in range(nplot):
            I_plt = np.copy(catalog_hazFootprints[evID_shuffled[j]])
            I_plt[I_plt >= Imax] = Imax
            ax[i,j].contourf(grid.xx, grid.yy, I_plt, cmap = 'Reds', levels = np.linspace(0, Imax, 100))
            ax[i,j].contourf(grid.xx, grid.yy, ls.hillshade(topoLayer_z, vert_exag=.1), cmap='gray', alpha = .1)
            ax[i,j].set_xlim(grid.xmin, grid.xmax)
            ax[i,j].set_ylim(grid.ymin, grid.ymax)
            ax[i,j].set_xlabel('x [km]')
            ax[i,j].set_ylabel('y [km]')
            ax[i,j].set_title(evID_shuffled[j], pad = 10)
            ax[i,j].set_aspect(1)
        if nplot < nstoch:
            for j in np.arange(nplot, nstoch):
                ax[i,j].set_axis_off()
    plt.tight_layout()    
    plt.savefig(wd + '/figures/hazFootprints.jpg', dpi = 300)
    plt.pause(1)
    plt.show()


def plot_lossFootprints(catalog_lossFootprints, ELT, grid, topoLayer_z, Lmax, nstoch = 5):
    evIDs = ELT['evID'].values
    ev_peril = ELT['ID'].values
    perils = np.unique(ev_peril)
    nperil = len(perils)

    plt.rcParams['font.size'] = '18'
    fig, ax = plt.subplots(nperil, nstoch, figsize=(20, nperil*20/nstoch))

    for i in range(nperil):
        indperil = np.where(ev_peril == perils[i])[0]
        nev = len(indperil)
        nplot = np.min([nstoch, nev])
        evID_shuffled = evIDs[indperil]
        if nev > nplot:
            np.random.shuffle(evID_shuffled)
        for j in range(nplot):
            L_plt = np.copy(catalog_lossFootprints[evID_shuffled[j]])
            L_plt[L_plt >= Lmax] = Lmax
            ax[i,j].contourf(grid.xx, grid.yy, L_plt, cmap = 'Reds', levels = np.linspace(0, Lmax, 100))
            ax[i,j].contourf(grid.xx, grid.yy, ls.hillshade(topoLayer_z, vert_exag=.1), cmap='gray', alpha = .1)
            ax[i,j].set_xlim(grid.xmin, grid.xmax)
            ax[i,j].set_ylim(grid.ymin, grid.ymax)
            ax[i,j].set_xlabel('x [km]')
            ax[i,j].set_ylabel('y [km]')
            ax[i,j].set_title(evID_shuffled[j], pad = 10)
            ax[i,j].set_aspect(1)
        if nplot < nstoch:
            for j in np.arange(nplot, nstoch):
                ax[i,j].set_axis_off()
    plt.tight_layout()    
    plt.savefig(wd + '/figures/lossFootprints.jpg', dpi = 300)
    plt.pause(1)
    plt.show()


def plot_vulnFunctions():
    pi_kPa = np.linspace(0, 50, 100)
    MDR_blast = f_D(pi_kPa, 'AI')      # or 'Ex'
    PGAi = np.linspace(0, 15, 100)     # m/s2
    MDR_EQ = f_D(PGAi, 'EQ')
    hwi = np.linspace(0, 7, 1000)      # m
    MDR_flood = f_D(hwi, 'FF')         # or 'SS'
    hsi = np.linspace(0, 7, 100)       # m
    MDR_LS = f_D(hsi, 'LS')
    hai = np.linspace(0, 2, 100)       # m
    MDR_VE = f_D(hai, 'VE')
    g_earth = 9.81                   # [m/s^2]
    rho_ash = 900                    # [kg/m3]  (dry ash)
    pi_VE_kPa = rho_ash * g_earth * hai * 1e-3
    vi = np.linspace(0, 100, 100)      # m/s
    MDR_WS = f_D(vi, 'WS')

    plt.rcParams['font.size'] = '18'
    fig, ax = plt.subplots(2,3, figsize = (20,12))
    ax[0,0].plot(pi_kPa, MDR_blast, color = 'black')
    ax[0,0].set_title('Blast (AI, Ex)', pad = 20)
    ax[0,0].set_xlabel('Overpressure $P$ [kPa]')
    ax[0,0].set_ylabel('MDR')
    ax[0,0].set_ylim(0,1.01)
    ax[0,0].spines['right'].set_visible(False)
    ax[0,0].spines['top'].set_visible(False)

    ax[0,1].plot(PGAi, MDR_EQ, color = 'black')
    ax[0,1].set_title('Earthquake (EQ)', pad = 20)
    ax[0,1].set_xlabel('PGA [m/s$^2$]')
    ax[0,1].set_ylabel('MDR')
    ax[0,1].set_ylim(0,1.01)
    ax[0,1].spines['right'].set_visible(False)
    ax[0,1].spines['top'].set_visible(False)

    ax[0,2].plot(hwi, MDR_flood, color = 'black')
    ax[0,2].set_title('Flooding (FF, SS)', pad = 20)
    ax[0,2].set_xlabel('Inundation depth $h$ [m]')
    ax[0,2].set_ylabel('MDR')
    ax[0,2].set_ylim(0,1.01)
    ax[0,2].spines['right'].set_visible(False)
    ax[0,2].spines['top'].set_visible(False)

    ax[1,0].plot(hsi, MDR_LS, color = 'black')
    ax[1,0].set_title('Landslide (LS)', pad = 20)
    ax[1,0].set_xlabel('Deposited height $h$ [m]')
    ax[1,0].set_ylabel('MDR')
    ax[1,0].set_ylim(0,1.01)
    ax[1,0].spines['right'].set_visible(False)
    ax[1,0].spines['top'].set_visible(False)

    ax[1,1].plot(pi_VE_kPa, MDR_VE, color = 'black')
    ax[1,1].set_title('Volcanic eruption (VE)', pad = 20)
    ax[1,1].set_xlabel('Ash load $P$ [kPa]')
    ax[1,1].set_ylabel('MDR')
    ax[1,1].set_ylim(0,1.01)
    ax[1,1].spines['right'].set_visible(False)
    ax[1,1].spines['top'].set_visible(False)
    ax2 = ax[1,1].twiny()
    ax2.set_xlabel('Ash thickness $h$ [m]')
    ax2.plot(hai, MDR_VE, color = 'white', alpha = 0)
    ax2.spines['right'].set_visible(False)

    ax[1,2].plot(vi, MDR_WS, color = 'black')
    ax[1,2].set_title('Windstorm (WS)', pad = 20)
    ax[1,2].set_xlabel('Maximum wind speed $v_{max}$ [m/s]')
    ax[1,2].set_ylabel('MDR')
    ax[1,2].set_ylim(0,1.01)
    ax[1,2].spines['right'].set_visible(False)
    ax[1,2].spines['top'].set_visible(False)

    fig.tight_layout()
    plt.savefig(wd + '/figures/vulnFunctions.jpg', dpi = 300)
    plt.pause(1)
    plt.show()


## landslide plotting ##
def get_FS_state(FS_value):
        FS_code = np.copy(FS_value)
        FS_code[FS_value > 1.5] = 0                                 # stable
        FS_code[np.logical_and(FS_value > 1, FS_value <= 1.5)] = 1  # critical
        FS_code[FS_value <= 1] = 2                                  # unstable
        return FS_code

colors = [(0, 100/255., 0),                  # 0 - stable FS (dark green)
          (255/255.,215/255.,0/255.),        # 1 - critical FS (gold)
          (178/255.,34/255.,34/255.)]        # 2 - unstable FS (darkred)
col_FS = plt_col.LinearSegmentedColormap.from_list('col_FS', colors, N = 3)

legend_FS = [Patch(facecolor=(0, 100/255., 0, .5), edgecolor='black', label='>1.5 (stable)'),
             Patch(facecolor=(255/255.,215/255.,0/255., .5), edgecolor='black',label='[1,1.5] (critical)'),
             Patch(facecolor=(178/255.,34/255.,34/255., .5), edgecolor='black', label='<1 (unstable)')]

def h_code(h, h0):
    h_plot = np.copy(h)
    h_plot[h == 0] = 1                               # erosion +++ (scarp)
    h_plot[h == h0] = 2                              # intact
    h_plot[np.logical_and(h > 0, h <= h0/2)] = 3     # erosion ++
    h_plot[np.logical_and(h > h0/2, h < h0)] = 4     # erosion +
    h_plot[np.logical_and(h > h0, h <= 2*h0)] = 5    # landslide +
    h_plot[h > 2*h0] = 6                             # landslide ++
    return h_plot

#dimgrey, gin, khaki, deer, dirt, pastel brown
colors = [(105/255,105/255,105/255), 
          (216/255,228/255,188/255),
          (195/255,176/255,145/255),
          (186/255,135/255,89/255),
          (155/255,118/255,83/255),
          (131/255,105/255,83/255)]
h_col = plt_col.LinearSegmentedColormap.from_list('h_col', colors, N=6)
