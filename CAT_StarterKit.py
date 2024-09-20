# Title: CAT Risk Modelling Starter-Kit (in Python)
# Author: Arnaud Mignan
# Date: 01.02.2024
# Description: A basic template to develop a catastrophe (CAT) risk model (here with ad-hoc parameters & models).
# License: MIT
# Version: 1.1
# Dependencies: numpy, pandas, scipy, matplotlib
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


import numpy as np
import pandas as pd
from scipy.special import erf
import matplotlib.pyplot as plt


# ad-hoc functions
def func_src2ev(src_S):
    '''
    The maximum size S of an event (ev_Smax) is constrained by the source from which it originates, 
    with src_S the source characteristic from which ev_Smax can be inferred - See model examples in Section 2.2.
    '''
    c = .1
    k = 2
    ev_Smax = c * src_S**k     # ad-hoc model
    return ev_Smax             # maximum-size event (or characteristic event for the source)

def func_intensity(S, r):
    '''
    The impact of each event on the environment is assessed by the hazard intensity I(x,y) across a 
    geographical grid (x,y). A static footprint is defined as function of distance from source r(x,y)
    and event size S - See model examples in Section 2.4.
    '''
    c0 = -1
    c1 = .1
    c2 = 1
    c3 = 5
    I = np.exp(c0 + c1 * S - c2 * np.log(r+c3))  # ad-hoc model
    return I

def func_vulnerability(I, Theta_nu):
    '''
    The vulnerability curve expresses the damage D (or Mean Damage Ratio, MDR) expected on an asset of 
    characteristics Theta_nu given a hazard intensity load I - See model examples in Section 3.2.5.
    '''
    MDR = 1/2 + 1/2 * erf((np.log(I) - Theta_nu['mu']) / np.sqrt(2 * Theta_nu['sigma']**2))   # here cum. lognormal distr.
    return MDR


# ad-hoc parameters (to be replaced by peril- & region-specific values)
xmin, xmax, dx = [0, 100, 1]              # [km]
ymin, ymax, dy = [0, 100, 1]              # [km]
src1_x0, src1_y0, src1_S = [25, 25, 5]    # source 'src1' of coordinates (x0,y0) and size S
src2_x0, src2_y0, src2_S = [75, 50, 8]    #
Smin = .1                                 # minimum event size [ad hoc unit] (e.g., energy)
dS_log10 = .1                             # event size increment (in log10 scale)
a, b = [.5, 1]                            # for Eq. 2.38: log10(rate_cum(S)) = a - b log10(S); see some values in Tab. 2.5
Theta_nu = {'mu':np.log(.04), 'sigma':.1} # for Eq. 3.4: cum. lognormal distr.; see some values in Section 3.2.5

# define environment (i.e., grid for footprints)
x = np.arange(xmin, xmax, dx)
y = np.arange(ymin, ymax, dy)
grid_x, grid_y = np.meshgrid(x, y)

# exposure footprint defined below as a square town of uniform asset value 1
padW, padE, padN, padS = [30, 25, 40, 20]
padding = ((padS, padN),(padW, padE))
nu_grid = np.ones((len(x)-(padN+padS), len(y)-(padW+padE)))
nu_grid = np.pad(nu_grid, padding)


## HAZARD ASSESSMENT ##
# define source model (here, 2 point sources)
Src = pd.DataFrame({'ID': ['src1', 'src2'], 'x0': [src1_x0, src2_x0], 'y0': [src1_y0, src2_y0], 'S': [src1_S, src2_S]})


# define size distribution
Smax1 = func_src2ev(src1_S)
Smax2 = func_src2ev(src2_S)
Smax = np.max([Smax1, Smax2])
Si = 10**np.arange(np.log10(Smin), np.log10(Smax), dS_log10)
Si1 = 10**np.arange(np.log10(Smin), np.log10(Smax1), dS_log10)
Si2 = 10**np.arange(np.log10(Smin), np.log10(Smax2), dS_log10)
N1, N2 = [len(Si1), len(Si2)]
ratei = 10**(a - b * (np.log10(Si) - dS_log10 / 2)) - 10**(a - b * (np.log10(Si) + dS_log10 / 2))  # e.g., Eq. 2.65


# define event table
# peril ID: e.g., EQ, VE, AI... Tab. 1.7
EventTable = pd.DataFrame({'ID': ['ID' + str(i+1) for i in range(N1+N2)],\
                          'Src': np.concatenate([np.repeat('src1', N1), np.repeat('src2', N2)]),\
                          'S': np.concatenate([Si1, Si2]),\
                          'rate': np.concatenate([ratei[0:N1], ratei[0:N2]])})

# correct rate, which is function of the stochastic set definition:
# in the present case, we have two sources with equal share of the overall event activity defined by (a,b)
Nevent_perSi = np.concatenate([np.repeat(2, 2 * N1), np.repeat(1, N2-N1)]) # if N1 < N2 (i.e., src1_S < src2_S)
EventTable['rate'] = EventTable['rate'] / Nevent_perSi
# Whichever stochastic construct, we must have EventTable['rate'].sum() = np.sum(ratei)


# define intensity I grid footprint catalog
I_grid = np.zeros((N1+N2, len(x), len(y)))
for i in range(N1+N2):
    ind = np.where(Src['ID'] == EventTable['Src'][i])[0][0]
    r_grid = np.sqrt((grid_x - Src['x0'][ind])**2 + (grid_y - Src['y0'][ind])**2)
    I_grid[i,:,:] = func_intensity(EventTable['S'][i], r_grid)

# -> calculate hazard metrics (see solution to exercise #2.4)


## RISK ASSESSMENT ##
# calculate damage D grid & loss L grid footprints
D_grid = np.zeros((N1+N2, len(x), len(y)))
L_grid = np.zeros((N1+N2, len(x), len(y)))
for i in range(N1+N2):
    D_grid[i,:,:] = func_vulnerability(I_grid[i,:,:], Theta_nu)
    L_grid[i,:,:] = D_grid[i,:,:] * nu_grid

# update event table as loss table
ELT = EventTable
ELT['loss'] = [np.sum(L_grid[i,:,:]) for i in range(N1+N2)]

# -> calculate risk metrics (see solution to exercise #3.2)




## plot templates ##
fig, ax = plt.subplots(3,4, figsize = (20,15))
src_Si = np.arange(.1,10)
ax[0,0].plot(src_Si, func_src2ev(src_Si), color = 'black')
ax[0,0].axhline(Smin, color = 'black', linestyle = 'dashed')
ax[0,0].axvline(src1_S, color = 'orange', linestyle = 'dashed')
ax[0,0].axvline(src2_S, color = 'red', linestyle = 'dashed')
ax[0,0].set_xlabel('Source size src_S')
ax[0,0].set_ylabel('Max. event size Smax')
ax[0,0].set_title('Characteristic event size')

ax[0,1].plot(Si, ratei, color = 'black')
ind1 = np.where(EventTable['Src'] == 'src1')[0]
ax[0,1].scatter(EventTable['S'][ind1], EventTable['rate'][ind1], color = 'orange')
ind2 = np.where(EventTable['Src'] == 'src2')[0]
offset = 1.1 # to avoid overlap
ax[0,1].scatter(EventTable['S'][ind2], EventTable['rate'][ind2]*offset, color = 'red')
ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')
ax[0,1].set_xlabel('Event size S')
ax[0,1].set_ylabel('Rate')
ax[0,1].set_title('Stochastic set distribution')

ri = np.arange(0,50,.1)
ax[0,2].plot(ri, func_intensity(Smax1, ri), color = 'orange')
ax[0,2].plot(ri, func_intensity(Smax2, ri), color = 'red')
ax[0,2].set_xlabel('Distance from source r')
ax[0,2].set_ylabel('Intensity I')
ax[0,2].set_title('Hazard intensity model')

Ii = np.arange(.001,.1,.001)
ax[0,3].plot(Ii, func_vulnerability(Ii, Theta_nu), color = 'black')
ax[0,3].set_xlabel('Intensity I')
ax[0,3].set_ylabel('Mean damage ratio')
ax[0,3].set_title('Vulnerability curve')

ax[1,0].contourf(x, y, np.ma.masked_where(nu_grid == 0, nu_grid), cmap = 'Greens', vmin = 0, vmax = 1)
ax[1,0].set_xlabel('x')
ax[1,0].set_ylabel('y')
ax[1,0].set_title('Exposure footprint')

ax[1,1].contourf(x, y, I_grid[ind1[-1],:,:], cmap = 'Reds', vmin = 0, vmax = np.max(I_grid))
ax[1,1].scatter(src1_x0, src1_y0, color = 'black', marker = '+')
ax[1,1].set_xlabel('x')
ax[1,1].set_ylabel('y')
ax[1,1].set_title('Hazard intensity footprint (Smax1)')

ax[1,2].contourf(x, y, D_grid[ind1[-1],:,:], cmap = 'Blues', vmin = 0, vmax = 1)
ax[1,2].scatter(src1_x0, src1_y0, color = 'black', marker = '+')
ax[1,2].set_xlabel('x')
ax[1,2].set_ylabel('y')
ax[1,2].set_title('Expected damage footprint (Smax1)')

ax[1,3].contourf(x, y, np.ma.masked_where(L_grid[ind1[-1],:,:] == 0, L_grid[ind1[-1],:,:]), cmap = 'Purples', vmin = 0, vmax = 1)
ax[1,3].scatter(src1_x0, src1_y0, color = 'black', marker = '+')
ax[1,3].set_xlabel('x')
ax[1,3].set_ylabel('y')
ax[1,3].set_title('Loss footprint (Smax1)')

ax[2,0].contourf(x, y, np.ma.masked_where(nu_grid == 0, nu_grid), cmap = 'Greens', vmin = 0, vmax = 1)
ax[2,0].set_xlabel('x')
ax[2,0].set_ylabel('y')
ax[2,0].set_title('Exposure footprint')

ax[2,1].contourf(x, y, I_grid[ind2[-1],:,:], cmap = 'Reds', vmin = 0, vmax = np.max(I_grid))
ax[2,1].scatter(src2_x0, src2_y0, color = 'black', marker = '+')
ax[2,1].set_xlabel('x')
ax[2,1].set_ylabel('y')
ax[2,1].set_title('Hazard intensity footprint (Smax2)')

ax[2,2].contourf(x, y, D_grid[ind2[-1],:,:], cmap = 'Blues', vmin = 0, vmax = 1)
ax[2,2].scatter(src2_x0, src2_y0, color = 'black', marker = '+')
ax[2,2].set_xlabel('x')
ax[2,2].set_ylabel('y')
ax[2,2].set_title('Expected damage footprint (Smax2)')

ax[2,3].contourf(x, y, np.ma.masked_where(L_grid[ind2[-1],:,:] == 0, L_grid[ind2[-1],:,:]), cmap = 'Purples', vmin = 0, vmax = 1)
ax[2,3].scatter(src2_x0, src2_y0, color = 'black', marker = '+')
ax[2,3].set_xlabel('x')
ax[2,3].set_ylabel('y')
ax[2,3].set_title('Loss footprint (Smax2)')

fig.tight_layout()
plt.savefig('plots_template_Python.pdf') 